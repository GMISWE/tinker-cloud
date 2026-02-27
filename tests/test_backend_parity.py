"""
Backend parity tests — validate that Miles and NeMo RL produce equivalent
training behavior for identical inputs.

Tests verify:
- Loss values within 1% tolerance (T046)
- Gradient norm trends agree (T047)
- Logprob values within floating-point tolerance (T048)

NOTE: These tests require GPU hardware, a Ray cluster, and both Miles
and NeMo RL installed. They are designed for manual validation, not CI.

Usage:
    pytest tests/test_backend_parity.py -v -m parity --timeout=600
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

import pytest
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

PARITY_CONFIG = {
    "base_model": "meta-llama/Llama-3.1-8B",
    "num_gpus": 4,
    "seq_len": 256,
    "batch_size": 4,
    "response_len": 128,
    "num_training_steps": 5,
    "loss_tolerance_pct": 1.0,      # 1% relative tolerance
    "logprob_atol": 0.01,            # absolute tolerance for logprobs
    "lora_config": {"rank": 32},
}


# ---------------------------------------------------------------------------
# Synthetic data generation (deterministic for reproducibility)
# ---------------------------------------------------------------------------

def make_deterministic_batch(
    batch_size: int,
    seq_len: int,
    response_len: int,
    seed: int = 42,
) -> List[Dict]:
    """Create a deterministic batch of synthetic Tinker Datum dicts."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    prompt_len = seq_len - response_len
    batch = []

    for i in range(batch_size):
        tokens = torch.randint(0, 32000, (seq_len,), generator=gen, dtype=torch.long)
        loss_masks = torch.cat([
            torch.zeros(prompt_len, dtype=torch.float32),
            torch.ones(response_len, dtype=torch.float32),
        ])
        advantages = torch.randn(response_len, generator=gen, dtype=torch.float32) * 0.1
        log_probs = torch.randn(response_len, generator=gen, dtype=torch.float32) * 0.5 - 5.0
        rollout_log_probs = log_probs + torch.randn(response_len, generator=gen) * 0.01
        ref_log_probs = torch.randn(response_len, generator=gen, dtype=torch.float32) * 0.5 - 5.0

        batch.append({
            "tokens": tokens.tolist(),
            "loss_masks": loss_masks.tolist(),
            "advantages": advantages.tolist(),
            "log_probs": log_probs.tolist(),
            "rollout_log_probs": rollout_log_probs.tolist(),
            "ref_log_probs": ref_log_probs.tolist(),
        })

    return batch


# ---------------------------------------------------------------------------
# Fixtures: backend initialization
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    """Create a shared event loop for the module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def miles_backend():
    """Create a Miles backend instance."""
    from tinkercloud.training.backends.factory import BackendFactory
    return BackendFactory.create(backend_type="miles")


@pytest.fixture(scope="module")
def nemo_rl_backend():
    """Create a NeMo RL backend instance."""
    from tinkercloud.training.backends.factory import BackendFactory
    return BackendFactory.create(backend_type="nemo_rl")


@pytest.fixture(scope="module")
def miles_handle(miles_backend, event_loop):
    """Create a Miles model handle."""
    handle = event_loop.run_until_complete(
        miles_backend.create_model(
            model_id="parity_test_miles",
            request_id="test",
            base_model=PARITY_CONFIG["base_model"],
            num_gpus=PARITY_CONFIG["num_gpus"],
            lora_config=PARITY_CONFIG["lora_config"],
        )
    )
    yield handle
    event_loop.run_until_complete(miles_backend.delete_model(handle))


@pytest.fixture(scope="module")
def nemo_rl_handle(nemo_rl_backend, event_loop):
    """Create a NeMo RL model handle."""
    handle = event_loop.run_until_complete(
        nemo_rl_backend.create_model(
            model_id="parity_test_nemo_rl",
            request_id="test",
            base_model=PARITY_CONFIG["base_model"],
            num_gpus=PARITY_CONFIG["num_gpus"],
            lora_config=PARITY_CONFIG["lora_config"],
        )
    )
    yield handle
    event_loop.run_until_complete(nemo_rl_backend.delete_model(handle))


# ---------------------------------------------------------------------------
# T046: Loss value parity test
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parity
class TestLossParity:
    """Verify loss values are within tolerance across backends."""

    def test_forward_backward_loss_within_tolerance(
        self,
        miles_backend,
        nemo_rl_backend,
        miles_handle,
        nemo_rl_handle,
        event_loop,
    ):
        """
        T046: Run forward_backward on both backends with identical data,
        assert total_loss within 1% relative tolerance.
        """
        data = make_deterministic_batch(
            batch_size=PARITY_CONFIG["batch_size"],
            seq_len=PARITY_CONFIG["seq_len"],
            response_len=PARITY_CONFIG["response_len"],
        )

        # Miles: forward_backward returns real metrics
        miles_result = event_loop.run_until_complete(
            miles_backend.forward_backward(miles_handle, data, "importance_sampling")
        )

        # NeMo RL: forward_backward buffers; need apply_optimizer_step for real metrics
        event_loop.run_until_complete(
            nemo_rl_backend.forward_backward(nemo_rl_handle, data, "importance_sampling")
        )
        nemo_result = event_loop.run_until_complete(
            nemo_rl_backend.apply_optimizer_step(nemo_rl_handle)
        )

        # Also run Miles optimizer step to keep both in sync
        event_loop.run_until_complete(miles_backend.apply_optimizer_step(miles_handle))

        # Compare losses
        miles_loss = _extract_loss(miles_result)
        nemo_loss = _extract_loss(nemo_result)

        if miles_loss is not None and nemo_loss is not None and miles_loss > 0:
            relative_diff = abs(miles_loss - nemo_loss) / miles_loss * 100
            logger.info(
                "Loss parity: Miles=%.6f, NeMo RL=%.6f, diff=%.2f%%",
                miles_loss, nemo_loss, relative_diff,
            )
            assert relative_diff < PARITY_CONFIG["loss_tolerance_pct"], (
                f"Loss values differ by {relative_diff:.2f}% "
                f"(Miles={miles_loss:.6f}, NeMo RL={nemo_loss:.6f})"
            )
        else:
            pytest.skip("Could not extract comparable loss values from both backends")


# ---------------------------------------------------------------------------
# T047: Gradient norm and KL metric trend parity
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parity
class TestMetricTrendParity:
    """Verify training metric trends agree across backends."""

    def test_grad_norm_trend_agreement(
        self,
        miles_backend,
        nemo_rl_backend,
        miles_handle,
        nemo_rl_handle,
        event_loop,
    ):
        """
        T047: Run N training steps on both backends, verify metrics
        track the same trends (both increase or both decrease between steps).
        """
        num_steps = PARITY_CONFIG["num_training_steps"]

        miles_grad_norms = []
        nemo_grad_norms = []

        for step in range(num_steps):
            data = make_deterministic_batch(
                batch_size=PARITY_CONFIG["batch_size"],
                seq_len=PARITY_CONFIG["seq_len"],
                response_len=PARITY_CONFIG["response_len"],
                seed=42 + step,
            )

            # Miles
            event_loop.run_until_complete(
                miles_backend.forward_backward(miles_handle, data, "importance_sampling")
            )
            miles_result = event_loop.run_until_complete(
                miles_backend.apply_optimizer_step(miles_handle)
            )

            # NeMo RL
            event_loop.run_until_complete(
                nemo_rl_backend.forward_backward(nemo_rl_handle, data, "importance_sampling")
            )
            nemo_result = event_loop.run_until_complete(
                nemo_rl_backend.apply_optimizer_step(nemo_rl_handle)
            )

            miles_gn = miles_result.get("grad_norm", 0.0)
            nemo_gn = nemo_result.get("grad_norm", 0.0)

            miles_grad_norms.append(miles_gn)
            nemo_grad_norms.append(nemo_gn)

            logger.info(
                "Step %d: Miles grad_norm=%.4f, NeMo RL grad_norm=%.4f",
                step, miles_gn, nemo_gn,
            )

        # Check trend agreement
        agreements = 0
        total = len(miles_grad_norms) - 1
        for i in range(total):
            miles_delta = miles_grad_norms[i + 1] - miles_grad_norms[i]
            nemo_delta = nemo_grad_norms[i + 1] - nemo_grad_norms[i]
            if (miles_delta >= 0 and nemo_delta >= 0) or (miles_delta <= 0 and nemo_delta <= 0):
                agreements += 1

        if total > 0:
            agreement_rate = agreements / total
            logger.info(
                "Gradient norm trend agreement: %d/%d (%.1f%%)",
                agreements, total, agreement_rate * 100,
            )
            assert agreement_rate >= 0.5, (
                f"Gradient norm trends disagree too often: {agreements}/{total}"
            )


# ---------------------------------------------------------------------------
# T048: Logprobs parity test
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parity
class TestLogprobsParity:
    """Verify logprobs values within floating-point tolerance."""

    def test_logprobs_within_tolerance(
        self,
        miles_backend,
        nemo_rl_backend,
        miles_handle,
        nemo_rl_handle,
        event_loop,
    ):
        """
        T048: Compute logprobs on both backends for identical data,
        verify values are within floating-point tolerance.
        """
        data = make_deterministic_batch(
            batch_size=2,
            seq_len=PARITY_CONFIG["seq_len"],
            response_len=PARITY_CONFIG["response_len"],
        )

        miles_logprobs = event_loop.run_until_complete(
            miles_backend.get_logprobs(miles_handle, data)
        )
        nemo_logprobs = event_loop.run_until_complete(
            nemo_rl_backend.get_logprobs(nemo_rl_handle, data)
        )

        if not miles_logprobs or not nemo_logprobs:
            pytest.skip("Could not extract logprobs from one or both backends")

        for i in range(min(len(miles_logprobs), len(nemo_logprobs))):
            miles_lp = torch.tensor(miles_logprobs[i], dtype=torch.float32)
            nemo_lp = torch.tensor(nemo_logprobs[i], dtype=torch.float32)

            min_len = min(len(miles_lp), len(nemo_lp))
            if min_len == 0:
                continue

            miles_lp = miles_lp[:min_len]
            nemo_lp = nemo_lp[:min_len]

            max_diff = (miles_lp - nemo_lp).abs().max().item()
            mean_diff = (miles_lp - nemo_lp).abs().mean().item()

            logger.info(
                "Sample %d logprobs: max_diff=%.6f, mean_diff=%.6f",
                i, max_diff, mean_diff,
            )

            assert mean_diff < PARITY_CONFIG["logprob_atol"], (
                f"Sample {i} logprobs differ by mean={mean_diff:.6f} "
                f"(tolerance={PARITY_CONFIG['logprob_atol']})"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_loss(result: Dict[str, Any]) -> Optional[float]:
    """Extract loss value from backend result dict."""
    if "loss" in result and result["loss"] is not None:
        return float(result["loss"])
    metrics = result.get("metrics", {})
    if "total_loss" in metrics:
        return float(metrics["total_loss"])
    if "total_loss:sum" in metrics:
        return float(metrics["total_loss:sum"])
    return None
