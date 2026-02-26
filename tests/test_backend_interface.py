"""
Contract tests for TrainingBackend interface compliance.

Tests that both MilesBackend and NemoRLBackend implement the TrainingBackend
interface correctly with identical synthetic inputs. These tests verify
interface compliance, NOT parity of results (see test_backend_parity.py).

Tests are organized by dependency level:
- TestBackendError, TestUnsupportedFeatureError: No imports beyond base
- TestBackendFactory, TestTrainingBackendABC: Imports factory (lazy Miles/NeMo)
- TestNemoRLArgumentBuilder: No NeMo RL runtime needed
- TestNemoRLDataConverter: Requires NeMo RL (skipped if not installed)
- TestNemoRLBackendBuffering: Requires NeMo RL (skipped if not installed)
"""
import asyncio
import importlib.util
import logging
import pytest

from tinkercloud.training.backends.base import (
    BackendError,
    BackendHandle,
    TrainingBackend,
    UnsupportedFeatureError,
)
from tinkercloud.training.backends.factory import BackendFactory

# Guard for NeMo RL availability
HAS_NEMO_RL = importlib.util.find_spec("nemo_rl") is not None


# ---------------------------------------------------------------------------
# Fixtures: synthetic data for contract testing
# ---------------------------------------------------------------------------

def make_synthetic_datum(seq_len: int = 128, response_len: int = 64) -> dict:
    """Create a synthetic Tinker Datum dict for testing."""
    import torch

    prompt_len = seq_len - response_len
    tokens = torch.randint(0, 32000, (seq_len,), dtype=torch.long).tolist()
    loss_masks = [0.0] * prompt_len + [1.0] * response_len
    advantages = [0.1] * response_len
    log_probs = [-5.0 + 0.1 * i for i in range(response_len)]
    rollout_log_probs = [-5.5 + 0.1 * i for i in range(response_len)]
    ref_log_probs = [-5.2 + 0.1 * i for i in range(response_len)]

    return {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "advantages": advantages,
        "log_probs": log_probs,
        "rollout_log_probs": rollout_log_probs,
        "ref_log_probs": ref_log_probs,
    }


def make_synthetic_batch(batch_size: int = 4, seq_len: int = 128) -> list:
    """Create a batch of synthetic Tinker Datum dicts."""
    return [make_synthetic_datum(seq_len=seq_len) for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Unit tests: base classes and factory (no external deps)
# ---------------------------------------------------------------------------

class TestBackendError:
    """Test BackendError exception behavior."""

    def test_error_message_format(self):
        err = BackendError("test error", backend="miles", operation="forward")
        assert "[miles]" in str(err)
        assert "forward" in str(err)
        assert "test error" in str(err)

    def test_error_with_original_exception(self):
        original = ValueError("original error")
        err = BackendError(
            "wrapper", backend="nemo_rl", operation="create_model",
            original_error=original,
        )
        assert err.original_error is original
        assert "ValueError" in str(err)

    def test_error_attributes(self):
        err = BackendError("msg", backend="miles", operation="forward_backward")
        assert err.backend == "miles"
        assert err.operation == "forward_backward"


class TestUnsupportedFeatureError:
    """Test UnsupportedFeatureError subclass."""

    def test_inherits_backend_error(self):
        err = UnsupportedFeatureError("rlve_server", backend="nemo_rl")
        assert isinstance(err, BackendError)
        assert err.backend == "nemo_rl"
        assert err.operation == "feature_check"

    def test_with_suggestion(self):
        err = UnsupportedFeatureError(
            "rlve_server", backend="nemo_rl",
            suggestion="Use client-side RLVE via tinker-cookbook",
        )
        assert "tinker-cookbook" in str(err)


class TestBackendFactory:
    """Test BackendFactory dispatch."""

    def test_create_miles(self):
        backend = BackendFactory.create(backend_type="miles")
        assert backend is not None
        assert hasattr(backend, "create_model")

    def test_create_nemo_rl(self):
        backend = BackendFactory.create(backend_type="nemo_rl")
        assert backend is not None
        assert hasattr(backend, "create_model")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendFactory.create(backend_type="invalid_backend")


class TestTrainingBackendABC:
    """Test that TrainingBackend ABC enforces method signatures."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            TrainingBackend()

    def test_required_methods(self):
        required = [
            "create_model", "forward", "forward_backward",
            "apply_optimizer_step", "update_inference_weights",
            "save_checkpoint", "load_checkpoint", "delete_model",
            "get_logprobs",
        ]
        for method_name in required:
            assert hasattr(TrainingBackend, method_name), (
                f"TrainingBackend missing required method: {method_name}"
            )


# ---------------------------------------------------------------------------
# Contract tests: NemoRLArgumentBuilder (no NeMo RL runtime needed)
# ---------------------------------------------------------------------------

class TestNemoRLArgumentBuilder:
    """Test NemoRLArgumentBuilder config generation."""

    @pytest.fixture
    def builder(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        return NemoRLArgumentBuilder()

    def test_build_args_returns_tuple(self, builder):
        config_dict, hf_path = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert isinstance(config_dict, dict)
        assert isinstance(hf_path, str)
        assert hf_path == "meta-llama/Llama-3.1-8B"

    def test_config_has_required_sections(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert "policy" in config_dict
        assert "loss_fn" in config_dict
        assert "cluster" in config_dict

    def test_parallelism_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=8,
            parallelism={"tensor_parallel": 2, "pipeline_parallel": 1},
        )
        assert config_dict["policy"]["dtensor_cfg"]["tensor_parallel_size"] == 2
        assert config_dict["dp_size"] == 4  # 8 / (2 * 1)

    def test_lora_config_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
            lora_config={"rank": 32, "alpha": 64, "dropout": 0.1},
        )
        lora_cfg = config_dict["policy"]["dtensor_cfg"]["lora_cfg"]
        assert lora_cfg["enabled"] is True
        assert lora_cfg["dim"] == 32
        assert lora_cfg["alpha"] == 64
        assert lora_cfg["dropout"] == 0.1

    def test_rl_config_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
            rl_config={"kl_penalty_coef": 0.01, "eps_clip": 0.3},
        )
        assert config_dict["loss_fn"]["reference_policy_kl_penalty"] == 0.01
        assert config_dict["loss_fn"]["ratio_clip_min"] == 0.3

    def test_overrides_applied(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        builder = NemoRLArgumentBuilder(overrides={
            "policy": {"precision": "float32"},
        })
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert config_dict["policy"]["precision"] == "float32"

    def test_rlve_miles_only_warning(self, builder, caplog):
        with caplog.at_level(logging.WARNING):
            builder.build_args(
                base_model="meta-llama/Llama-3.1-8B",
                num_gpus=4,
                rlve_config={
                    "enabled": True,
                    "custom_prompt_preprocessor": "math",
                },
            )
        assert "Miles-only" in caplog.text


# ---------------------------------------------------------------------------
# Contract tests: NemoRLDataConverter (requires NeMo RL installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NEMO_RL, reason="NeMo RL not installed")
class TestNemoRLDataConverter:
    """Test NemoRLDataConverter with synthetic data."""

    @pytest.fixture
    def converter(self):
        from tinkercloud.training.backends.nemo_rl.converter import NemoRLDataConverter
        return NemoRLDataConverter()

    def test_forward_to_backend_shapes(self, converter):
        data = make_synthetic_batch(batch_size=4, seq_len=128)
        config = {}
        result = converter.forward_to_backend(data, config)

        assert "input_ids" in result
        assert "input_lengths" in result
        assert "token_mask" in result
        assert result["input_ids"].shape[0] == 4  # batch size
        assert result["input_ids"].shape[1] == 128  # max seq len

    def test_forward_backward_to_backend_shapes(self, converter):
        data = make_synthetic_batch(batch_size=4, seq_len=128)
        config = {}
        result = converter.forward_backward_to_backend(data, "importance_sampling", config)

        expected_keys = {
            "input_ids", "input_lengths", "token_mask",
            "advantages", "prev_logprobs", "generation_logprobs",
            "reference_policy_logprobs", "sample_mask",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["input_ids"].shape == (4, 128)
        assert result["sample_mask"].shape == (4,)
        assert result["sample_mask"].sum().item() == 4.0

    def test_forward_backward_empty_data(self, converter):
        result = converter.forward_backward_to_backend([], "importance_sampling", {})
        assert result["input_ids"].shape[0] == 0

    def test_variable_length_sequences(self, converter):
        data = [
            make_synthetic_datum(seq_len=64, response_len=32),
            make_synthetic_datum(seq_len=128, response_len=64),
        ]
        result = converter.forward_backward_to_backend(data, "importance_sampling", {})
        assert result["input_ids"].shape == (2, 128)
        assert result["input_ids"][0, 65:].sum().item() == 0

    def test_deferred_result_conversion(self, converter):
        result = converter.backend_to_forward_backward_result(None, [], loss_fn="importance_sampling")
        assert result["deferred"] is True
        assert result["metrics"] == {}
        assert result["loss_fn_output_type"] == "importance_sampling"


# ---------------------------------------------------------------------------
# Contract tests: NemoRLBackend buffering (requires NeMo RL installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_NEMO_RL, reason="NeMo RL not installed")
class TestNemoRLBackendBuffering:
    """Test the R9 buffering strategy without GPU."""

    @pytest.fixture
    def backend(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend
        return NemoRLBackend()

    @pytest.fixture
    def handle(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle
        return NemoRLHandle(model_id="test", backend_type="nemo_rl", config={})

    def test_forward_backward_returns_deferred(self, backend, handle):
        """forward_backward() should return deferred=True."""
        data = make_synthetic_batch(batch_size=2)
        result = asyncio.run(
            backend.forward_backward(handle, data, "importance_sampling")
        )
        assert result["deferred"] is True
        assert result["metrics"] == {}
        assert len(handle.data_buffer) == 1

    def test_forward_backward_accumulates_buffer(self, backend, handle):
        """Multiple forward_backward() calls accumulate in buffer."""
        for _ in range(3):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )
        assert len(handle.data_buffer) == 3

    def test_forward_backward_empty_data_raises(self, backend, handle):
        """forward_backward() with empty data should raise BackendError."""
        with pytest.raises(BackendError, match="Empty data"):
            asyncio.run(
                backend.forward_backward(handle, [], "importance_sampling")
            )

    def test_apply_optimizer_step_empty_buffer_raises(self, backend, handle):
        """apply_optimizer_step() with empty buffer should raise BackendError."""
        with pytest.raises(BackendError, match="No buffered data"):
            asyncio.run(
                backend.apply_optimizer_step(handle)
            )

    def test_buffer_overflow_raises(self, backend, handle):
        """forward_backward() should raise when buffer exceeds max_buffer_size (CHK006)."""
        handle.max_buffer_size = 3
        for _ in range(3):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )
        assert len(handle.data_buffer) == 3
        with pytest.raises(BackendError, match="Buffer full"):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )

    def test_delete_model_clears_buffer(self, backend, handle):
        """delete_model() should warn and clear pending buffer (CHK007)."""
        asyncio.run(
            backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
        )
        assert len(handle.data_buffer) == 1
        # delete_model will fail on policy shutdown (no real policy), but
        # the buffer should be cleared before that error
        try:
            asyncio.run(backend.delete_model(handle))
        except BackendError:
            pass  # Expected — no real policy
        assert len(handle.data_buffer) == 0

    def test_deferred_result_format(self, backend, handle):
        """forward_backward() deferred result must have exact contract fields (CHK010)."""
        data = make_synthetic_batch(batch_size=2)
        result = asyncio.run(
            backend.forward_backward(handle, data, "importance_sampling")
        )
        assert result == {"metrics": {}, "deferred": True, "loss_fn_outputs": []}

    def test_concurrent_forward_backward_thread_safe(self, backend, handle):
        """Concurrent forward_backward() calls should be serialized by lock (CHK018)."""
        async def run_concurrent():
            tasks = [
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_concurrent())
        assert len(handle.data_buffer) == 5
        assert all(r["deferred"] is True for r in results)
