"""AutomodelWorker activation checkpointing on Qwen3.5-9B-Base, LoRA rank 128, 4 ranks, one 16,384-token row
per rank and per micro-batch: with checkpointing the training step completes; without it the same step runs out
of GPU memory, or holds at least 10 GiB more on some rank.

Needs >= 4 GPUs, the Automodel training env and the model in the HF cache; starts its own Ray instance. Skipped
elsewhere.

Run on a GPU pod (two 9B group builds, no vLLM):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_activation_checkpointing.py -s
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import math  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import ray  # noqa: E402
from ray.exceptions import RayTaskError  # noqa: E402
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs >= 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.automodel.config import AutomodelConfig  # noqa: E402
from tinkercloud.training.backends.automodel.converter import coalesce, convert_request, shard  # noqa: E402
from tinkercloud.training.backends.automodel.lora import lora_spec  # noqa: E402
from tinkercloud.training.backends.automodel.worker import AutomodelWorker  # noqa: E402
from tinkercloud.training.models.requests import AdamParams, LoraConfig  # noqa: E402
from tests.datums import random_ce_datum  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B-Base"
RANKS = 4
ROW = 16384
LORA = lora_spec(LoraConfig(rank=128), "adamw")
ADAM = AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
MIN_GAP_BYTES = 10 * 2**30
STEP_TIMEOUT_S = 600


def _step(activation_checkpointing: bool, batch) -> tuple[str, list[int], float]:
    """One forward_backward and optimizer step on a fresh group, killed after. Returns the outcome: "completed",
    else "oom" when a rank ran out of memory; each rank's peak allocated bytes during the step (empty after an
    OOM); and the batch loss."""
    # defined here, not at module level, so Ray pickles it by value (test_automodel_worker.py's note)
    class PeakWorker(AutomodelWorker):
        def reset_peak(self) -> None:
            torch.cuda.reset_peak_memory_stats()

        def peak(self) -> int:
            return torch.cuda.max_memory_allocated()

    cls = ray.remote(num_gpus=1, num_cpus=1)(PeakWorker)
    config = AutomodelConfig(train_gpus=RANKS, max_loras=1, max_lora_rank=128, max_cpu_loras=1, micro_batch_rows=1,
                             activation_checkpointing=activation_checkpointing)
    ranks = [cls.remote(r, RANKS, MODEL, LORA, config) for r in range(RANKS)]
    try:
        addr = ray.get(ranks[0].rendezvous_address.remote())
        ray.get([w.setup.remote(addr) for w in ranks])
        ray.get([w.add_tenant.remote("t", LORA, 1234) for w in ranks])
        ray.get([w.reset_peak.remote() for w in ranks])
        refs = [w.forward_backward.remote("t", s, None) for w, s in zip(ranks, shard(batch, RANKS))]
        # a rank that runs out of memory leaves its peers inside a collective: watch every ref, stop at the first error
        pending = list(refs)
        while pending:
            done, pending = ray.wait(pending, num_returns=1, timeout=STEP_TIMEOUT_S)
            assert done, f"forward_backward made no progress in {STEP_TIMEOUT_S}s"
            try:
                ray.get(done[0])
            except RayTaskError as e:
                if isinstance(e.cause, torch.OutOfMemoryError):
                    return "oom", [], math.nan
                raise
        outs = ray.get(refs)
        ray.get([w.optim_step.remote("t", ADAM) for w in ranks])
        loss = float(sum(o["row_loss"].sum() for o in outs))
        return "completed", ray.get([w.peak.remote() for w in ranks]), loss
    finally:
        for w in ranks:
            ray.kill(w)


def test_checkpointing_lets_a_long_row_train_that_does_not_fit_without_it():
    g = torch.Generator().manual_seed(0)
    batch = coalesce([convert_request([random_ce_datum(g, ROW) for _ in range(RANKS)], "cross_entropy")], RANKS)

    on, on_peak, on_loss = _step(True, batch)
    print(f"checkpointing on: {on}, loss {on_loss:.4f}, peak GiB per rank {[round(p / 2**30, 1) for p in on_peak]}")
    assert on == "completed" and math.isfinite(on_loss)

    off, off_peak, _ = _step(False, batch)
    print(f"checkpointing off: {off}, peak GiB per rank {[round(p / 2**30, 1) for p in off_peak]}")
    if off == "completed":
        gap = max(b - a for a, b in zip(on_peak, off_peak))
        assert gap >= MIN_GAP_BYTES, f"off holds at most {gap / 2**30:.1f} GiB more than on"
