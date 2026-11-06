"""The optimizer step's per-datum logprobs on NemoRLBackend line up with forward()'s.

apply_optimizer_step returns each datum's training logprobs from the workers'
per-microbatch outputs, concatenated across microbatches and data-parallel
ranks and unsorted after dynamic batching. forward() reaches the same weights
through policy.get_logprobs, an independent path. Row i of the step must be
datum i's: same length, same values, and closer to its own forward row than to
any other datum's of that length. Then a checkpoint saves from the offloaded
policy a refit leaves behind (the Megatron worker's offload frees its
parameter buffers).

Shared by test_nemo_rl_logprob_order_{dtensor,megatron}.py: one create per Ray
instance (nemo_rl's placement group name is deterministic), and the GPU
conftest starts one Ray per module. Train-only (no vLLM): 2 GPUs, DP=2.
"""
import asyncio

import torch

from tests.datums import random_ce_datum
from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend
from tinkercloud.training.models.requests import AdamParams, LoraConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# unsorted lengths, two equal: 9 rows over DP=2 also leave one padding row
LENGTHS = (37, 64, 5, 80, 21, 50, 9, 50, 33)


def check_step_logprobs_follow_the_datums(megatron: bool, tmp_path) -> None:
    g = torch.Generator().manual_seed(0)
    data = [random_ce_datum(g, n) for n in LENGTHS]

    async def go():
        backend = NemoRLBackend(overrides={"megatron": megatron, "debug_train_only": True})
        handle = await backend.create_model("m", "req-1", MODEL, num_gpus=2,
                                            lora_config=LoraConfig(rank=8, alpha=16).dict(),
                                            native_root=tmp_path, optimizer="adamw")
        try:
            fwd = await backend.forward(handle, data, "cross_entropy")
            await backend.forward_backward(handle, data, "cross_entropy")
            step = await backend.apply_optimizer_step(handle, AdamParams(learning_rate=1e-4, eps=1e-8))
            # train-only runs no refit: offload as a refit does, then save
            await asyncio.to_thread(handle.policy.offload_after_refit)
            handle.training_resident = False
            await backend.save_checkpoint(handle, tmp_path / "ckpt", step=1)
        finally:
            await backend.delete_model(handle)
        return fwd["loss_fn_outputs"], step["loss_fn_outputs"]

    fwd, step = asyncio.run(go())
    assert any((tmp_path / "ckpt" / "weights").iterdir()), "the save wrote no weights"
    ref = [torch.tensor(o["logprobs"]["data"]) for o in fwd]
    got = [torch.tensor(o["logprobs"]["data"]) for o in step]
    # each row is its datum's length up to one constant offset, on both paths
    assert [len(t) for t in got] == [len(t) for t in ref], ([len(t) for t in got], [len(t) for t in ref])
    assert len({len(t) - n for t, n in zip(ref, LENGTHS)}) == 1, [len(t) for t in ref]
    for i, (a, b) in enumerate(zip(got, ref)):
        gap = (a - b).abs()
        assert gap.max() < 0.05, f"datum {i}: step vs forward max gap {gap.max():.4f}"
        for j, other in enumerate(ref):
            if j != i and len(other) == len(a):
                assert (a - other).abs().mean() > 10 * gap.mean(), f"datum {i} is as close to datum {j}"
