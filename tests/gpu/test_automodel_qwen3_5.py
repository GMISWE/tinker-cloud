"""AutomodelBackend on Qwen3.5-9B-Base, LoRA rank 128: the Gated DeltaNet LoRA
(per-slice in_proj_qkv) builds under FSDP2 with fp32 LoRA, has Tinker's element
count, and its rank-3r export served by vLLM scores tokens as the trainer does,
before training and after the adapter has moved the model.

Needs >= 5 GPUs (4 ranks + vLLM), the Automodel training env with /opt/vllm-env,
and the model in the HF cache; starts its own Ray instance. Skipped elsewhere.

Run on a GPU pod (vLLM boots in ~4 minutes):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_qwen3_5.py -s
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 5:
    pytest.skip("needs >= 5 GPUs", allow_module_level=True)

from tinkercloud.training.backends.automodel.backend import AutomodelBackend  # noqa: E402
from tinkercloud.training.core import routing  # noqa: E402
from tinkercloud.training.models.requests import AdamParams  # noqa: E402
from tests.datums import ce_datum, rl_datum  # noqa: E402

MODEL = "Qwen/Qwen3.5-9B-Base"
RANK = 128
CONFIG = dict(train_gpus=4, max_loras=2, max_lora_rank=RANK, max_cpu_loras=4, micro_batch_rows=1, activation_checkpointing=False)
# Tinker's per-rank count for Qwen3.5-9B with train_attn and train_mlp (tinker_cookbook.hyperparam_utils)
TINKER_ELEMENTS = RANK * (1_572_864 + 1_130_496)
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
MAX_STEPS = 30
BACKEND_LOGGER = "tinkercloud.training.backends.automodel.backend"
TEXT = ("The mitochondrion is the powerhouse of the cell: it turns the chemical energy of food into ATP, "
        "the molecule that drives most of the work a cell does, from muscle contraction to protein synthesis.")


def _gaps(a, b):
    gaps = sorted(abs(x - y) for x, y in zip(a, b))
    return sum(gaps) / len(gaps), gaps[int(0.99 * (len(gaps) - 1))]


def test_qwen3_5_lora_trains_and_serves_through_the_rank_3r_export(caplog, tmp_path):
    caplog.set_level(logging.INFO, logger=BACKEND_LOGGER)

    async def go():
        backend = AutomodelBackend(overrides=CONFIG)
        handle = await backend.create_model("q35", "req-1", MODEL, num_gpus=4, lora_config={"rank": RANK},
                                            native_root=tmp_path, optimizer="adamw")
        routing.table.publish("q35", routing.InferenceEndpoint(handle.inference_endpoints))
        try:
            tokens = handle.tokenizer.encode(TEXT)
            score = ce_datum(tokens, tokens[1:] + tokens[:1])   # position i scores token i+1 (the last wraps)

            async def sampler_logprobs():
                out = await backend.sample(handle, "lp", tokens, 1, {"temperature": 1.0, "top_p": 1.0, "max_tokens": 1},
                                           prompt_logprobs=True)
                assert out["prompt_logprobs"][0] is None
                return out["prompt_logprobs"][1:]

            async def trainer_logprobs():
                out = await backend.forward(handle, [score], "cross_entropy")
                return out["loss_fn_outputs"][0]["logprobs"]["data"][:-1]   # position i scores token i+1

            base = await trainer_logprobs()
            step0 = _gaps(await sampler_logprobs(), base)
            print(f"step 0 sampler vs trainer: mean |d| {step0[0]:.4f}, p99 {step0[1]:.4f} (n {len(base)})")

            # positive advantage on the fixed tokens: each step raises their logprobs
            rl = rl_datum(tokens[:-1], tokens[1:], base, [1.0] * (len(tokens) - 1))
            trained, moved, steps = base, 0.0, 0
            while moved < 10 * step0[0]:
                assert steps < MAX_STEPS, f"the adapter moved the logprobs by {moved:.4f} in {steps} steps"
                await backend.forward_backward(handle, [rl], "importance_sampling")
                await backend.apply_optimizer_step(handle, AdamParams(learning_rate=1e-4, **ADAM))
                steps += 1
                trained = await trainer_logprobs()
                moved = _gaps(trained, base)[0]
            final = _gaps(await sampler_logprobs(), trained)
            print(f"adapter effect after {steps} steps: trainer vs base mean |d| {moved:.4f}")
            print(f"final sampler vs trainer: mean |d| {final[0]:.4f}, p99 {final[1]:.4f}")
            assert final[0] <= 2 * step0[0], (step0, final)
            adapter = tmp_path / "adapters" / f"v{handle.generation_synced_version}" / "adapter_config.json"
            print(f"served adapter {adapter.read_text()}")
        finally:
            await backend.delete_model(handle)
            routing.table.withdraw("q35")
            await backend.close()

    asyncio.run(go())
    sizes = [int(n) for n in re.findall(r"LoRA (\d+) elements", caplog.text)]
    ranks = [int(n) for n in re.findall(r"max LoRA rank (\d+)", caplog.text)]
    for r in caplog.records:   # the backend's own lines: the element count and the rank vLLM boots at
        if re.search(r"LoRA \d+ elements|max LoRA rank \d+", r.getMessage()):
            print("backend log:", r.getMessage())
    print(f"LoRA elements {sizes}, vLLM max LoRA rank {ranks}")
    assert sizes == [TINKER_ELEMENTS] == [346_030_080]
    assert ranks == [512]   # the smallest rank vLLM 0.29 accepts that holds the rank-384 export
