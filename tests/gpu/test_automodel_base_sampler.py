"""AutomodelBackend's base-model sampler on real GPUs (Qwen2.5-0.5B-Instruct): with no training model
on the base, a base-model handle samples and scores from the base model's vLLM with no adapter; a
LoRA model created on that base then shares the vLLM, and its step-0 adapter (B = 0) scores as the
base does. Everything the test starts is gone at the end.

Needs >= 3 GPUs (2 ranks + vLLM), the Automodel training env with /opt/vllm-env, and the model in
the HF cache; starts its own Ray instance. Skipped elsewhere.

Run on a GPU pod (vLLM boots in ~2 minutes):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_base_sampler.py -s
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import math
import time

import pytest

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import ray
import torch

if torch.cuda.device_count() < 3:
    pytest.skip("needs >= 3 GPUs", allow_module_level=True)

from tinkercloud.training.backends.automodel.backend import AutomodelBackend
from tinkercloud.training.core import routing

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = dict(train_gpus=2, max_loras=2, max_lora_rank=8, max_cpu_loras=4, micro_batch_rows=2, activation_checkpointing=False)
T1 = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 8}


def am_actors():
    return {n for n in ray.util.list_named_actors() if n.startswith("am-")}


def test_base_model_sampler_serves_the_base_and_shares_its_vllm(tmp_path):
    async def go():
        backend = AutomodelBackend(overrides=CONFIG)
        lora = None
        try:
            await backend.check_base_model(MODEL)
            with pytest.raises(ValueError, match="not/a-model"):
                await backend.check_base_model("not/a-model")
            base = await backend.base_model_handle(MODEL)
            assert base.lora is None and len(base.inference_endpoints) == 1
            assert await backend.base_model_handle(MODEL) is base
            prompt = base.tokenizer.encode("The capital of France is")

            out = await backend.sample(base, "b0", prompt, 2, T1)
            assert len(out["sequences"]) == 2 and all(0 < len(s["tokens"]) <= 8 for s in out["sequences"])
            assert all(lp <= 0 for s in out["sequences"] for lp in s["logprobs"])
            scored = await backend.sample(base, "b1", prompt, 1, {**T1, "max_tokens": 1}, prompt_logprobs=True)
            base_lp = scored["prompt_logprobs"]
            assert base_lp[0] is None and all(math.isfinite(x) for x in base_lp[1:])
            print(f"base sampler: {len(out['sequences'])} samples, prompt logprobs {base_lp[1:]}")

            lora = await backend.create_model("m", "req-1", MODEL, num_gpus=2, lora_config={"rank": 8},
                                              native_root=tmp_path, optimizer="adamw")
            routing.table.publish("m", routing.InferenceEndpoint(lora.inference_endpoints))
            assert lora.inference_endpoints == base.inference_endpoints, "one vLLM per base model"
            lora_lp = (await backend.sample(lora, "l0", prompt, 1, {**T1, "max_tokens": 1},
                                            prompt_logprobs=True))["prompt_logprobs"]
            gap = [abs(a - b) for a, b in zip(lora_lp[1:], base_lp[1:])]
            print(f"step-0 adapter vs base: mean |d| {sum(gap) / len(gap):.5f}, max {max(gap):.5f}")
            assert sum(gap) / len(gap) <= 1e-2

            await backend.delete_model(lora)
            routing.table.withdraw("m")
            lora = None
            again = await backend.sample(base, "b2", prompt, 1, {**T1, "max_tokens": 1}, prompt_logprobs=True)
            assert again["prompt_logprobs"][0] is None, "the base sampler outlives the LoRA tenant's delete"
        finally:
            if lora is not None:
                await backend.delete_model(lora)
                routing.table.withdraw("m")
            await backend.close()

    asyncio.run(go())
    deadline = time.monotonic() + 15
    while am_actors() and time.monotonic() < deadline:   # ray.kill returns before the actor is gone
        time.sleep(0.5)
    assert not am_actors(), f"left behind: {am_actors()}"
