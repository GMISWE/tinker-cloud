"""AutomodelBackend on real GPUs (Qwen2.5-0.5B-Instruct, LoRA rank 8): one
tenant's lifecycle through the backend, training and sampling through its
vLLM with versioned adapters (R4, D20), and the leftover sweep (R5).

Needs >= 3 GPUs (2 ranks + vLLM), the Automodel training env with
/opt/vllm-env, and the model in the HF cache; starts its own Ray instance.
Skipped elsewhere.

Run on a GPU pod (about 5 minutes; vLLM boots in ~2):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_backend.py -v

HF_HUB_OFFLINE is set before huggingface_hub loads: create_model reads the
raw HF config through hf_hub_download in this process. An actor class
defined at module top level fails to import inside the Ray actor process;
define it inside the test so it is pickled by value.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import ray  # noqa: E402
import torch  # noqa: E402

if torch.cuda.device_count() < 3:
    pytest.skip("needs >= 3 GPUs", allow_module_level=True)

from ray.util.placement_group import placement_group, placement_group_table  # noqa: E402

from tinkercloud.training.backends.automodel.backend import AutomodelBackend  # noqa: E402
from tinkercloud.training.backends.automodel.group import sweep_leftovers  # noqa: E402
from tinkercloud.training.backends.base import BackendError  # noqa: E402
from tinkercloud.training.core import routing  # noqa: E402
from tinkercloud.training.models.requests import Datum  # noqa: E402
from tests.datums import random_ce_datum  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# two adapter versions coexist while a sample of the older one is in flight
CONFIG = dict(train_gpus=2, max_loras=2, max_lora_rank=8, max_cpu_loras=4, micro_batch_rows=2)
GREEDY = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8}
LORA = {"rank": 8, "alpha": 16}
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
BACKEND_LOGGER = "tinkercloud.training.backends.automodel.backend"


def datums(lengths, weight=1.0):
    g = torch.Generator().manual_seed(0)
    return [random_ce_datum(g, n, weight) for n in lengths]


def am_actors(prefix):
    return {n for n in ray.util.list_named_actors() if n.startswith(prefix)}


def pg_states(name):
    return {info["state"] for info in placement_group_table().values() if info["name"] == name}


def wait_until(cond, seconds=15):
    deadline = time.monotonic() + seconds
    while not cond():
        assert time.monotonic() < deadline, "Ray did not reach the expected state in time"
        time.sleep(0.5)


def prompt_datum(tokens):
    """The prompt as a CE datum: position i scores token i+1 (the last wraps)."""
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": tokens[1:] + [tokens[0]]}, "weights": {"data": [1.0] * len(tokens)}}})


def port_open(url):
    host, port = url.rsplit("://", 1)[1].split(":")
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, int(port))) == 0


def test_lifecycle_through_the_backend(caplog, tmp_path):
    caplog.set_level(logging.INFO, logger=BACKEND_LOGGER)
    x, y = datums((37, 64, 5)), datums((80, 21, 50, 9, 33))

    async def go():
        backend = AutomodelBackend(overrides=CONFIG)
        handle = await backend.create_model("m", "req-1", MODEL, num_gpus=2, lora_config=LORA, native_root=tmp_path)
        key, skey = handle.group_key, handle.sampler_key
        routing.table.publish("m", routing.InferenceEndpoint(handle.inference_endpoints))
        try:
            assert handle.context_length is not None and handle.context_length > 0
            assert handle.weight_version == 0 and handle.max_topk_logprobs == 20
            assert am_actors(f"am-{key}-r") == {f"am-{key}-r0", f"am-{key}-r1"}
            assert pg_states(f"am-{key}") == {"CREATED"}
            assert am_actors(f"am-infer-{skey}") == {f"am-infer-{skey}-host"}
            assert pg_states(f"am-infer-{skey}") == {"CREATED"}
            (url,) = handle.inference_endpoints
            assert port_open(url)

            # v0 (B = 0, so the base model's distribution) is published on the first sample
            prompt = handle.tokenizer.encode("The capital of France is")
            s0 = await backend.sample(handle, "s0", prompt, 1, GREEDY)   # vLLM refuses n > 1 when greedy
            assert s0["weight_version"] == 0 and s0["latest_weight_version"] == 0
            assert len(s0["sequences"]) == 1
            assert 0 < len(s0["sequences"][0]["tokens"]) <= 8 and s0["sequences"][0]["text"]
            assert all(lp <= 0 for lp in s0["sequences"][0]["logprobs"])
            assert (tmp_path / "adapters" / "v0" / "adapter_config.json").is_file()
            # the sampler's prompt logprobs and the trainer's logprobs score the same weights
            scored = await backend.sample(handle, "s0lp", prompt, 1, {**GREEDY, "max_tokens": 1}, prompt_logprobs=True)
            trained = (await backend.forward(handle, [prompt_datum(prompt)], "cross_entropy"))["loss_fn_outputs"][0]
            vllm_lp, train_lp = scored["prompt_logprobs"][1:], trained["logprobs"]["data"][:-1]
            gap = [abs(a - b) for a, b in zip(vllm_lp, train_lp)]
            assert scored["prompt_logprobs"][0] is None and len(gap) == len(prompt) - 1
            assert sum(gap) / len(gap) < 0.1 and max(gap) < 0.5, f"sampler vs trainer logprobs: {gap}"

            with pytest.raises(BackendError, match="already serves m"):
                await backend.create_model("m2", "req-2", MODEL, num_gpus=2, lora_config=LORA, native_root=tmp_path)

            at_init = await backend.forward(handle, x, "cross_entropy")
            a, b = await asyncio.gather(
                backend.forward_backward(handle, x, "cross_entropy"),
                backend.forward_backward(handle, y, "cross_entropy"),
            )
            # forward and forward_backward score the same state
            assert a["loss"] == pytest.approx(at_init["loss"], rel=1e-3)
            assert a["loss"] > 0 and b["loss"] > 0
            assert [o["logprobs"]["shape"] for o in a["loss_fn_outputs"]] == [[37], [64], [5]]
            assert all(v <= 0 for o in a["loss_fn_outputs"] for v in o["logprobs"]["data"])
            assert a["metrics"] == {"loss:sum": a["loss"], "clamped_positions:sum": 0}

            step = await backend.apply_optimizer_step(handle, 1e-3, ADAM)
            assert step["grad_norm"] > 0
            assert step["weight_version"] == handle.weight_version == 1
            after = await backend.forward(handle, x, "cross_entropy")
            assert after["loss"] != at_init["loss"]   # the step moved the adapter
            # the step published v1 and retired v0 (no sample of it in flight)
            assert handle.generation_synced_version == 1 and handle.loaded_versions == [1]
            assert (tmp_path / "adapters" / "v1").is_dir() and not (tmp_path / "adapters" / "v0").exists()
            s1 = await backend.sample(handle, "s1", prompt, 1, GREEDY)
            assert s1["weight_version"] == 1 and s1["latest_weight_version"] == 1

            # R10 through the backend: a non-finite norm fails that step only.
            # The worker's own message reaches the client, not Ray's traceback.
            await backend.forward_backward(handle, datums((37, 64, 5), weight=3e38), "cross_entropy")
            with pytest.raises(BackendError, match=r"grad norm is (inf|nan); the optimizer step was not taken") as err:
                await backend.apply_optimizer_step(handle, 1e-3, ADAM)
            assert "ray::" not in str(err.value) and len(str(err.value)) < 300
            assert handle.weight_version == 1
            await backend.forward_backward(handle, x, "cross_entropy")
            assert (await backend.apply_optimizer_step(handle, 1e-3, ADAM))["grad_norm"] > 0
            assert handle.weight_version == 2
        finally:
            await backend.delete_model(handle)
            routing.table.withdraw("m")

        wait_until(lambda: not am_actors(f"am-{key}-r") and not am_actors(f"am-infer-{skey}"))
        assert pg_states(f"am-{key}") == {"REMOVED"} and pg_states(f"am-infer-{skey}") == {"REMOVED"}
        assert not port_open(url), "vLLM died with its host"
        assert not (tmp_path / "adapters").exists()
        with pytest.raises(BackendError, match="deleted"):
            await backend.forward(handle, x, "cross_entropy")
        await backend.close()

    asyncio.run(go())
    # the LoRA size is reported once, by the backend's "group up" line
    sizes = re.findall(r"LoRA (\d+) elements", caplog.text)
    assert len(sizes) == 1 and int(sizes[0]) > 0


def test_sweep_removes_leftover_actors_and_placement_groups():
    @ray.remote(num_cpus=1)
    class Stale:
        def ping(self):
            return "up"

    pg = placement_group([{"CPU": 1}], name="am-stale")
    ray.get(pg.ready())
    stale = Stale.options(name="am-stale-r0").remote()
    assert ray.get(stale.ping.remote()) == "up"

    assert sweep_leftovers() == 2
    wait_until(lambda: not am_actors("am-stale"))
    assert pg_states("am-stale") == {"REMOVED"}
    assert sweep_leftovers() == 0
