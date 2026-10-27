"""AutomodelBackend on real GPUs (Qwen2.5-0.5B-Instruct, LoRA rank 8): one
tenant's lifecycle through the backend, training and sampling through its
vLLM with versioned adapters, the leftover sweep, and a checkpoint round
trip through fresh models.

Needs >= 3 GPUs (2 ranks + vLLM), the Automodel training env with
/opt/vllm-env, and the model in the HF cache; starts its own Ray instance.
Skipped elsewhere.

Run on a GPU pod (about 13 minutes; vLLM boots in ~2, three times for the round trip):
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

import httpx  # noqa: E402
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

            # the same signature joins the group: no new ranks; leaving keeps them
            m2 = await backend.create_model("m2", "req-2", MODEL, num_gpus=2, lora_config=LORA, native_root=tmp_path / "m2")
            assert m2.group_key == key and am_actors(f"am-{key}-r") == {f"am-{key}-r0", f"am-{key}-r1"}
            await backend.delete_model(m2)
            assert am_actors(f"am-{key}-r") == {f"am-{key}-r0", f"am-{key}-r1"} and pg_states(f"am-{key}") == {"CREATED"}

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
            assert a["metrics"] == {"loss:sum": a["loss"], "clamped_positions:sum": 0, "swap_seconds:max": 0.0}, \
                "m was swapped in by at_init's forward; the passes behind it swap nothing"

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


def vllm_models(url):
    return {m["id"] for m in httpx.get(f"{url}/v1/models", timeout=10).json()["data"]}


def test_checkpoint_round_trip_through_fresh_models(tmp_path):
    """Save after one step, delete, create afresh: a weights-only load scores
    the saved loss and publishes v1 to vLLM; a load with the optimizer
    reproduces the uninterrupted run's next step, which a weights-only load
    does not."""
    x = datums((37, 64, 5))
    root = tmp_path / "weights" / "c1"
    root.mkdir(parents=True)

    async def model(name):
        h = await backend.create_model(name, f"req-{name}", MODEL, num_gpus=2, lora_config=LORA,
                                       native_root=tmp_path / f"native-{name}")
        routing.table.publish(name, routing.InferenceEndpoint(h.inference_endpoints))
        return h

    async def drop(h):
        await backend.delete_model(h)
        routing.table.withdraw(h.model_id)

    async def loss(h):
        return (await backend.forward(h, x, "cross_entropy"))["loss"]

    async def step(h):
        await backend.forward_backward(h, x, "cross_entropy")
        await backend.apply_optimizer_step(h, 1e-3, ADAM)

    async def go():
        h = await model("m")
        try:
            await step(h)
            l1 = await loss(h)
            await backend.save_checkpoint(h, root, step=1)
            assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {
                "hf_adapter/adapter_model.safetensors", "hf_adapter/adapter_config.json", "optimizer/adamw.safetensors"}
            await step(h)
            l2 = await loss(h)                  # the uninterrupted second step
        finally:
            await drop(h)

        h2 = await model("m2")
        try:
            await backend.load_checkpoint(h2, root, optimizer=False)
            assert h2.weight_version == 1 and h2.generation_synced_version == 1
            assert "m2@v1" in vllm_models(h2.inference_endpoints[0])
            assert await loss(h2) == pytest.approx(l1, rel=1e-3)
            prompt = h2.tokenizer.encode("The capital of France is")
            s = await backend.sample(h2, "s", prompt, 1, GREEDY)
            assert s["weight_version"] == 1 and s["latest_weight_version"] == 1
            await step(h2)
            l2_weights_only = await loss(h2)
        finally:
            await drop(h2)

        h3 = await model("m3")
        try:
            await backend.load_checkpoint(h3, root, optimizer=True)
            assert h3.weight_version == 1
            await step(h3)
            l2_resumed = await loss(h3)
        finally:
            await drop(h3)
        assert l2_resumed == pytest.approx(l2, rel=1e-3)
        assert abs(l2_weights_only - l2) > 10 * abs(l2_resumed - l2), (l2, l2_resumed, l2_weights_only)
        await backend.close()

    backend = AutomodelBackend(overrides=CONFIG)
    asyncio.run(go())


def test_two_tenants_share_a_group_and_match_their_solo_runs(tmp_path):
    """The multi-tenant gate: A and B (different seeds) on one base model and
    signature, forward_backward + optimizer step interleaved A, B, A, B, A, B.
    Every per-step loss and grad norm equals the tenant's solo run's exactly.
    After A's first step A samples v1 and B samples v0, both in vLLM; a save
    of B while A is live is B's published v0 export byte for byte. Prints the
    swap share of the shared run's wall time."""
    batches = [datums((37, 64, 5)), datums((80, 21, 50, 9, 33)), datums((37, 64, 5))]
    lora_a, lora_b = {**LORA, "seed": 1}, {**LORA, "seed": 2}
    backend = AutomodelBackend(overrides=CONFIG)

    async def model(name, lora, phase):
        h = await backend.create_model(name, f"req-{name}-{phase}", MODEL, num_gpus=2, lora_config=lora,
                                       native_root=tmp_path / f"native-{name}-{phase}")
        routing.table.publish(name, routing.InferenceEndpoint(h.inference_endpoints))
        return h

    async def drop(h):
        await backend.delete_model(h)
        routing.table.withdraw(h.model_id)

    async def one_step(h, batch):
        """(loss, grad_norm, swap seconds of the pass and the step)."""
        r = await backend.forward_backward(h, batch, "cross_entropy")
        s = await backend.apply_optimizer_step(h, 1e-3, ADAM)
        return r["loss"], s["grad_norm"], r["metrics"]["swap_seconds:max"] + s["metrics"]["swap_seconds"]

    async def solo(name, lora):
        h = await model(name, lora, "solo")
        try:
            return [(await one_step(h, batch))[:2] for batch in batches]
        finally:
            await drop(h)

    async def go():
        solo_a = await solo("a", lora_a)
        solo_b = await solo("b", lora_b)

        a = await model("a", lora_a, "shared")
        b = await model("b", lora_b, "shared")
        key = a.group_key
        assert b.group_key == key and am_actors(f"am-{key}-r") == {f"am-{key}-r0", f"am-{key}-r1"}
        shared_a, shared_b, swap, started = [], [], 0.0, time.monotonic()
        try:
            for i, batch in enumerate(batches):
                la, na, sa = await one_step(a, batch)
                if i == 0:
                    # A stepped, B has not: each samples its own version, both in vLLM
                    prompt = a.tokenizer.encode("The capital of France is")
                    assert (await backend.sample(a, "sa", prompt, 1, GREEDY))["weight_version"] == 1
                    assert (await backend.sample(b, "sb", prompt, 1, GREEDY))["weight_version"] == 0
                    assert {"a@v1", "b@v0"} <= vllm_models(a.inference_endpoints[0])
                    # A is live; a save of B reads B's slot: its published v0 export, byte for byte
                    root = tmp_path / "weights" / "b0"
                    root.mkdir(parents=True)
                    await backend.save_checkpoint(b, root, step=0)
                    v0 = b.native_root / "adapters" / "v0"
                    assert (root / "hf_adapter" / "adapter_model.safetensors").read_bytes() == (v0 / "adapter_model.safetensors").read_bytes()
                    assert (root / "hf_adapter" / "adapter_config.json").read_bytes() == (v0 / "adapter_config.json").read_bytes()
                lb, nb, sb = await one_step(b, batch)
                shared_a.append((la, na))
                shared_b.append((lb, nb))
                swap += sa + sb
            wall = time.monotonic() - started
            print(f"\nswap share of the shared run: {swap:.3f} s of {wall:.1f} s = {100 * swap / wall:.2f}%")
            assert swap > 0, "every op of the interleaved run but the first swaps its tenant in"
            assert shared_a == solo_a, f"A: shared {shared_a} vs solo {solo_a}"
            assert shared_b == solo_b, f"B: shared {shared_b} vs solo {solo_b}"
        finally:
            await drop(a)
            await drop(b)
        wait_until(lambda: not am_actors("am-"))
        assert pg_states(f"am-{key}") == {"REMOVED"}
        await backend.close()

    asyncio.run(go())
