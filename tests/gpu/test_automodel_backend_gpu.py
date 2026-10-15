"""AutomodelBackend on real GPUs (Qwen2.5-0.5B-Instruct, LoRA rank 8): one
tenant's lifecycle through the backend, and the leftover sweep (R5).

Needs >= 2 GPUs, the Automodel training env and the model in the HF cache;
starts its own Ray instance. Skipped elsewhere.

Run on a GPU pod (about 2 minutes):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_backend_gpu.py -v

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
import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import ray  # noqa: E402
import torch  # noqa: E402

if torch.cuda.device_count() < 2:
    pytest.skip("needs >= 2 GPUs", allow_module_level=True)

from ray.util.placement_group import placement_group, placement_group_table  # noqa: E402

from tinkercloud.training.backends.automodel.backend import AutomodelBackend  # noqa: E402
from tinkercloud.training.backends.automodel.group import sweep_leftovers  # noqa: E402
from tinkercloud.training.backends.base import BackendError  # noqa: E402
from tinkercloud.training.models.requests import Datum  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONFIG = dict(train_gpus=2, max_loras=1, max_lora_rank=8, max_cpu_loras=1, micro_batch_rows=2)
LORA = {"rank": 8, "alpha": 16}
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
BACKEND_LOGGER = "tinkercloud.training.backends.automodel.backend"


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    ray.init(num_gpus=torch.cuda.device_count(), runtime_env={"env_vars": {
        "PYTHONPATH": os.environ["PYTHONPATH"], "HF_HUB_OFFLINE": "1"}})
    yield
    ray.shutdown()


def ce_datum(g, length, weight=1.0):
    """Random tokens; each position predicts the next (the last wraps), with
    the first third of the positions weighted 0."""
    tokens = torch.randint(100, 20000, (length,), generator=g).tolist()
    targets = tokens[1:] + [tokens[0]]
    weights = [0.0] * (length // 3) + [weight] * (length - length // 3)
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": targets}, "weights": {"data": weights}}})


def datums(lengths, weight=1.0):
    g = torch.Generator().manual_seed(0)
    return [ce_datum(g, n, weight) for n in lengths]


def am_actors(prefix):
    return {n for n in ray.util.list_named_actors() if n.startswith(prefix)}


def pg_states(name):
    return {info["state"] for info in placement_group_table().values() if info["name"] == name}


def wait_until(cond, seconds=15):
    deadline = time.monotonic() + seconds
    while not cond():
        assert time.monotonic() < deadline, "Ray did not reach the expected state in time"
        time.sleep(0.5)


def test_lifecycle_through_the_backend(caplog):
    caplog.set_level(logging.INFO, logger=BACKEND_LOGGER)
    x, y = datums((37, 64, 5)), datums((80, 21, 50, 9, 33))

    async def go():
        backend = AutomodelBackend(overrides=CONFIG)
        handle = await backend.create_model("m", "req-1", MODEL, num_gpus=2, lora_config=LORA)
        key = handle.group_key
        try:
            assert handle.context_length is not None and handle.context_length > 0
            assert handle.weight_version == 0
            assert am_actors(f"am-{key}-r") == {f"am-{key}-r0", f"am-{key}-r1"}
            assert pg_states(f"am-{key}") == {"CREATED"}

            with pytest.raises(BackendError, match="already serves m"):
                await backend.create_model("m2", "req-2", MODEL, num_gpus=2, lora_config=LORA)

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
            assert a["metrics"] == {"loss:sum": a["loss"], "clamped_positions": 0}

            step = await backend.apply_optimizer_step(handle, 1e-3, ADAM)
            assert step["grad_norm"] > 0
            assert step["weight_version"] == handle.weight_version == 1
            after = await backend.forward(handle, x, "cross_entropy")
            assert after["loss"] != at_init["loss"]   # the step moved the adapter

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

        wait_until(lambda: not am_actors(f"am-{key}-r"))
        assert pg_states(f"am-{key}") == {"REMOVED"}
        with pytest.raises(BackendError, match="deleted"):
            await backend.forward(handle, x, "cross_entropy")

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
