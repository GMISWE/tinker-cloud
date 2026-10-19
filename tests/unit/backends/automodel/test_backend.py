"""AutomodelBackend over stand-ins for the ranks and for vLLM: per-request
results, the two fault paths (refused step vs lost group), request validation,
delete, and the versioned adapter publish behind sample() (R4, D20)."""
import asyncio
import json

import httpx
import pytest
import torch

from tinkercloud.training.backends.automodel.backend import AutomodelBackend, AutomodelHandle, _Group, _Sampler
from tinkercloud.training.backends.automodel.converter import LossFn
from tinkercloud.training.backends.automodel.errors import NonFiniteGradNorm
from tinkercloud.training.backends.automodel.lora import LoraSpec
from tinkercloud.training.backends.automodel.scheduler import GroupScheduler
from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.http_pool import HttpClientPool
from tinkercloud.training.core import routing
from tinkercloud.training.models.requests import Datum

CONFIG = dict(train_gpus=2, max_loras=1, max_lora_rank=8, max_cpu_loras=1, micro_batch_rows=2)
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
LORA = LoraSpec(rank=8, alpha=16, target_modules=("q_proj",), dropout=0.0)
VLLM = "http://vllm:1"
SP = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4}
run = asyncio.run


def ce(tokens):
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": tokens}, "weights": {"data": [1.0] * len(tokens)}}})


class Ranks:
    """TrainingGroup stand-in: every real position scores logprob -1 and term
    +1, so a row's loss is its length; optim_step reports the lr as the norm,
    or raises `fault` once; export_adapter writes a marker file."""
    num_ranks = 2

    def __init__(self):
        self.passes = []        # (kind, rows per rank, clip)
        self.fault = None
        self.stopped = False
        self.exports = []

    async def export_adapter(self, dest):
        dest.mkdir(parents=True, exist_ok=True)   # a version is re-exported when its load failed
        (dest / "adapter_model.safetensors").write_bytes(b"")
        self.exports.append(dest)

    async def forward_backward(self, shards, clip):
        return self._run("forward_backward", shards, clip)

    async def forward(self, shards, clip):
        return self._run("forward", shards, clip)

    def _run(self, kind, shards, clip):
        self.passes.append((kind, [s.num_rows for s in shards], clip))
        outs = []
        for s in shards:
            valid = torch.arange(s.input_ids.shape[1])[None, :] < s.lengths[:, None]
            outs.append({
                "row_loss": s.lengths.float(),
                "logprobs": -valid.float(),
                "elementwise": valid.float(),
                "clamped": torch.zeros(s.num_rows),
            })
        return outs

    async def optim_step(self, adam):
        if self.fault is not None:
            fault, self.fault = self.fault, None
            raise fault
        return [{"grad_norm": adam.learning_rate}] * self.num_ranks

    async def stop(self):
        self.stopped = True


class Host:
    """SamplerGroup stand-in."""
    def __init__(self):
        self.stopped = False

    async def stop(self):
        self.stopped = True


class Tokenizer:
    def decode(self, toks):
        return " ".join(map(str, toks))


class Vllm:
    """The base model's vLLM: the stock LoRA routes and the plugin's generate
    route over an httpx MockTransport. generate answers one choice [served
    version, 7] and waits on `hold[version]` when the test set one."""
    def __init__(self, fail_load=False):
        self.loaded = {}        # adapter name -> path
        self.unloaded = []
        self.generated = []     # model names, in request order
        self.hold = {}          # adapter name -> asyncio.Event a generate waits for
        self.fail_load = fail_load
        self.transport = httpx.MockTransport(self.handle)

    async def handle(self, request):
        body = json.loads(request.read())
        if request.url.path == "/v1/load_lora_adapter":
            if self.fail_load:
                return httpx.Response(500, text="engine gone")
            self.loaded[body["lora_name"]] = body["lora_path"]
            return httpx.Response(200, text="Success")
        if request.url.path == "/v1/unload_lora_adapter":
            del self.loaded[body["lora_name"]]
            self.unloaded.append(body["lora_name"])
            return httpx.Response(200, text="Success")
        assert request.url.path == "/tinkercloud/v1/generate"
        name = body["model"]
        if name not in self.loaded:
            return httpx.Response(404, json={"error": f"model {name!r} is neither the base model nor a loaded adapter"})
        self.generated.append(name)
        if name in self.hold:
            await self.hold[name].wait()
        version = int(name.rsplit("@v", 1)[1])
        return httpx.Response(200, json={"request_id": "r", "prompt_logprobs": None, "choices": [
            {"index": 0, "finish_reason": "length", "token_ids": [version, 7],
             "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.0}]}}]})


def wired(tmp_path=None, vllm=None, tenants=("m",)):
    """A backend with one group "g" serving model "m" on a Ranks stand-in and,
    given tmp_path, a sampler "s" for its base on a Vllm stand-in (routed for
    "m"). Call inside a running loop: the scheduler starts its worker task."""
    backend = AutomodelBackend(overrides=CONFIG)
    ranks = Ranks()
    scheduler = GroupScheduler("g", lambda pk, payloads: backend._run_pass("g", pk, payloads))
    backend._groups["g"] = _Group("g", ranks, scheduler, {"m"})
    handle = AutomodelHandle(model_id="m", base_model="b", lora=LORA, group_key="g")
    if tmp_path is not None:
        vllm = vllm or Vllm()
        backend._http_pool = HttpClientPool(transport=vllm.transport)
        backend._samplers["s"] = _Sampler("s", Host(), VLLM, set(tenants))
        handle.sampler_key, handle.native_root, handle.tokenizer = "s", tmp_path, Tokenizer()
    return backend, ranks, handle


@pytest.fixture
def routed():
    routing.table.publish("m", routing.InferenceEndpoint((VLLM,)))
    yield
    routing.table.withdraw("m")


async def until_stopped(ranks, tries=100):
    for _ in range(tries):
        if ranks.stopped:
            return
        await asyncio.sleep(0)
    raise AssertionError("the ranks were not stopped")


def test_pass_results_are_split_per_request():
    async def go():
        backend, ranks, handle = wired()
        a, b = await asyncio.gather(
            backend.forward_backward(handle, [ce([1, 2, 3]), ce([1, 2, 3, 4, 5])], "cross_entropy"),
            backend.forward_backward(handle, [ce([7, 8])], "cross_entropy"),
        )
        assert a["loss"] == 8.0 and b["loss"] == 2.0
        assert a["loss_fn_output_type"] == LossFn.CROSS_ENTROPY.value
        assert a["metrics"] == {"loss:sum": 8.0, "clamped_positions:sum": 0}
        assert [o["logprobs"]["shape"] for o in a["loss_fn_outputs"]] == [[3], [5]]
        assert a["loss_fn_outputs"][1]["logprobs"]["data"] == [-1.0] * 5
        assert a["loss_fn_outputs"][0]["elementwise_loss"]["data"] == [1.0] * 3
        assert b["loss_fn_outputs"][0]["logprobs"] == {"data": [-1.0, -1.0], "shape": [2], "dtype": "float32"}
        for kind, rows, clip in ranks.passes:
            assert kind == "forward_backward" and clip is None
            assert len(set(rows)) == 1, "every rank runs the same number of rows"
        await backend.close()
    run(go())


def test_supported_losses_are_the_wire_strings():
    assert AutomodelBackend.SUPPORTED_LOSS_FNS == {"cross_entropy", "importance_sampling", "ppo"}


def test_nonfinite_step_fails_the_request_and_keeps_the_group(tmp_path):
    async def go():
        backend, ranks, handle = wired(tmp_path)   # the step that succeeds publishes v1
        ranks.fault = NonFiniteGradNorm("grad norm is inf")
        with pytest.raises(BackendError, match="grad norm is inf"):
            await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert not ranks.stopped and handle.weight_version == 0
        result = await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert result["grad_norm"] == 0.5 and result["weight_version"] == 1
        await backend.close()
    run(go())


def test_rank_fault_releases_the_group():
    async def go():
        backend, ranks, handle = wired()
        ranks.fault = RuntimeError("CUDA error")
        with pytest.raises(BackendError, match="released"):
            await backend.apply_optimizer_step(handle, 0.5, ADAM)
        await until_stopped(ranks)
        assert "g" not in backend._groups
        with pytest.raises(BackendError, match="CUDA error"):
            await backend.forward_backward(handle, [ce([1, 2])], "cross_entropy")
        with pytest.raises(BackendError, match="CUDA error"):
            await backend.sample(handle, "r", [1, 2], 1, SP)
        await backend.delete_model(handle)   # nothing left to free; not an error
    run(go())


def test_bad_request_fails_before_the_ranks():
    async def go():
        backend, ranks, handle = wired()
        no_weights = Datum.model_validate({"model_input": {"tokens": [1, 2]},
                                           "loss_fn_inputs": {"target_tokens": {"data": [1, 2]}}})
        with pytest.raises(BackendError, match="missing weights"):
            await backend.forward_backward(handle, [no_weights], "cross_entropy")
        with pytest.raises(BackendError, match="not served"):
            await backend.forward_backward(handle, [ce([1, 2])], "dro")
        with pytest.raises(BackendError, match="adam_params"):
            await backend.apply_optimizer_step(handle)
        assert ranks.passes == []
        await backend.close()
    run(go())


def test_delete_closes_the_scheduler_and_stops_the_ranks():
    async def go():
        backend, ranks, handle = wired()
        await backend.delete_model(handle)
        assert ranks.stopped and "g" not in backend._groups
        with pytest.raises(BackendError, match="deleted"):
            await backend.forward(handle, [ce([1, 2])], "cross_entropy")
    run(go())



def test_first_sample_publishes_v0_and_later_samples_reuse_it(tmp_path, routed):
    async def go():
        backend, ranks, handle = wired(tmp_path)
        vllm = Vllm(); backend._http_pool = HttpClientPool(transport=vllm.transport)
        res = await backend.sample(handle, "r", [1, 2, 3], 1, SP)
        assert ranks.exports == [tmp_path / "adapters" / "v0"]
        assert vllm.loaded == {"m@v0": str(tmp_path / "adapters" / "v0")}
        assert res["sequences"] == [{"tokens": [0, 7], "logprobs": [-0.5, -1.0], "text": "0 7", "stop_reason": "length"}]
        assert res["weight_version"] == 0 and res["latest_weight_version"] == 0
        await backend.sample(handle, "r2", [1], 1, SP)
        await backend.prepare_for_generation(handle)
        await backend.update_inference_weights(handle)
        assert len(ranks.exports) == 1 and vllm.generated == ["m@v0", "m@v0"]
        assert handle.loaded_versions == [0] and handle.in_flight == {0: 0}
        await backend.close()
    run(go())


def test_step_publishes_the_next_version_and_retires_the_old_one_after_its_samples(tmp_path, routed):
    async def go():
        backend, ranks, handle = wired(tmp_path)
        vllm = Vllm(); backend._http_pool = HttpClientPool(transport=vllm.transport)
        await backend.prepare_for_generation(handle)
        vllm.hold["m@v0"] = asyncio.Event()
        early = asyncio.create_task(backend.sample(handle, "r0", [1], 1, SP))
        await asyncio.sleep(0.05)    # r0 is at the engine on v0
        step = await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert step["weight_version"] == 1
        assert ranks.exports[-1] == tmp_path / "adapters" / "v1"
        assert set(vllm.loaded) == {"m@v0", "m@v1"}, "v0 stays while r0 runs on it"
        late = await backend.sample(handle, "r1", [1], 1, SP)
        assert late["weight_version"] == 1 and late["sequences"][0]["tokens"] == [1, 7]
        vllm.hold["m@v0"].set()
        res = await early
        assert res["weight_version"] == 0 and res["latest_weight_version"] == 1
        assert vllm.unloaded == ["m@v0"] and set(vllm.loaded) == {"m@v1"}
        assert not (tmp_path / "adapters" / "v0").exists() and (tmp_path / "adapters" / "v1").is_dir()
        assert handle.loaded_versions == [1]
        await backend.close()
    run(go())


def test_a_step_whose_publish_fails_reports_the_step_as_taken(tmp_path, routed):
    async def go():
        vllm = Vllm(fail_load=True)
        backend, ranks, handle = wired(tmp_path, vllm)
        await backend.forward_backward(handle, [ce([1, 2])], "cross_entropy")
        with pytest.raises(BackendError, match=r"step was taken \(weights at v1\).*500"):
            await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert handle.weight_version == 1 and handle.generation_synced_version == -1
        assert not ranks.stopped, "a vLLM fault is not the group's"
        vllm.fail_load = False
        res = await backend.sample(handle, "r", [1], 1, SP)   # the next publish recovers
        assert res["weight_version"] == 1 and set(vllm.loaded) == {"m@v1"}
        await backend.close()
    run(go())


def test_delete_unloads_the_adapters_and_the_last_tenant_stops_vllm(tmp_path, routed):
    async def go():
        vllm = Vllm()
        backend, ranks, handle = wired(tmp_path, vllm, tenants=("m", "other"))
        host = backend._samplers["s"].host
        await backend.prepare_for_generation(handle)
        await backend.delete_model(handle)
        assert ranks.stopped and vllm.unloaded == ["m@v0"] and not host.stopped
        assert backend._samplers["s"].tenants == {"other"}
        assert not (tmp_path / "adapters").exists()

        backend, ranks, handle = wired(tmp_path, Vllm())
        host = backend._samplers["s"].host
        await backend.prepare_for_generation(handle)
        await backend.delete_model(handle)
        assert host.stopped and "s" not in backend._samplers
        await backend.close()
    run(go())
