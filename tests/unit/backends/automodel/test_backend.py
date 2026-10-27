"""AutomodelBackend over stand-ins for the ranks and for vLLM: per-request
results, the two fault paths (refused step vs lost group), request validation,
delete, the versioned adapter publish behind sample(), and the checkpoint
seam: what a save writes, what a load refuses before it
touches a rank, and the publish after a load."""
import asyncio
import json

import httpx
import pytest
import torch

from tinkercloud.training.backends.automodel.backend import AutomodelBackend, AutomodelHandle, _Group, _Sampler
from tinkercloud.training.backends.automodel.converter import LossFn
from tinkercloud.training.backends.automodel.errors import NonFiniteGradNorm, StateMismatch
from tinkercloud.training.backends.automodel.lora import LoraSpec
from tinkercloud.training.backends.automodel.scheduler import GroupScheduler
from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.http_pool import HttpClientPool
from tinkercloud.training.core import routing
from tinkercloud.training.models.requests import Datum
from tests.datums import ce_datum

CONFIG = dict(train_gpus=2, max_loras=1, max_lora_rank=8, max_cpu_loras=1, micro_batch_rows=2)
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
LORA = LoraSpec(rank=8, alpha=16, target_modules=("q_proj",), dropout=0.0)
VLLM = "http://vllm:1"
SP = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4}
run = asyncio.run


class Ranks:
    """TrainingGroup stand-in: every real position scores logprob -1 and term
    +1, so a row's loss is its length; optim_step reports the lr as the norm
    (waiting on `hold_step` when set), or raises `fault` once; so do
    add_tenant, remove_tenant and load_state; export_adapter writes a marker
    file; save_state writes the three checkpoint files (waiting on
    `hold_save` when set). `ops` records every call with its tenant."""
    num_ranks = 2

    def __init__(self):
        self.passes = []        # (kind, rows per rank, clip)
        self.ops = []           # (method, tenant), in call order
        self.fault = None
        self.stopped = False
        self.exports = []
        self.saves = []
        self.loads = []         # (adapter_dir, optimizer_dir)
        self.hold_save = None   # asyncio.Event a save waits for
        self.hold_step = None   # asyncio.Event a step waits for
        self.swap_seconds = 0.0

    def _maybe_fault(self):
        if self.fault is not None:
            fault, self.fault = self.fault, None
            raise fault

    async def add_tenant(self, tenant, seed):
        self.ops.append(("add_tenant", tenant))
        self._maybe_fault()

    async def remove_tenant(self, tenant):
        self.ops.append(("remove_tenant", tenant))
        self._maybe_fault()

    async def export_adapter(self, tenant, dest):
        self.ops.append(("export_adapter", tenant))
        dest.mkdir(parents=True, exist_ok=True)   # a version is re-exported when its load failed
        (dest / "adapter_model.safetensors").write_bytes(b"")
        self.exports.append(dest)

    async def save_state(self, tenant, root):
        self.ops.append(("save_state", tenant))
        if self.hold_save is not None:
            await self.hold_save.wait()
        write_root(root)
        self.saves.append(root)

    async def load_state(self, tenant, adapter_dir, optimizer_dir):
        self.ops.append(("load_state", tenant))
        self._maybe_fault()
        self.loads.append((adapter_dir, optimizer_dir))

    async def forward_backward(self, tenant, shards, clip):
        return self._run("forward_backward", tenant, shards, clip)

    async def forward(self, tenant, shards, clip):
        return self._run("forward", tenant, shards, clip)

    def _run(self, kind, tenant, shards, clip):
        self.ops.append((kind, tenant))
        self.passes.append((kind, [s.num_rows for s in shards], clip))
        outs = []
        for s in shards:
            valid = torch.arange(s.input_ids.shape[1])[None, :] < s.lengths[:, None]
            outs.append({
                "row_loss": s.lengths.float(),
                "logprobs": -valid.float(),
                "elementwise": valid.float(),
                "clamped": torch.zeros(s.num_rows),
                "swap_seconds": self.swap_seconds,
            })
        return outs

    async def optim_step(self, tenant, adam):
        self.ops.append(("optim_step", tenant))
        if self.hold_step is not None:
            await self.hold_step.wait()
        self._maybe_fault()
        return [{"grad_norm": adam.learning_rate, "swap_seconds": self.swap_seconds}] * self.num_ranks

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


ADAPTER_CONFIG = {"peft_type": "LORA", "r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}


def write_root(root, config=ADAPTER_CONFIG, optimizer=True):
    """A checkpoint root as the worker writes it (empty tensors: the stand-in
    never reads them), or without optimizer state as another backend writes it."""
    (root / "hf_adapter").mkdir(parents=True)
    (root / "hf_adapter" / "adapter_model.safetensors").write_bytes(b"")
    (root / "hf_adapter" / "adapter_config.json").write_text(json.dumps(config))
    if optimizer:
        (root / "optimizer").mkdir()
        (root / "optimizer" / "adamw.safetensors").write_bytes(b"")
    return root


def files_under(path):
    return {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}


def wired(tmp_path=None, vllm=None, tenants=("m",), group_tenants=("m",)):
    """A backend with one group "g" serving `group_tenants` on a Ranks stand-in
    and, given tmp_path, a sampler "s" for their base, serving `tenants`, on a
    Vllm stand-in. Returns the handle of "m" (route it with `routed` before a
    sample); `tenant_handle` makes the others. Call inside a running loop:
    the scheduler starts its worker task."""
    backend = AutomodelBackend(overrides=CONFIG)
    ranks = Ranks()
    scheduler = GroupScheduler("g", lambda pk, payloads: backend._run_pass("g", pk, payloads))
    backend._groups["g"] = _Group("g", ranks, scheduler, set(group_tenants))
    handle = AutomodelHandle(model_id="m", base_model="b", lora=LORA, group_key="g")
    if tmp_path is not None:
        vllm = vllm or Vllm()
        backend._http_pool = HttpClientPool(transport=vllm.transport)
        backend._samplers["s"] = _Sampler("s", Host(), VLLM, set(tenants))
        handle.sampler_key, handle.native_root, handle.tokenizer = "s", tmp_path, Tokenizer()
    return backend, ranks, handle


def tenant_handle(handle, model_id):
    """Another tenant of `handle`'s group and sampler, with its own native root."""
    root = None if handle.native_root is None else handle.native_root / model_id
    return AutomodelHandle(model_id=model_id, base_model=handle.base_model, lora=handle.lora, group_key=handle.group_key,
                           sampler_key=handle.sampler_key, native_root=root, tokenizer=handle.tokenizer)


def creatable(monkeypatch):
    """A backend whose create_model runs on stand-ins: every TrainingGroup it
    starts is a Ranks (collected in the returned list), the sampler is a host
    that answers with VLLM's URL, the HF config read and the tokenizer are
    constants. The Ray sweep is skipped."""
    import sys
    import types

    from tinkercloud.training.backends.automodel import backend as mod

    started = []

    class Group(Ranks):
        def __init__(self, key, num_ranks):
            super().__init__()
            self.key = key
            started.append(self)

        async def start(self, base_model, lora, micro_batch_rows):
            return 42

    class Sampler(Host):
        def __init__(self, key):
            super().__init__()
            self.key = key

        async def start(self, base_model, max_model_len, config):
            return VLLM

    monkeypatch.setattr(mod, "TrainingGroup", Group)
    monkeypatch.setattr(mod, "SamplerGroup", Sampler)
    monkeypatch.setattr(mod, "sweep_leftovers", lambda: 0)
    monkeypatch.setattr(mod, "read_raw_hf_config", lambda base_model: {})
    monkeypatch.setattr(mod, "classification_spec", lambda raw: None)
    monkeypatch.setattr(mod, "model_context_length", lambda raw: 2048)
    tokenizers = types.SimpleNamespace(from_pretrained=lambda base_model: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=tokenizers))
    return AutomodelBackend(overrides=CONFIG), started


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
            backend.forward_backward(handle, [ce_datum([1, 2, 3]), ce_datum([1, 2, 3, 4, 5])], "cross_entropy"),
            backend.forward_backward(handle, [ce_datum([7, 8])], "cross_entropy"),
        )
        assert a["loss"] == 8.0 and b["loss"] == 2.0
        assert a["loss_fn_output_type"] == LossFn.CROSS_ENTROPY.value
        assert a["metrics"] == {"loss:sum": 8.0, "clamped_positions:sum": 0, "swap_seconds:max": 0.0}
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
        ranks.swap_seconds = 0.25
        result = await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert result["grad_norm"] == 0.5 and result["weight_version"] == 1
        assert result["swap_seconds"] == 0.25 and result["metrics"] == {"swap_seconds": 0.25}
        assert (await backend.forward(handle, [ce_datum([1, 2])], "cross_entropy"))["metrics"]["swap_seconds:max"] == 0.25
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
            await backend.forward_backward(handle, [ce_datum([1, 2])], "cross_entropy")
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
            await backend.forward_backward(handle, [ce_datum([1, 2])], "dro")
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
            await backend.forward(handle, [ce_datum([1, 2])], "cross_entropy")
    run(go())


# --- tenants of one group ----------------------------------------------------

LORA_A = {"rank": 8, "alpha": 16}


def test_tenant_create_joins_the_group_or_opens_its_own_group(monkeypatch, tmp_path):
    backend, started = creatable(monkeypatch)

    async def go():
        a = await backend.create_model("a", "r1", "base", 2, lora_config=LORA_A, native_root=tmp_path / "a")
        b = await backend.create_model("b", "r2", "base", 2, lora_config=LORA_A, native_root=tmp_path / "b")
        assert a.group_key == b.group_key and len(started) == 1, "the same signature joins the group"
        assert backend._groups[a.group_key].tenants == {"a", "b"}
        assert started[0].ops == [("add_tenant", "a"), ("add_tenant", "b")]
        c = await backend.create_model("c", "r3", "base", 2, lora_config={**LORA_A, "alpha": 32}, native_root=tmp_path / "c")
        assert c.group_key != a.group_key and len(started) == 2, "another signature opens its own group"
        assert started[1].ops == [("add_tenant", "c")] and backend._groups[c.group_key].tenants == {"c"}
        assert a.sampler_key == b.sampler_key == c.sampler_key, "one vLLM per base model"
        # every pass names its tenant to the one group
        await backend.forward_backward(a, [ce_datum([1, 2])], "cross_entropy")
        assert (await backend.forward_backward(b, [ce_datum([1, 2, 3])], "cross_entropy"))["loss"] == 3.0
        assert started[0].ops[2:] == [("forward_backward", "a"), ("forward_backward", "b")]
        await backend.close()
    run(go())


def test_tenant_failed_joins_leave_the_group_untouched(monkeypatch, tmp_path):
    backend, started = creatable(monkeypatch)

    async def go():
        a = await backend.create_model("a", "r1", "base", 2, lora_config=LORA_A, native_root=tmp_path / "a")
        started[0].fault = RuntimeError("pinned memory exhausted")
        with pytest.raises(BackendError, match="model b could not join training group .*pinned memory exhausted"):
            await backend.create_model("b", "r2", "base", 2, lora_config=LORA_A, native_root=tmp_path / "b")
        group = backend._groups[a.group_key]
        assert group.tenants == {"a"} and group.lost is None and not started[0].stopped
        assert (await backend.forward(a, [ce_datum([1, 2])], "cross_entropy"))["loss"] == 2.0
        await backend.close()
    run(go())


def test_tenant_delete_frees_the_slot_and_the_last_delete_stops_the_ranks(tmp_path):
    async def go():
        vllm = Vllm()
        backend, ranks, m = wired(tmp_path, vllm, tenants=("m", "b"), group_tenants=("m", "b"))
        b = tenant_handle(m, "b")
        host = backend._samplers["s"].host
        await backend.prepare_for_generation(m)
        await backend.prepare_for_generation(b)
        assert set(vllm.loaded) == {"m@v0", "b@v0"}

        await backend.delete_model(m)
        assert not ranks.stopped and ranks.ops[-1] == ("remove_tenant", "m")
        group = backend._groups["g"]
        assert group.tenants == {"b"} and group.lost is None
        assert vllm.unloaded == ["m@v0"] and set(vllm.loaded) == {"b@v0"} and not host.stopped
        assert (await backend.forward(b, [ce_datum([1, 2, 3])], "cross_entropy"))["loss"] == 3.0
        with pytest.raises(BackendError, match="model m has no training group: the model was deleted"):
            await backend.forward(m, [ce_datum([1, 2])], "cross_entropy")
        assert ranks.ops[-1] == ("forward", "b"), "a deleted tenant's handle never reaches the ranks"

        await backend.delete_model(b)
        assert ranks.stopped and "g" not in backend._groups and host.stopped
        await backend.close()
    run(go())


def test_tenant_fault_in_one_pass_fails_every_queued_tenant_and_names_the_cause_after():
    async def go():
        backend, ranks, a = wired(group_tenants=("a", "b"))
        a.model_id = "a"
        b = tenant_handle(a, "b")
        ranks.hold_step = asyncio.Event()
        ranks.fault = RuntimeError("CUDA error")
        step_a = asyncio.create_task(backend.apply_optimizer_step(a, 0.5, ADAM))
        await asyncio.sleep(0.02)
        pass_b = asyncio.create_task(backend.forward_backward(b, [ce_datum([1, 2])], "cross_entropy"))
        await asyncio.sleep(0.02)
        assert ranks.passes == [], "b's pass waits behind a's step"
        ranks.hold_step.set()
        with pytest.raises(BackendError, match="released"):
            await step_a
        with pytest.raises(BackendError, match="CUDA error"):
            await pass_b
        await until_stopped(ranks)
        assert ranks.passes == [], "the queued pass never ran"
        for handle in (a, b):
            with pytest.raises(BackendError, match="CUDA error"):
                await backend.forward(handle, [ce_datum([1, 2])], "cross_entropy")
        await backend.close()
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
        await backend.forward_backward(handle, [ce_datum([1, 2])], "cross_entropy")
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


# --- checkpoints ------------------------------------------------------------

def test_checkpoint_ephemeral_save_writes_nothing_and_calls_no_rank(tmp_path):
    async def go():
        backend, ranks, handle = wired()
        root = tmp_path / "sampler_weights" / "e1"
        root.mkdir(parents=True)
        await backend.save_checkpoint(handle, root, step=None, persist=False)
        assert files_under(tmp_path) == set() and ranks.saves == []
        await backend.close()
    run(go())


def test_checkpoint_save_writes_the_three_files_under_the_root_as_one_barrier(tmp_path):
    async def go():
        backend, ranks, handle = wired()
        root = tmp_path / "weights" / "c1"
        root.mkdir(parents=True)
        ranks.hold_save = asyncio.Event()
        save = asyncio.create_task(backend.save_checkpoint(handle, root, step=1))
        await asyncio.sleep(0.02)
        fwd = asyncio.create_task(backend.forward_backward(handle, [ce_datum([1, 2])], "cross_entropy"))
        await asyncio.sleep(0.02)
        assert ranks.passes == [], "a pass waits behind the save barrier"
        ranks.hold_save.set()
        await save
        assert (await fwd)["loss"] == 2.0
        assert ranks.saves == [root]
        assert files_under(tmp_path) == {"weights/c1/hf_adapter/adapter_model.safetensors",
                                         "weights/c1/hf_adapter/adapter_config.json",
                                         "weights/c1/optimizer/adamw.safetensors"}
        await backend.close()
    run(go())


def test_checkpoint_save_on_a_rank_fault_releases_the_group(tmp_path):
    async def go():
        backend, ranks, handle = wired()
        root = tmp_path / "weights" / "c1"
        root.mkdir(parents=True)
        ranks.hold_save = asyncio.Event()
        ranks.hold_save.set()
        ranks.save_state = _raise(RuntimeError("CUDA error"))
        with pytest.raises(BackendError, match="released"):
            await backend.save_checkpoint(handle, root, step=1)
        await until_stopped(ranks)
    run(go())


def _raise(exc):
    async def fail(*args):
        raise exc
    return fail


def test_checkpoint_load_refuses_a_root_without_adapter_or_for_another_lora(tmp_path):
    async def go():
        backend, ranks, handle = wired(tmp_path)
        empty = tmp_path / "weights" / "empty"
        empty.mkdir(parents=True)
        with pytest.raises(BackendError, match="holds no hf_adapter/adapter_model.safetensors"):
            await backend.load_checkpoint(handle, empty)
        cases = (({**ADAPTER_CONFIG, "r": 4}, r"r: 4 vs 8"),
                 ({**ADAPTER_CONFIG, "lora_alpha": 32}, r"lora_alpha: 32 vs 16"),
                 ({**ADAPTER_CONFIG, "target_modules": ["q_proj", "v_proj"]}, r"target_modules: .*v_proj"),
                 ({"peft_type": "LORA", "r": 8}, r"lora_alpha: absent"),
                 # the wrong type is a mismatch that names the field, never a crash
                 ({**ADAPTER_CONFIG, "r": 8.5}, r"r: 8\.5 vs 8"),
                 ({**ADAPTER_CONFIG, "r": "abc"}, r"r: 'abc' vs 8"),
                 ({**ADAPTER_CONFIG, "r": True}, r"r: True vs 8"),
                 ({**ADAPTER_CONFIG, "lora_alpha": None}, r"lora_alpha: None vs 16"),
                 ({**ADAPTER_CONFIG, "target_modules": None}, r"target_modules: None vs"),
                 ({**ADAPTER_CONFIG, "target_modules": "q_proj"}, r"target_modules: 'q_proj' vs"),
                 ([1, 2], r"adapter_config.json is not a JSON object"))
        for i, (config, field) in enumerate(cases):
            root = write_root(tmp_path / "weights" / f"other{i}", config)
            with pytest.raises(BackendError, match=f"another LoRA \\({field}"):
                await backend.load_checkpoint(handle, root)
        assert ranks.loads == [] and handle.weight_version == 0
        fits = write_root(tmp_path / "weights" / "fits")
        await backend.load_checkpoint(handle, fits)   # the model is still usable
        assert ranks.loads == [(fits / "hf_adapter", None)]
        # the float 16.0 names the same alpha
        await backend.load_checkpoint(handle, write_root(tmp_path / "weights" / "float", {**ADAPTER_CONFIG, "lora_alpha": 16.0}))
        assert len(ranks.loads) == 2
        assert (await backend.forward(handle, [ce_datum([1, 2, 3])], "cross_entropy"))["loss"] == 3.0
        await backend.close()
    run(go())


def test_checkpoint_load_refuses_keys_that_do_not_fit_and_keeps_the_group(tmp_path):
    async def go():
        backend, ranks, handle = wired(tmp_path)
        root = write_root(tmp_path / "weights" / "c1")
        ranks.fault = StateMismatch("adapter keys are not this model's LoRA parameters: 2 missing")
        with pytest.raises(BackendError, match="does not fit this model: adapter keys"):
            await backend.load_checkpoint(handle, root)
        assert not ranks.stopped and handle.weight_version == 0
        ranks.fault = RuntimeError("CUDA error")
        with pytest.raises(BackendError, match="released"):
            await backend.load_checkpoint(handle, root)
        await until_stopped(ranks)
    run(go())


def test_checkpoint_load_with_optimizer_needs_optimizer_state(tmp_path):
    async def go():
        backend, ranks, handle = wired(tmp_path)
        foreign = write_root(tmp_path / "weights" / "foreign", optimizer=False)
        with pytest.raises(BackendError, match="carries no optimizer state"):
            await backend.load_checkpoint(handle, foreign, optimizer=True)
        assert ranks.loads == [] and handle.weight_version == 0
        await backend.load_checkpoint(handle, foreign, optimizer=False)
        assert ranks.loads == [(foreign / "hf_adapter", None)]
        own = write_root(tmp_path / "weights" / "own")
        await backend.load_checkpoint(handle, own, optimizer=True)
        assert ranks.loads[-1] == (own / "hf_adapter", own / "optimizer")
        await backend.close()
    run(go())


def test_checkpoint_load_publishes_the_next_version_before_it_returns(tmp_path, routed):
    async def go():
        vllm = Vllm()
        backend, ranks, handle = wired(tmp_path, vllm)
        await backend.prepare_for_generation(handle)   # v0 is in vLLM
        root = write_root(tmp_path / "weights" / "c1")
        await backend.load_checkpoint(handle, root)
        assert handle.weight_version == 1 and handle.generation_synced_version == 1
        assert ranks.exports[-1] == tmp_path / "adapters" / "v1" and set(vllm.loaded) == {"m@v1"}
        assert vllm.unloaded == ["m@v0"] and handle.loaded_versions == [1]
        res = await backend.sample(handle, "r", [1], 1, SP)
        assert res["weight_version"] == 1 and res["latest_weight_version"] == 1
        assert res["sequences"][0]["tokens"] == [1, 7]
        # a model whose v0 was never published: the load publishes v1 only
        vllm2 = Vllm()
        backend2, ranks2, handle2 = wired(tmp_path, vllm2)
        await backend2.load_checkpoint(handle2, root)
        assert handle2.weight_version == 1 and set(vllm2.loaded) == {"m@v1"} and vllm2.unloaded == []
        await backend.close()
        await backend2.close()
    run(go())


def test_checkpoint_load_whose_publish_fails_reports_the_load_as_done(tmp_path, routed):
    async def go():
        vllm = Vllm(fail_load=True)
        backend, ranks, handle = wired(tmp_path, vllm)
        root = write_root(tmp_path / "weights" / "c1")
        with pytest.raises(BackendError, match=r"loaded \(weights at v1\).*500"):
            await backend.load_checkpoint(handle, root)
        assert ranks.loads == [(root / "hf_adapter", None)] and handle.weight_version == 1
        assert not ranks.stopped, "a vLLM fault is not the group's"
        vllm.fail_load = False
        assert (await backend.sample(handle, "r", [1], 1, SP))["weight_version"] == 1
        await backend.close()
    run(go())


def test_external_weights_export_leaves_the_adapter_at_the_root(tmp_path):
    async def go():
        backend, ranks, handle = wired()
        root = tmp_path / "external_weights" / "x"
        root.mkdir(parents=True)
        await backend.export_external_weights(handle, root, step=1)
        assert ranks.saves == [root / ".native"]
        assert files_under(root) == {"adapter_model.safetensors", "adapter_config.json"}
        await backend.close()
    run(go())
