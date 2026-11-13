"""The verl backend's one sampler and the passes of the server's models: a pass of any model runs
with the sampler asleep, after its in-flight samples, and a sample wakes it only when no pass runs.

The workers and the sampler are stand-ins that record the order of their calls; a stand-in can hold a
call open until the test releases it."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("verl", reason="the backend module imports verl's types")

from tinkercloud.training.backends.verl import backend as verl_backend
from tinkercloud.training.backends.verl.backend import VerlBackend, VerlHandle
from tinkercloud.training.models.requests import AdamParams

ADAM = AdamParams(learning_rate=1e-5, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
PASSES = ("forward", "forward_backward", "optim_step")
SETTLE_S = 0.2   # how long a test watches for a call that must not happen; not measured


class _Workers:
    """A model's worker group. `hold` keeps every worker call open until it is set."""

    def __init__(self, model_id: str, events: list):
        self.model_id, self.events, self.hold = model_id, events, threading.Event()
        self.hold.set()

    def _call(self):
        self.events.append(("workers called", self.model_id))
        self.hold.wait()
        self.events.append(("workers returned", self.model_id))

    def forward_backward(self, td):
        self._call()

    def compute_log_prob(self, td):
        self._call()

    def optimizer_step(self, params, clip):
        self._call()
        return {"grad_norm": 1.0}

    def release_cached_memory(self):
        pass


class _Sampler:
    """The sampler's two sides: the checkpoint manager that sleeps and wakes it, and its client."""

    def __init__(self, events: list):
        self.events, self.release = events, asyncio.Event()
        self.release.set()

    def sleep_replicas(self):
        self.events.append("sleep")

    def update_weights(self, global_steps: int):
        self.events.append(("wake", global_steps))

    async def generate(self, request_id, prompt_ids, sampling_params):
        self.events.append("sample started")
        await self.release.wait()
        self.events.append("sample returned")
        return SimpleNamespace(token_ids=[7, 8, 9], log_probs=[-0.1, -0.2, -0.3], stop_reason="stop")


@pytest.fixture
def events():
    return []


@pytest.fixture
def backend(monkeypatch):
    backend = VerlBackend()
    # the conversions around a worker call need torch tensors and verl's protocol objects
    monkeypatch.setattr(backend, "_ensure_loss_fn", lambda h, loss_fn: None)
    monkeypatch.setattr(backend, "_to_tensordict", lambda h, data, loss_fn, infer=False: None)
    monkeypatch.setattr(backend, "_converter", SimpleNamespace(extract_logprobs=lambda logprobs, data: []))
    monkeypatch.setattr(verl_backend, "_wg_call", lambda method, td: method(td))
    monkeypatch.setattr(verl_backend, "_wg_scalar_call", lambda method, *args: method(*args))
    monkeypatch.setattr(verl_backend, "padded_logprobs", lambda out, td: None)
    monkeypatch.setattr(verl_backend, "_scalar_metrics", lambda out: {})
    return backend


def _model(model_id: str, events: list) -> VerlHandle:
    return VerlHandle(model_id=model_id, backend_type="verl", trainer_config=None, worker_group=_Workers(model_id, events))


def _with_sampler(backend: VerlBackend, handle: VerlHandle, events: list, awake: bool) -> _Sampler:
    """Make `handle` the model that holds the server's sampler, built and awake or asleep."""
    sampler = _Sampler(events)
    handle.llm_manager, handle.llm_client, handle.ckpt_manager = object(), sampler, sampler
    handle.rollout_awake, handle.rollout_synced_version = awake, handle.weight_version
    backend._sampler_owner = handle
    return sampler


def _run_pass(backend: VerlBackend, handle: VerlHandle, operation: str):
    if operation == "forward":
        return backend.forward(handle, [], "cross_entropy")
    if operation == "forward_backward":
        return backend.forward_backward(handle, [], "cross_entropy")
    return backend.apply_optimizer_step(handle, ADAM)


def _sample(backend: VerlBackend, handle: VerlHandle):
    return backend.sample(handle, "s", [1, 2, 3], 1, {"max_tokens": 8, "temperature": 1.0})


@pytest.mark.parametrize("operation", PASSES)
@pytest.mark.parametrize("runs", ["another model", "the sampler's model"])
def test_a_pass_of_any_model_puts_the_awake_sampler_to_sleep_first(backend, events, operation, runs):
    """Only a pass of the model that holds the sampler slept it: another model's pass ran beside the
    awake sampler on the same GPUs and ran out of memory."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        _with_sampler(backend, a, events, awake=True)
        model = b if runs == "another model" else a
        await _run_pass(backend, model, operation)
        assert events == ["sleep", ("workers called", model.model_id), ("workers returned", model.model_id)]
        assert a.rollout_awake is False

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", PASSES)
@pytest.mark.parametrize("sampler", ["asleep", "none built"])
def test_a_pass_makes_no_sleep_call_without_an_awake_sampler(backend, events, operation, sampler):
    """The branch that must not fire: a sleep call on a sampler that is asleep, or on none."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        if sampler == "asleep":
            _with_sampler(backend, a, events, awake=False)
        await _run_pass(backend, b, operation)
        assert events == [("workers called", "b"), ("workers returned", "b")]

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", PASSES)
def test_a_pass_waits_for_the_in_flight_samples_of_the_sampler(backend, events, operation):
    """The pass calls its workers only after the sample has its tokens; the sample is not aborted."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        sampler = _with_sampler(backend, a, events, awake=True)
        sampler.release.clear()
        sample = asyncio.ensure_future(_sample(backend, a))
        await asyncio.sleep(SETTLE_S)
        run = asyncio.ensure_future(_run_pass(backend, b, operation))
        await asyncio.sleep(SETTLE_S)
        assert events == ["sample started"], "the pass slept the sampler or called its workers under a sample"
        sampler.release.set()
        sampled, _ = await asyncio.gather(sample, run)
        assert sampled["sequences"][0]["tokens"] == [7, 8, 9]
        assert events == ["sample started", "sample returned", "sleep", ("workers called", "b"), ("workers returned", "b")]

    asyncio.run(scenario())


def test_a_sample_during_another_models_pass_wakes_the_sampler_after_the_pass(backend, events):
    """The sample gets its tokens, and the sampler wakes at its model's current weight version."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        _with_sampler(backend, a, events, awake=True)
        a.weight_version = 3
        b.worker_group.hold.clear()
        run = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        sample = asyncio.ensure_future(_sample(backend, a))
        await asyncio.sleep(SETTLE_S)
        assert events == ["sleep", ("workers called", "b")], "the sample woke the sampler under the pass"
        b.worker_group.hold.set()
        sampled, _ = await asyncio.gather(sample, run)
        assert sampled["sequences"][0]["tokens"] == [7, 8, 9]
        assert events == ["sleep", ("workers called", "b"), ("workers returned", "b"),
                          ("wake", 3), "sample started", "sample returned"]

    asyncio.run(scenario())


def test_a_sample_under_two_passes_wakes_the_sampler_after_the_last(backend, events):
    """The first pass to return must not wake the sampler under the second: the passes are counted."""
    async def scenario():
        a, b, c = _model("a", events), _model("b", events), _model("c", events)
        _with_sampler(backend, a, events, awake=True)
        b.worker_group.hold.clear()
        c.worker_group.hold.clear()
        runs = [asyncio.ensure_future(_run_pass(backend, model, "forward")) for model in (b, c)]
        await asyncio.sleep(SETTLE_S)
        sample = asyncio.ensure_future(_sample(backend, a))
        b.worker_group.hold.set()
        await runs[0]
        await asyncio.sleep(SETTLE_S)
        assert not any(e[0] == "wake" for e in events if isinstance(e, tuple)), "the sampler woke while model c's pass ran"
        c.worker_group.hold.set()
        sampled, *_ = await asyncio.gather(sample, *runs)
        assert sampled["sequences"][0]["tokens"] == [7, 8, 9]
        assert events.index(("workers returned", "c")) < events.index(("wake", 0))
        assert events.count("sleep") == 1

    asyncio.run(scenario())
