"""The verl backend's passes on shared GPUs: the server runs one model's pass at a time, in the order
the passes arrive, and a pass whose workers fail ends its model while the other models go on.

The workers are the stand-ins of `test_sampler_sleep.py`; here one can also fail its calls, and the
kills of a model's actors are recorded in the same list as the worker calls."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl", reason="the backend module imports verl's types")

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.verl import backend as verl_backend
from tinkercloud.training.backends.verl.backend import (
    VLLM_SERVER_NAME_PREFIX, VerlBackend, VerlHandle, _SharedGpus, worker_name_prefix,
)

from . import test_sampler_sleep
from .test_sampler_sleep import ADAM, PASSES, SETTLE_S, _model, _run_pass, _sample, _with_sampler, _Workers

# that suite's fixtures: the list the stand-ins record in, and the backend around them
events = test_sampler_sleep.events
sleep_backend = test_sampler_sleep.backend

WORKER_ERROR = "CUDA out of memory on rank 0"
# the backend's own name of each pass, as its errors carry it
OPERATION = {"forward": "get_logprobs", "forward_backward": "forward_backward", "optim_step": "apply_optimizer_step"}
NUM_GPUS = 4


class _FailingWorkers(_Workers):
    def _call(self):
        self.events.append(("workers called", self.model_id))
        raise RuntimeError(WORKER_ERROR)


@pytest.fixture
def backend(sleep_backend, events, monkeypatch):
    """The stand-in backend, with two models' shares of the GPUs taken and every kill recorded."""
    monkeypatch.setattr(verl_backend, "_kill_named_actors", lambda prefix: events.append(("killed", prefix)))
    gpus = _SharedGpus(models_per_gpu=2)
    gpus.pool, gpus.model_ids = SimpleNamespace(pgs=[], store=[NUM_GPUS]), {"a", "b"}
    monkeypatch.setattr(sleep_backend, "_gpus", gpus)
    return sleep_backend


def _failing_model(events: list) -> VerlHandle:
    handle = _model("a", events)
    handle.worker_group = _FailingWorkers("a", events)
    return handle


def _calls(events: list) -> list:
    return [e for e in events if e[0] in ("workers called", "workers returned")]


@pytest.mark.parametrize("first", PASSES)
@pytest.mark.parametrize("second", PASSES)
def test_a_pass_of_another_model_waits_for_the_running_pass(backend, events, first, second):
    """Passes of two models ran at the same time on the same GPUs and ran out of memory."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        a.worker_group.hold.clear()
        run_a = asyncio.ensure_future(_run_pass(backend, a, first))
        await asyncio.sleep(SETTLE_S)
        run_b = asyncio.ensure_future(_run_pass(backend, b, second))
        await asyncio.sleep(SETTLE_S)
        assert events == [("workers called", "a")], "model b's workers were called inside model a's pass"
        a.worker_group.hold.set()
        await asyncio.gather(run_a, run_b)
        assert events == [("workers called", "a"), ("workers returned", "a"),
                          ("workers called", "b"), ("workers returned", "b")]

    asyncio.run(scenario())


def test_three_passes_submitted_together_run_one_at_a_time_in_order(backend, events):
    """Three models' forwards submitted together overlapped and one ran out of memory."""
    async def scenario():
        models = [_model(name, events) for name in ("a", "b", "c")]
        for model in models:
            model.worker_group.hold.clear()
        runs = [asyncio.ensure_future(_run_pass(backend, model, operation))
                for model, operation in zip(models, PASSES, strict=True)]
        for served, model in enumerate(models):
            await asyncio.sleep(SETTLE_S)
            done = [e for name in "abc"[:served] for e in (("workers called", name), ("workers returned", name))]
            assert events == [*done, ("workers called", model.model_id)], "two worker calls were open at once"
            model.worker_group.hold.set()
            await runs[served]
        assert [e[1] for e in events if e[0] == "workers called"] == ["a", "b", "c"]

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", PASSES)
@pytest.mark.parametrize("holds_sampler", [False, True])
def test_a_worker_failure_stops_the_model_and_the_next_models_pass_runs(backend, events, operation, holds_sampler):
    """After one rank of a pass ran out of memory the model's next pass never returned, and its
    delete waited behind it. The model's actors go before the next pass gets the GPUs."""
    async def scenario():
        a, b = _failing_model(events), _model("b", events)
        if holds_sampler:
            _with_sampler(backend, a, events, awake=False)
        failed = asyncio.ensure_future(_run_pass(backend, a, operation))
        await asyncio.sleep(0)   # model a's pass takes its turn first
        after = asyncio.ensure_future(_run_pass(backend, b, operation))
        with pytest.raises(BackendError, match=WORKER_ERROR) as raised:
            await failed
        assert raised.value.operation == OPERATION[operation]
        await after
        sampler_kill = [("killed", VLLM_SERVER_NAME_PREFIX)] if holds_sampler else []
        assert events == [("workers called", "a"), *sampler_kill, ("killed", worker_name_prefix("a")),
                          ("workers called", "b"), ("workers returned", "b")]
        assert a.worker_group is None and a.llm_client is None
        assert backend._sampler_owner is None

    asyncio.run(scenario())


def test_a_stop_that_fails_keeps_the_workers_error_and_the_model_stopped(backend, events, monkeypatch):
    """A kill that raised replaced the workers' error in the request's answer: the client saw
    "kill failed" and not what the pass died of."""
    async def scenario():
        a, b = _failing_model(events), _model("b", events)

        def kill(prefix):
            raise RuntimeError("kill failed")

        with monkeypatch.context() as patched:
            patched.setattr(verl_backend, "_kill_named_actors", kill)
            with pytest.raises(BackendError, match=f"{WORKER_ERROR}; then stopping model a's actors failed: kill failed"):
                await _run_pass(backend, a, "forward")
        with pytest.raises(BackendError, match="model a is stopped"):
            await _run_pass(backend, a, "forward")
        await _run_pass(backend, b, "forward")
        await backend.delete_model(a)   # repeats the kills
        assert events == [("workers called", "a"), ("workers called", "b"), ("workers returned", "b"),
                          ("killed", worker_name_prefix("a"))]

    asyncio.run(scenario())


def test_the_kill_passes_over_an_actor_that_died_after_the_listing(monkeypatch):
    """A rank that is dying of the failure can be gone between the listing and the lookup (read in the
    code, not seen on a server); the lookup's error would end the kill loop with the other ranks alive."""
    import ray

    killed = []

    def get_actor(name):
        if name == "verl_a_1":
            raise ValueError(f"Failed to look up actor with name '{name}'")
        return name

    monkeypatch.setattr(ray.util, "list_named_actors", lambda: ["verl_a_0", "verl_a_1", "verl_a_2", "verl_b_0"])
    monkeypatch.setattr(ray, "get_actor", get_actor)
    monkeypatch.setattr(ray, "kill", killed.append)
    verl_backend._kill_named_actors(worker_name_prefix("a"))
    assert killed == ["verl_a_0", "verl_a_2"]


def _later_requests(backend: VerlBackend, handle: VerlHandle) -> dict:
    return {
        "forward": lambda: _run_pass(backend, handle, "forward"),
        "forward_backward": lambda: _run_pass(backend, handle, "forward_backward"),
        "optim_step": lambda: backend.apply_optimizer_step(handle, ADAM),
        "sample": lambda: _sample(backend, handle),
        "save_checkpoint": lambda: backend.save_checkpoint(handle, Path("/nonexistent")),
        "save for a sampler": lambda: backend.save_checkpoint(handle, Path("/nonexistent"), persist=False),
        "prepare_for_generation": lambda: backend.prepare_for_generation(handle),
        "load_checkpoint": lambda: backend.load_checkpoint(handle, Path("/nonexistent")),
    }


@pytest.mark.parametrize("first_failure", PASSES)
def test_a_stopped_model_refuses_every_later_request_by_name(backend, events, first_failure):
    """A request to a model whose ranks one failure left out of step never returned. A stopped model
    answers at once, and its error says which model and which pass."""
    async def scenario():
        a = _failing_model(events)
        with pytest.raises(BackendError, match=WORKER_ERROR):
            await _run_pass(backend, a, first_failure)
        seen = list(events)
        for name, request in _later_requests(backend, a).items():
            with pytest.raises(BackendError, match=f"model a is stopped.*{OPERATION[first_failure]}: {WORKER_ERROR}"):
                await request()
            assert events == seen, f"{name} reached the workers or the sampler of a stopped model"

    asyncio.run(scenario())


def test_a_request_that_waited_behind_the_failing_pass_is_refused(backend, events):
    """A training loop's shape: an optimizer step submitted right behind its forward_backward. It must not
    reach workers that are gone."""
    async def scenario():
        a = _failing_model(events)
        failed = asyncio.ensure_future(_run_pass(backend, a, "forward_backward"))
        await asyncio.sleep(0)
        waiting = asyncio.ensure_future(backend.apply_optimizer_step(a, ADAM))
        with pytest.raises(BackendError, match=WORKER_ERROR):
            await failed
        with pytest.raises(BackendError, match="model a is stopped"):
            await waiting
        assert _calls(events) == [("workers called", "a")]

    asyncio.run(scenario())


def test_delete_removes_a_stopped_model_and_gives_its_share_back(backend, events):
    """The delete of a model whose pass had failed waited behind that model's hung pass and never
    returned, so its share of the GPUs was lost until the server restarted."""
    async def scenario():
        a = _failing_model(events)
        with pytest.raises(BackendError, match=WORKER_ERROR):
            await _run_pass(backend, a, "forward")
        with pytest.raises(BackendError, match="the GPUs hold 2 models"):
            backend._gpus.acquire("a_again", NUM_GPUS)
        await backend.delete_model(a)
        assert backend._gpus.model_ids == {"b"}
        backend._gpus.acquire("a_again", NUM_GPUS)
        assert backend._gpus.model_ids == {"a_again", "b"}

    asyncio.run(scenario())


def test_an_error_before_the_worker_call_leaves_the_model_usable(backend, events, monkeypatch):
    """Only a worker's failure leaves the ranks in an unknown state: refused datums end no model."""
    async def scenario():
        a = _model("a", events)

        def refuse(h, data, loss_fn, infer=False):
            raise ValueError("weights and target_tokens differ in shape")

        with monkeypatch.context() as patched:
            patched.setattr(backend, "_to_tensordict", refuse)
            with pytest.raises(BackendError, match="differ in shape"):
                await _run_pass(backend, a, "forward_backward")
        assert events == [] and a.failed is None and a.worker_group is not None
        await _run_pass(backend, a, "forward_backward")
        assert events == [("workers called", "a"), ("workers returned", "a")]

    asyncio.run(scenario())
