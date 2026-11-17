"""The verl backend's one scheduler: every operation of every model runs in arrival order, one at a
time, beside the sampler's generates; a deleted model is refused by name, and close() fails what is
still queued.

The workers and the sampler are the stand-ins of `test_sampler_sleep.py`, with the GPU shares and the
recorded kills of `test_pass_lock.py`; here the workers also record saves and loads, and the sampler's
wake can be held."""
import asyncio
import threading
from pathlib import Path

import pytest

pytest.importorskip("verl", reason="the backend module imports verl's types")

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.verl.backend import VerlBackend, VerlHandle, worker_name_prefix

from . import test_pass_lock, test_sampler_sleep
from .test_sampler_sleep import ADAM, SETTLE_S, _model, _run_pass, _sample, _Sampler, _with_sampler, _Workers

# the fixtures of those suites: the list the stand-ins record in, and the backend around them
events = test_sampler_sleep.events
sleep_backend = test_sampler_sleep.backend
backend = test_pass_lock.backend

STOPPED = "model a is stopped: it was deleted"
WAIT_S = 5   # a bound on a wait that must end; not measured


class _CheckpointWorkers(_Workers):
    def save_checkpoint(self, local_path, hdfs_path, step):
        self.events.append(("save", self.model_id))

    def load_checkpoint(self, local_path):
        self.events.append(("load", self.model_id))


class _HeldWakeSampler(_Sampler):
    """A sampler whose wake stays open until `wake_gate` is set."""

    def __init__(self, events: list):
        super().__init__(events)
        self.wake_gate = threading.Event()

    def update_weights(self, global_steps: int):
        self.events.append("wake started")
        self.wake_gate.wait()
        super().update_weights(global_steps)


def _checkpoint_model(model_id: str, events: list) -> VerlHandle:
    handle = _model(model_id, events)
    handle.worker_group = _CheckpointWorkers(model_id, events)
    return handle


async def _until(condition) -> None:
    async def poll():
        while not condition():
            await asyncio.sleep(0)
    await asyncio.wait_for(poll(), timeout=WAIT_S)


def test_a_save_load_and_delete_wait_for_another_models_pass(backend, events, tmp_path):
    """A model's save, load and delete ran beside another model's pass on the same GPUs."""
    async def scenario():
        a, b = _checkpoint_model("a", events), _model("b", events)
        b.worker_group.hold.clear()
        run_b = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        later = [asyncio.ensure_future(backend.save_checkpoint(a, tmp_path, step=1)),
                 asyncio.ensure_future(backend.load_checkpoint(a, tmp_path)),
                 asyncio.ensure_future(backend.delete_model(a))]
        await asyncio.sleep(SETTLE_S)
        assert events == [("workers called", "b")], "model a's operation ran inside model b's pass"
        b.worker_group.hold.set()
        await asyncio.gather(run_b, *later)
        assert events == [("workers called", "b"), ("workers returned", "b"),
                          ("save", "a"), ("load", "a"), ("killed", worker_name_prefix("a"))]

    asyncio.run(scenario())


def test_a_sample_between_two_passes_wakes_between_them(backend, events):
    """A sample waits for the passes ahead of it only; a pass behind it waits for its generate."""
    async def scenario():
        a, b, c = _model("a", events), _model("b", events), _model("c", events)
        _with_sampler(backend, a, events, awake=False)
        b.worker_group.hold.clear()
        run_b = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        sample = asyncio.ensure_future(_sample(backend, a))
        await asyncio.sleep(SETTLE_S)
        run_c = asyncio.ensure_future(_run_pass(backend, c, "forward"))
        await asyncio.sleep(SETTLE_S)
        b.worker_group.hold.set()
        sampled, _, _ = await asyncio.gather(sample, run_b, run_c)
        assert sampled["sequences"][0]["tokens"] == [7, 8, 9]
        assert events == [("workers called", "b"), ("workers returned", "b"), ("wake", 0),
                          "sample started", "sample returned", "sleep",
                          ("workers called", "c"), ("workers returned", "c")]

    asyncio.run(scenario())


def test_a_sample_cancelled_during_its_wake_leaves_no_sample_counted(backend, events):
    """A sample cancelled while its wake ran stayed counted in flight, and every later pass waited for it."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        sampler = _HeldWakeSampler(events)
        _with_sampler(backend, a, events, awake=False)
        a.llm_client = a.ckpt_manager = sampler
        sample = asyncio.ensure_future(_sample(backend, a))
        await _until(lambda: "wake started" in events)
        sample.cancel()
        await asyncio.sleep(SETTLE_S)
        sampler.wake_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await sample
        await asyncio.wait_for(_run_pass(backend, b, "forward"), timeout=WAIT_S)
        assert a._active_samplers == 0
        assert events[-2:] == [("workers called", "b"), ("workers returned", "b")]

    asyncio.run(scenario())


def test_a_sample_cancelled_while_queued_wakes_nothing(backend, events):
    """A sample cancelled while it waited behind another model's pass still woke and synced the
    sampler for no one, and its caller heard of the cancel only after that pass."""
    async def scenario():
        a, b, c = _model("a", events), _model("b", events), _model("c", events)
        _with_sampler(backend, a, events, awake=False)
        b.worker_group.hold.clear()
        run_b = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        sample = asyncio.ensure_future(_sample(backend, a))
        await asyncio.sleep(SETTLE_S)
        sample.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(sample, timeout=WAIT_S)
        assert events == [("workers called", "b")], "the cancel waited for the pass ahead of it"
        b.worker_group.hold.set()
        await run_b
        await asyncio.wait_for(_run_pass(backend, c, "forward"), timeout=WAIT_S)
        assert a._active_samplers == 0
        assert events == [("workers called", "b"), ("workers returned", "b"),
                          ("workers called", "c"), ("workers returned", "c")]

    asyncio.run(scenario())


def _requests(backend: VerlBackend, handle: VerlHandle, root: Path) -> dict:
    return {
        "forward": lambda: _run_pass(backend, handle, "forward"),
        "forward_backward": lambda: _run_pass(backend, handle, "forward_backward"),
        "optim_step": lambda: backend.apply_optimizer_step(handle, ADAM),
        "sample": lambda: _sample(backend, handle),
        "save_checkpoint": lambda: backend.save_checkpoint(handle, root, step=1),
        "load_checkpoint": lambda: backend.load_checkpoint(handle, root),
        "prepare_for_generation": lambda: backend.prepare_for_generation(handle),
        "delete_model": lambda: backend.delete_model(handle),
    }


def test_an_operation_queued_behind_a_delete_is_refused_by_model(backend, events):
    """A pass and a sample queued behind their model's delete ran against the torn-down workers."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        b.worker_group.hold.clear()
        run_b = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        delete = asyncio.ensure_future(backend.delete_model(a))
        queued = [asyncio.ensure_future(_run_pass(backend, a, "forward_backward")),
                  asyncio.ensure_future(_sample(backend, a))]
        await asyncio.sleep(SETTLE_S)
        b.worker_group.hold.set()
        await asyncio.gather(run_b, delete)
        for request in queued:
            with pytest.raises(BackendError, match=STOPPED):
                await request
        assert events == [("workers called", "b"), ("workers returned", "b"), ("killed", worker_name_prefix("a"))]

    asyncio.run(scenario())


def test_an_operation_after_a_delete_is_refused_by_model(backend, events, tmp_path):
    """Every request to a deleted model is refused by name, and none reaches its workers or the sampler."""
    async def scenario():
        a = _model("a", events)
        await backend.delete_model(a)
        seen = list(events)
        for name, request in _requests(backend, a, tmp_path).items():
            with pytest.raises(BackendError, match=STOPPED):
                await request()
            assert events == seen, f"{name} reached the workers or the sampler of a deleted model"
        # a sampler's save writes nothing and waits for nothing: it answers for a deleted model too
        assert await backend.save_checkpoint(a, tmp_path, persist=False) is None
        assert events == seen

    asyncio.run(scenario())


def test_close_fails_the_queued_operations(backend, events):
    """At shutdown, an operation still queued is failed at once; the operation in progress finishes."""
    async def scenario():
        a, b = _model("a", events), _model("b", events)
        b.worker_group.hold.clear()
        run_b = asyncio.ensure_future(_run_pass(backend, b, "forward"))
        await asyncio.sleep(SETTLE_S)
        queued = asyncio.ensure_future(_run_pass(backend, a, "forward"))
        await asyncio.sleep(SETTLE_S)
        closing = asyncio.ensure_future(backend.close())
        with pytest.raises(BackendError, match="is closed: server shutdown"):
            await asyncio.wait_for(queued, timeout=WAIT_S)
        b.worker_group.hold.set()
        await asyncio.wait_for(asyncio.gather(run_b, closing), timeout=WAIT_S)
        assert events == [("workers called", "b"), ("workers returned", "b")]

    asyncio.run(scenario())
