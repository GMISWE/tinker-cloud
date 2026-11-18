"""NemoRLBackend's training operations on the model's scheduler: one at a time, in arrival order, and closed
with the model. A stand-in policy records each GPU call and can hold one; the converter and the step's GPU
helpers are replaced, so the order of operations is what the tests read, not the numbers."""
import asyncio
import gc
import threading
import weakref

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.nemo_rl import backend as nb
from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend, NemoRLHandle
from tinkercloud.training.models.requests import AdamParams
from tests.datums import ce_datum


class Batch:
    """A converted batch: `tags` names the requests it holds."""

    def __init__(self, tags):
        self.tags = list(tags)
        self.size = len(self.tags)

    def keys(self):
        return {"input_ids"}


class Converter:
    """Tags a request's batch with its datum count, so each request in a test has its own count."""

    def forward_to_backend(self, data, config):
        return Batch([f"fwd{len(data)}"])

    def forward_backward_to_backend(self, data, loss_fn, config, image_preprocessor=None):
        return Batch([f"fb{len(data)}"])

    def backend_to_forward_result(self, result, data, loss_fn):
        return result

    def backend_to_forward_backward_result(self, train_result, data, loss_fn):
        return {"metrics": {}, "grad_norm": 0.0}


class Policy:
    """Records each call as ("enter", name) and ("exit", name); `hold[name]` blocks that call until set."""

    def __init__(self):
        self.log = []
        self.hold = {}
        self.entered = {}
        self.shutdown_error = None

    def _call(self, name, result=None):
        self.log.append(("enter", name))
        self.entered.setdefault(name, threading.Event()).set()
        if name in self.hold:
            assert self.hold[name].wait(10), f"{name} was never released"
        self.log.append(("exit", name))
        return result

    def prepare_for_training(self):
        pass

    def get_logprobs(self, batch):
        return self._call("forward", batch.tags)

    def train(self, data, loss_fn, eval_mode, gbs):
        self._call(f"train{data.tags}")
        return {"loss": 0.0, "grad_norm": 0.0, "curr_logprobs": []}

    def save_checkpoint(self, weights_path, optimizer_path, **kw):
        self._call("save")

    def load(self):
        self._call("load")

    def shutdown(self):
        self._call("shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(nb, "_concatenate_batches", lambda bufs: Batch([t for b in bufs for t in b.tags]))
    monkeypatch.setattr(nb, "_maybe_pad_batch", lambda batch, *a: batch)
    monkeypatch.setattr(nb, "_set_learning_rate", lambda policy, lr: None)
    monkeypatch.setattr(nb, "_warn_on_adam_mismatch", lambda h, params: None)
    monkeypatch.setattr(nb, "_ensure_dyn_mb_budget", lambda h, data: None)
    monkeypatch.setattr(nb, "_step_loss_fn", lambda h, config: None)
    monkeypatch.setattr(nb, "_extract_loss_fn_outputs", lambda lp, data, n: [])
    monkeypatch.setattr(nb, "export_hf_adapter", lambda *a: None)
    monkeypatch.setattr(nb, "_stage_foreign_adapter", lambda path: path)
    monkeypatch.setattr(nb, "_policy_load_checkpoint", lambda policy, w, o: policy.load())
    backend = NemoRLBackend()
    backend._converter = Converter()
    policy = Policy()
    return backend, make_handle(policy), policy


def make_handle(policy):
    return NemoRLHandle(model_id="m", backend_type="nemo_rl", policy=policy, colocated_inference=False,
                        config={"policy": {"train_micro_batch_size": 1, "megatron_cfg": {"enabled": False},
                                           "dtensor_cfg": {"context_parallel_size": 1}},
                                "checkpointing": {}})


def data(n):
    return [ce_datum([1, 2, 3]) for _ in range(n)]


def step():
    return AdamParams(learning_rate=1e-4)


async def started(policy, name):
    """Wait, off the event loop, until the policy entered `name`."""
    event = policy.entered.setdefault(name, threading.Event())
    assert await asyncio.to_thread(event.wait, 10), f"{name} never started"


async def queued(backend, handle, n):
    """Wait until `n` operations wait in the model's scheduler behind the running one."""
    for _ in range(1000):
        if handle.scheduler.queued() == n:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"{handle.scheduler.queued()} queued, expected {n}")


def assert_closed(error):
    assert isinstance(error, BackendError) and "closed" in str(error), repr(error)


def run(rig, body):
    backend, handle, policy = rig

    async def main():
        backend._open_scheduler(handle)
        try:
            await body(backend, handle, policy)
        finally:
            for event in policy.hold.values():
                event.set()
            await backend.close()
    asyncio.run(main())


def test_operations_run_in_submission_order(rig):
    """Regression guarded: a training operation overtaking an earlier one on the same workers. A held forward
    lets the next four queue; each then runs only after the one before it returned."""
    async def body(backend, handle, policy):
        policy.hold["forward"] = threading.Event()
        tasks = [asyncio.create_task(backend.forward(handle, data(1), "cross_entropy"))]
        await started(policy, "forward")
        for i, op in enumerate([
            lambda: backend.forward_backward(handle, data(2), "cross_entropy"),
            lambda: backend.apply_optimizer_step(handle, step()),
            lambda: backend.save_checkpoint(handle, "/ckpt"),
            lambda: backend.load_checkpoint(handle, "/ckpt"),
        ], 1):
            tasks.append(asyncio.create_task(op()))
            await queued(backend, handle, i)
        policy.hold["forward"].set()
        await asyncio.gather(*tasks)
        names = [name for kind, name in policy.log if kind == "enter"]
        assert names == ["forward", "train['fb2']", "save", "load"]
        assert policy.log == [(k, n) for n in names for k in ("enter", "exit")]   # no call overlaps another

    run(rig, body)


def test_a_forward_backward_behind_a_running_step_buffers_for_the_next_step(rig):
    """Regression guarded: a forward_backward returning, or reaching the buffer, before the step running ahead
    of it has finished. Sent while the step's train call is held, it returns only after that step, and the
    next step trains it."""
    async def body(backend, handle, policy):
        await backend.forward_backward(handle, data(1), "cross_entropy")
        policy.hold["train['fb1']"] = threading.Event()
        first = asyncio.create_task(backend.apply_optimizer_step(handle, step()))
        await started(policy, "train['fb1']")
        later = asyncio.create_task(backend.forward_backward(handle, data(2), "cross_entropy"))
        await queued(backend, handle, 1)
        await asyncio.sleep(0.05)
        assert not later.done() and handle.data_buffer == []
        policy.hold["train['fb1']"].set()
        await first
        await later
        assert [b.tags for b in handle.data_buffer] == [["fb2"]]
        await backend.apply_optimizer_step(handle, step())
        assert [name for kind, name in policy.log if kind == "enter"] == ["train['fb1']", "train['fb2']"]

    run(rig, body)


def test_a_sampler_save_that_persists_nothing_does_not_wait_for_a_running_step(rig):
    """Regression guarded: an ephemeral sampler save queued behind a training step it does not need."""
    async def body(backend, handle, policy):
        await backend.forward_backward(handle, data(1), "cross_entropy")
        policy.hold["train['fb1']"] = threading.Event()
        running = asyncio.create_task(backend.apply_optimizer_step(handle, step()))
        await started(policy, "train['fb1']")
        await asyncio.wait_for(backend.save_checkpoint(handle, "/ckpt", persist=False), 1)
        assert not running.done()
        policy.hold["train['fb1']"].set()
        await running
        assert ("enter", "save") not in policy.log

    run(rig, body)


def test_an_operation_queued_behind_a_delete_fails_with_the_closed_error(rig):
    """Regression guarded: a step queued behind a delete running on the torn-down policy."""
    async def body(backend, handle, policy):
        await backend.forward_backward(handle, data(1), "cross_entropy")
        policy.hold["shutdown"] = threading.Event()
        delete = asyncio.create_task(backend.delete_model(handle))
        await started(policy, "shutdown")
        behind = asyncio.create_task(backend.apply_optimizer_step(handle, step()))
        await queued(backend, handle, 1)
        policy.hold["shutdown"].set()
        await delete
        with pytest.raises(BackendError) as e:
            await behind
        assert_closed(e.value)
        assert not any(name.startswith("train") for _, name in policy.log)

    run(rig, body)


def test_an_operation_after_a_delete_fails_with_the_closed_error(rig):
    """Regression guarded: a request sent after its model's delete reaching the torn-down policy."""
    async def body(backend, handle, policy):
        await backend.delete_model(handle)
        with pytest.raises(BackendError) as e:
            await backend.forward(handle, data(1), "cross_entropy")
        assert_closed(e.value)
        assert ("enter", "forward") not in policy.log

    run(rig, body)


def test_a_delete_that_fails_still_closes_the_model(rig):
    """Regression guarded: a delete that raised half-way leaving the model open to training."""
    async def body(backend, handle, policy):
        policy.shutdown_error = RuntimeError("actor died")
        with pytest.raises(BackendError, match="actor died"):
            await backend.delete_model(handle)
        with pytest.raises(BackendError) as e:
            await backend.forward_backward(handle, data(1), "cross_entropy")
        assert_closed(e.value)
        assert handle.data_buffer == []

    run(rig, body)


def test_close_closes_every_model(rig):
    """Regression guarded: a server shutdown leaving a model's scheduler open to new requests."""
    async def body(backend, handle, policy):
        await backend.close()
        with pytest.raises(BackendError) as e:
            await backend.forward_backward(handle, data(1), "cross_entropy")
        assert_closed(e.value)
        assert handle.data_buffer == []

    run(rig, body)


def test_a_retried_delete_after_a_failure_tears_down(rig):
    """Regression guarded: a delete that failed half-way can never be retried, since its scheduler is closed,
    and the workers it had not shut down stay up. The retry reaches the policy's shutdown again."""
    async def body(backend, handle, policy):
        policy.shutdown_error = RuntimeError("actor died")
        with pytest.raises(BackendError, match="actor died"):
            await backend.delete_model(handle)
        policy.shutdown_error = None
        await backend.delete_model(handle)
        assert [name for kind, name in policy.log if kind == "enter"] == ["shutdown", "shutdown"]
        with pytest.raises(BackendError) as e:
            await backend.forward_backward(handle, data(1), "cross_entropy")
        assert_closed(e.value)

    run(rig, body)


def test_a_deleted_model_is_not_held_by_its_scheduler(rig):
    """Regression guarded: every deleted model's policy, clusters and tokenizer kept alive by its closed
    scheduler for the life of the server."""
    backend, _, policy = rig
    handle = make_handle(policy)   # the fixture's handle stays referenced by pytest

    async def main(h):
        backend._open_scheduler(h)
        await backend.delete_model(h)
    asyncio.run(main(handle))
    gone = weakref.ref(handle)
    del handle
    gc.collect()
    assert gone() is None


def test_a_request_on_a_deleted_handle_never_reaches_a_model_recreated_under_its_id(rig):
    """Regression guarded: a request still holding a deleted model's handle queued on, and run by, a new
    model created under the same id (ids repeat across a resend)."""
    async def body(backend, handle, policy):
        await backend.delete_model(handle)
        again = make_handle(Policy())
        backend._open_scheduler(again)
        with pytest.raises(BackendError) as e:
            await backend.forward_backward(handle, data(1), "cross_entropy")
        assert_closed(e.value)
        await backend.forward_backward(again, data(2), "cross_entropy")
        assert [b.tags for b in again.data_buffer] == [["fb2"]]

    run(rig, body)
