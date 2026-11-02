"""Release under cancellation: a cancelled release keeps the
tenant and the group for the next delete; GroupScheduler.close() after a
cancelled worker returns instead of raising, and still waits for an op in
progress."""
import asyncio

import pytest

from tinkercloud.training.backends.automodel.scheduler import GroupScheduler, pass_key
from tinkercloud.training.backends.base import BackendError
from tests.unit.backends.automodel.test_backend import ADAM, wired

FB = "forward_backward"
run = asyncio.run


async def settle(n=5):
    for _ in range(n):
        await asyncio.sleep(0)


def test_cancelled_release_keeps_the_tenant_for_the_next_delete():
    async def go():
        backend, ranks, m = wired()
        ranks.hold_step = asyncio.Event()                       # the step never returns: the group hangs
        step = asyncio.ensure_future(backend.apply_optimizer_step(m, 0.5, ADAM))
        for _ in range(50):
            if ("optim_step", "m") in ranks.ops:
                break
            await asyncio.sleep(0)
        assert ("optim_step", "m") in ranks.ops

        delete = asyncio.ensure_future(backend.delete_model(m))  # waits in close() for the hung step
        await settle(20)
        assert not delete.done() and not ranks.stopped
        delete.cancel()
        with pytest.raises(asyncio.CancelledError):
            await delete
        group = backend._groups["g"]
        assert "m" in group.tenants and group.lost is None, "the interrupted release left the tenant in its group"
        assert not ranks.stopped, "the ranks were not stopped by the interrupted release"

        await backend.delete_model(m)                          # the next delete frees for real
        assert ranks.stopped and "g" not in backend._groups
        step.cancel()
    run(go())


def test_close_after_a_cancelled_worker_returns():
    async def go():
        gate = asyncio.Event()
        ran = []

        async def run_pass(key, payloads):
            ran.append(payloads)
            await gate.wait()
            return list(payloads)

        s = GroupScheduler("g", run_pass)
        pending = asyncio.ensure_future(s.submit_pass(pass_key("a", FB, "cross_entropy", None), 1))
        for _ in range(50):
            if ran:
                break
            await asyncio.sleep(0)
        closer = asyncio.ensure_future(s.close("first"))     # waits for the op in progress
        await settle()
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        await settle()
        assert s._worker.cancelled(), "cancelling close() cancelled the worker it awaited"

        await asyncio.wait_for(s.close("second"), timeout=1)   # returns, does not re-raise the worker's cancel
        await asyncio.wait_for(s.close("third"), timeout=1)    # and again: a no-op
        with pytest.raises(BackendError, match="closed"):
            await s.submit_pass(pass_key("a", FB, "cross_entropy", None), 2)
        pending.cancel()
    run(go())


def test_close_waits_for_the_op_in_progress():
    async def go():
        gate = asyncio.Event()
        ran = []

        async def run_pass(key, payloads):
            ran.append(payloads)
            await gate.wait()
            return list(payloads)

        s = GroupScheduler("g", run_pass)
        first = asyncio.ensure_future(s.submit_pass(pass_key("a", FB, "cross_entropy", None), 1))
        for _ in range(50):
            if ran:
                break
            await asyncio.sleep(0)
        closer = asyncio.ensure_future(s.close("deleted"))
        await settle(20)
        assert not closer.done(), "close() waits while the pass runs"
        gate.set()
        await asyncio.wait_for(closer, timeout=1)
        assert await first == 1, "the op in progress completed and delivered its result"
        assert s._worker.done() and not s._worker.cancelled()
    run(go())
