"""GroupScheduler: a worker that dies closes the group, and no submitter is left waiting."""
import asyncio

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.common.scheduler import GroupScheduler, same_key
from tests.unit.backends.common.test_scheduler import Engine, key, run, started

CLOSED = "is closed: scheduler cancelled"


def test_a_worker_that_dies_fails_the_queue_with_the_same_message():
    """Regression guarded: a queued op failed with the bare reason, a later submission with the closed error."""
    async def main():
        e = Engine()
        s = GroupScheduler("g", same_key, e.run_pass, "test")
        e.gate.clear()
        running = asyncio.ensure_future(s.submit_pass(key("a"), 0))
        await started(e, 1)
        queued = asyncio.ensure_future(s.submit_pass(key("b"), 1))
        await asyncio.sleep(0)
        s._worker.cancel()
        with pytest.raises(BackendError, match=CLOSED) as waited:
            await asyncio.wait_for(queued, timeout=5)
        with pytest.raises(BackendError, match=CLOSED) as later:
            await s.submit_pass(key("a"), 2)
        assert str(waited.value) == str(later.value)
        with pytest.raises(BackendError):
            await running
    run(main())


def test_a_worker_that_dies_fails_the_operation_in_progress():
    """Regression guarded: the worker's exit failed only the queue, so the future of the pass run or barrier
    it was awaiting was never set and its submitter waited forever."""
    async def main():
        e = Engine()

        async def held():
            e.runs.append(("a", "held"))
            await e.gate.wait()

        submits = {"pass": lambda s: s.submit_pass(key("a"), 0), "barrier": lambda s: s.submit_barrier(held)}
        for kind, submit in submits.items():
            e.runs.clear()
            e.gate.clear()
            s = GroupScheduler(f"g-{kind}", same_key, e.run_pass, "test")
            running = asyncio.ensure_future(submit(s))
            await started(e, 1)
            s._worker.cancel()
            with pytest.raises(BackendError, match=CLOSED):
                await asyncio.wait_for(running, timeout=5)
    run(main())
