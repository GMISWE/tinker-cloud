"""The miles pool queue where it takes the shared scheduler's rule: what a stop does to an operation
queued behind it, and what happens to a request whose submitter gave up."""
import asyncio

import pytest

from tests.fakes.miles_pool import PoolRig
from tinkercloud.training.backends.base import BackendError


def test_an_operation_queued_behind_a_stop_fails_with_pool_stopped():
    """A stop closes the queue: the request behind it gets an error and reaches no train call."""
    async def main():
        r = PoolRig()
        release, (stop, late) = await r.queue(r.stop(), r.fb("a", "a1"))
        await release()
        assert await asyncio.wait_for(stop, 5) is None
        with pytest.raises(BackendError, match="pool stopped"):
            await asyncio.wait_for(late, 5)
        assert r.log == []
    asyncio.run(main())


def test_a_cancelled_request_is_in_no_train_call():
    """A request cancelled while it waits is skipped; the requests around it merge into one call."""
    async def main():
        r = PoolRig(max_samples=8)
        release, (a1, b1, c1) = await r.queue(r.fb("a", "a1"), r.fb("b", "b1"), r.fb("c", "c1"))
        b1.cancel()
        await asyncio.sleep(0)
        await release()
        assert await asyncio.wait_for(asyncio.gather(a1, c1), 5) == ["a1", "c1"]
        assert b1.cancelled()
        assert r.log == [("a1", "c1")]
    asyncio.run(main())
