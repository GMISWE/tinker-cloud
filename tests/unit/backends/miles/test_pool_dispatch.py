"""The miles pool queue without a cluster: which queued operations one train call takes, and in what
order the rest run.

Every case queues its operations behind a held barrier, so the queue's worker sees them all at once."""
import asyncio

import pytest

from tests.fakes.miles_pool import PoolRig
from tinkercloud.training.backends.base import BackendError


def test_with_merging_off_every_request_is_its_own_train_call():
    async def main():
        r = PoolRig(max_samples=0)
        assert await r.run(r.fb("a", "a1"), r.fb("b", "b1")) == ["a1", "b1"]
        assert r.log == [("a1",), ("b1",)]
    asyncio.run(main())


def test_consecutive_requests_of_one_loss_merge_across_tenants_and_each_gets_its_own_result():
    async def main():
        r = PoolRig(max_samples=8)
        assert await r.run(r.fb("a", "a1"), r.fb("b", "b1"), r.fb("a", "a2")) == ["a1", "b1", "a2"]
        assert r.log == [("a1", "b1", "a2")]
    asyncio.run(main())


def test_a_request_of_another_loss_ends_the_call_and_heads_the_next():
    async def main():
        r = PoolRig(max_samples=8)
        await r.run(r.fb("a", "a1"), r.fb("b", "b1", loss="ppo"), r.fb("c", "c1"))
        assert r.log == [("a1",), ("b1",), ("c1",)]
    asyncio.run(main())


def test_a_request_over_the_sample_cap_ends_the_call_and_a_smaller_one_behind_it_does_not_overtake():
    async def main():
        r = PoolRig(max_samples=4)
        await r.run(r.fb("a", "a1", rows=2), r.fb("b", "b1", rows=3), r.fb("c", "c1", rows=1))
        assert r.log == [("a1",), ("b1", "c1")]
    asyncio.run(main())


def test_without_reorder_any_other_operation_ends_the_call():
    async def main():
        r = PoolRig(max_samples=8)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("b", "b1"), r.other("b", "b.step"))
        assert r.log == [("a1",), "a.step", ("b1",), "b.step"]
    asyncio.run(main())


def test_a_merge_the_e0_guard_refuses_ends_the_call_even_with_reorder_on():
    async def main():
        r = PoolRig(max_samples=8, reorder=True, e0_tokens=100)
        # 60 + 60 tokens cross the 100-token boundary; 60 + 10 do not, and c1 still waits for b1
        await r.run(r.fb("a", "a1", tokens=60), r.fb("b", "b1", tokens=60), r.fb("c", "c1", tokens=10))
        assert r.log == [("a1",), ("b1", "c1")]
    asyncio.run(main())


def test_reorder_defers_other_operations_past_the_call_and_replays_them_in_order():
    async def main():
        r = PoolRig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("b", "b1"), r.other("b", "b.step"))
        assert r.log == [("a1", "b1"), "a.step", "b.step"]
    asyncio.run(main())


def test_reorder_keeps_a_tenants_own_order_a_request_behind_its_deferred_operation_waits():
    async def main():
        r = PoolRig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("a", "a2"), r.fb("b", "b1"))
        assert r.log == [("a1", "b1"), "a.step", ("a2",)]
    asyncio.run(main())


def test_reorder_never_defers_a_barrier():
    async def main():
        r = PoolRig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other(None, "barrier"), r.fb("b", "b1"))
        assert r.log == [("a1",), "barrier", ("b1",)]
    asyncio.run(main())


def test_a_failed_train_call_fails_every_request_in_it_and_the_queue_goes_on():
    async def main():
        r = PoolRig(max_samples=8)
        r.failing = {"b1"}
        a1, b1, step = await r.run(r.fb("a", "a1"), r.fb("b", "b1"), r.other("a", "a.step"))
        assert [repr(a1), repr(b1)] == ["RuntimeError('CUDA OOM')"] * 2
        assert step == "a.step"
        assert r.log == [("a1", "b1"), "a.step"]
    asyncio.run(main())


def test_another_operations_failure_reaches_its_submitter_alone():
    async def main():
        r = PoolRig(max_samples=8)
        r.failing = {"a.step"}
        step, b1 = await r.run(r.other("a", "a.step"), r.fb("b", "b1"))
        assert repr(step) == "RuntimeError('disk full')"
        assert b1 == "b1"
    asyncio.run(main())


def test_stop_runs_what_was_queued_ahead_of_it_deferred_operations_included_then_refuses_submissions():
    async def main():
        r = PoolRig(max_samples=8, reorder=True)
        results = await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.stop())
        assert results == ["a1", "a.step", None]
        assert r.log == [("a1",), "a.step"]
        with pytest.raises(BackendError, match="is closed: pool stopped"):
            await r.fb("a", "a2")
    asyncio.run(main())
