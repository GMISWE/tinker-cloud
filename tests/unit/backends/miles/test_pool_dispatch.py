"""The miles pool dispatcher without a cluster: which queued operations one train call takes, and in what
order the rest run.

Every case queues its operations behind a held barrier, so the dispatcher sees them all at once. The train
call is a stand-in that records the requests it was given; the E0 guard is the converter's own."""
import asyncio
from argparse import Namespace

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.miles.backend import MilesBackend, MilesPool, _PoolOp

CE = "cross_entropy"


class Rig:
    """One pool and its dispatcher. `log` holds what ran, in order: a tuple of request tags for a train
    call, a string for any other operation."""

    def __init__(self, max_samples=0, reorder=False, e0_tokens=0):
        self.log = []
        self.failing = set()
        self.backend = MilesBackend()
        self.backend._execute_fb_batch = self._train
        self.pool = MilesPool(
            train_group=None, rollout_manager=None, placement_group=None, controller=None,
            args=Namespace(data_parallel_size=1), hf_path="", base_model="",
            cobatch_max_samples=max_samples, cobatch_reorder=reorder, cobatch_e0_tokens=e0_tokens,
        )
        self.pool.dispatcher = asyncio.create_task(self.backend._pool_dispatcher_loop(self.pool))

    async def _train(self, pool, batch):
        tags = tuple(op.rollout_data["tag"] for op in batch)
        self.log.append(tags)
        if self.failing & set(tags):
            raise RuntimeError("CUDA OOM")
        return list(tags)

    def fb(self, tenant, tag, rows=1, loss=CE, tokens=8):
        op = _PoolOp(kind="fb", tenant=tenant, loss_fn=loss, num_samples=rows,
                     rollout_data={"tag": tag, "tokens": [[0] * tokens for _ in range(rows)]})
        return self.backend._pool_submit(self.pool, op)

    def other(self, tenant, tag):
        """A non-train operation of `tenant` (an optimizer step, a save); tenant None is a barrier."""
        async def fn():
            self.log.append(tag)
            if tag in self.failing:
                raise RuntimeError("disk full")
            return tag
        return self.backend._pool_run(self.pool, fn, tenant=tenant)

    def stop(self):
        return self.backend._pool_submit(self.pool, _PoolOp(kind="stop"))

    async def run(self, *submissions):
        """Queue the submissions in order behind a held barrier, release it, and return each one's
        result or exception."""
        gate = asyncio.Event()
        hold = asyncio.ensure_future(self.backend._pool_run(self.pool, gate.wait))
        tasks = []
        for s in submissions:
            tasks.append(asyncio.ensure_future(s))
            await asyncio.sleep(0)
        gate.set()
        # bounded: an error outside the dispatcher's two try blocks ends the loop and resolves nothing
        await asyncio.wait_for(hold, 5)
        return await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)


def test_with_merging_off_every_request_is_its_own_train_call():
    async def main():
        r = Rig(max_samples=0)
        assert await r.run(r.fb("a", "a1"), r.fb("b", "b1")) == ["a1", "b1"]
        assert r.log == [("a1",), ("b1",)]
    asyncio.run(main())


def test_consecutive_requests_of_one_loss_merge_across_tenants_and_each_gets_its_own_result():
    async def main():
        r = Rig(max_samples=8)
        assert await r.run(r.fb("a", "a1"), r.fb("b", "b1"), r.fb("a", "a2")) == ["a1", "b1", "a2"]
        assert r.log == [("a1", "b1", "a2")]
    asyncio.run(main())


def test_a_request_of_another_loss_ends_the_call_and_heads_the_next():
    async def main():
        r = Rig(max_samples=8)
        await r.run(r.fb("a", "a1"), r.fb("b", "b1", loss="ppo"), r.fb("c", "c1"))
        assert r.log == [("a1",), ("b1",), ("c1",)]
    asyncio.run(main())


def test_a_request_over_the_sample_cap_ends_the_call_and_a_smaller_one_behind_it_does_not_overtake():
    async def main():
        r = Rig(max_samples=4)
        await r.run(r.fb("a", "a1", rows=2), r.fb("b", "b1", rows=3), r.fb("c", "c1", rows=1))
        assert r.log == [("a1",), ("b1", "c1")]
    asyncio.run(main())


def test_without_reorder_any_other_operation_ends_the_call():
    async def main():
        r = Rig(max_samples=8)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("b", "b1"), r.other("b", "b.step"))
        assert r.log == [("a1",), "a.step", ("b1",), "b.step"]
    asyncio.run(main())


def test_a_merge_the_e0_guard_refuses_ends_the_call_even_with_reorder_on():
    async def main():
        r = Rig(max_samples=8, reorder=True, e0_tokens=100)
        # 60 + 60 tokens cross the 100-token boundary; 60 + 10 do not, and c1 still waits for b1
        await r.run(r.fb("a", "a1", tokens=60), r.fb("b", "b1", tokens=60), r.fb("c", "c1", tokens=10))
        assert r.log == [("a1",), ("b1", "c1")]
    asyncio.run(main())


def test_reorder_defers_other_operations_past_the_call_and_replays_them_in_order():
    async def main():
        r = Rig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("b", "b1"), r.other("b", "b.step"))
        assert r.log == [("a1", "b1"), "a.step", "b.step"]
    asyncio.run(main())


def test_reorder_keeps_a_tenants_own_order_a_request_behind_its_deferred_operation_waits():
    async def main():
        r = Rig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.fb("a", "a2"), r.fb("b", "b1"))
        assert r.log == [("a1", "b1"), "a.step", ("a2",)]
    asyncio.run(main())


def test_reorder_never_defers_a_barrier():
    async def main():
        r = Rig(max_samples=8, reorder=True)
        await r.run(r.fb("a", "a1"), r.other(None, "barrier"), r.fb("b", "b1"))
        assert r.log == [("a1",), "barrier", ("b1",)]
    asyncio.run(main())


def test_a_failed_train_call_fails_every_request_in_it_and_the_queue_goes_on():
    async def main():
        r = Rig(max_samples=8)
        r.failing = {"b1"}
        a1, b1, step = await r.run(r.fb("a", "a1"), r.fb("b", "b1"), r.other("a", "a.step"))
        assert [repr(a1), repr(b1)] == ["RuntimeError('CUDA OOM')"] * 2
        assert step == "a.step"
        assert r.log == [("a1", "b1"), "a.step"]
    asyncio.run(main())


def test_another_operations_failure_reaches_its_submitter_alone():
    async def main():
        r = Rig(max_samples=8)
        r.failing = {"a.step"}
        step, b1 = await r.run(r.other("a", "a.step"), r.fb("b", "b1"))
        assert repr(step) == "RuntimeError('disk full')"
        assert b1 == "b1"
    asyncio.run(main())


def test_stop_runs_what_was_queued_ahead_of_it_deferred_operations_included_then_refuses_submissions():
    async def main():
        r = Rig(max_samples=8, reorder=True)
        results = await r.run(r.fb("a", "a1"), r.other("a", "a.step"), r.stop())
        assert results == ["a1", "a.step", None]
        assert r.log == [("a1",), "a.step"]
        await asyncio.wait_for(r.pool.dispatcher, 5)
        with pytest.raises(BackendError, match="pool dispatcher not running"):
            await r.fb("a", "a2")
    asyncio.run(main())
