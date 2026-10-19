"""GroupScheduler: FIFO per group, coalescing of consecutive same-key passes."""
import asyncio

import pytest

from tinkercloud.training.backends.automodel.scheduler import GroupScheduler, pass_key
from tinkercloud.training.backends.base import BackendError

FB = "forward_backward"


class Engine:
    """Records every run; a pass blocks until released so ops queue behind it."""

    def __init__(self):
        self.runs = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail_models = set()

    async def run_pass(self, key, payloads):
        self.runs.append((key.model_id, key.kind, key.loss_fn, tuple(payloads)))
        await self.gate.wait()
        if key.model_id in self.fail_models:
            raise RuntimeError("CUDA OOM")
        return [f"{key.model_id}:{p}" for p in payloads]

    def barrier(self, model_id, name):
        async def fn():
            self.runs.append((model_id, name))
            return name
        return fn


def key(model="a", kind=FB, loss="cross_entropy", cfg=None):
    return pass_key(model, kind, loss, cfg)


async def started(engine, n):
    while len(engine.runs) < n:
        await asyncio.sleep(0)


def run(coro):
    return asyncio.run(coro)


def test_passes_queued_behind_a_running_op_coalesce():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        first = asyncio.ensure_future(s.submit_pass(key(), 1))
        await started(e, 1)                      # pass 1 runs alone and blocks
        rest = [asyncio.ensure_future(s.submit_pass(key(), i)) for i in (2, 3, 4)]
        await asyncio.sleep(0)
        e.gate.set()
        assert await first == "a:1"
        assert await asyncio.gather(*rest) == ["a:2", "a:3", "a:4"]
        assert [r[3] for r in e.runs] == [(1,), (2, 3, 4)]
        await s.close("test done")
    run(main())


def test_run_breaks_at_model_kind_loss_and_config_and_keeps_fifo():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        blocker = asyncio.ensure_future(s.submit_pass(key("z"), 0))
        await started(e, 1)
        ops = [
            s.submit_pass(key("a"), 1), s.submit_pass(key("a"), 2),
            s.submit_pass(key("b"), 3),                                  # other tenant
            s.submit_pass(key("a"), 4),                                  # a again: no reordering
            s.submit_pass(key("a", kind="forward"), 5),                  # other kind
            s.submit_pass(key("a", loss="ppo"), 6),                      # other loss
            s.submit_pass(key("a", loss="ppo", cfg={"clip_low_threshold": 0.9}), 7),
        ]
        tasks = [asyncio.ensure_future(o) for o in ops]
        await asyncio.sleep(0)
        e.gate.set()
        await asyncio.gather(blocker, *tasks)
        assert [(r[0], r[1], r[2], r[3]) for r in e.runs[1:]] == [
            ("a", FB, "cross_entropy", (1, 2)),
            ("b", FB, "cross_entropy", (3,)),
            ("a", FB, "cross_entropy", (4,)),
            ("a", "forward", "cross_entropy", (5,)),
            ("a", FB, "ppo", (6,)),
            ("a", FB, "ppo", (7,)),
        ]
        await s.close("test done")
    run(main())


def test_barrier_runs_alone_and_splits_runs():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        blocker = asyncio.ensure_future(s.submit_pass(key(), 0))
        await started(e, 1)
        tasks = [asyncio.ensure_future(o) for o in (
            s.submit_pass(key(), 1), s.submit_barrier("a", e.barrier("a", "optim")),
            s.submit_pass(key(), 2), s.submit_pass(key(), 3))]
        await asyncio.sleep(0)
        e.gate.set()
        results = await asyncio.gather(blocker, *tasks)
        assert results[2] == "optim"
        assert [r[3] if len(r) == 4 else r[1] for r in e.runs] == [(0,), (1,), "optim", (2, 3)]
        await s.close("test done")
    run(main())


def test_failure_fails_the_whole_run_only():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        blocker = asyncio.ensure_future(s.submit_pass(key("z"), 0))
        await started(e, 1)
        a1, a2 = (asyncio.ensure_future(s.submit_pass(key(), i)) for i in (1, 2))
        b = asyncio.ensure_future(s.submit_pass(key("b"), 3))
        await asyncio.sleep(0)
        e.fail_models = {"a"}                    # the run a1+a2 fails; b, a separate run, succeeds
        e.gate.set()
        await blocker
        results = await asyncio.gather(a1, a2, b, return_exceptions=True)
        assert [type(r) for r in results[:2]] == [RuntimeError, RuntimeError]
        assert results[2] == "b:3"
        await s.close("test done")
    run(main())


def test_cancelled_submitter_is_skipped():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        blocker = asyncio.ensure_future(s.submit_pass(key(), 0))
        await started(e, 1)
        gone = asyncio.ensure_future(s.submit_pass(key(), 1))
        kept = asyncio.ensure_future(s.submit_pass(key(), 2))
        await asyncio.sleep(0)
        gone.cancel()
        e.gate.set()
        await blocker
        assert await kept == "a:2"
        assert e.runs[-1][3] == (2,)
        await s.close("test done")
    run(main())


def test_close_fails_queued_ops_and_refuses_new_ones():
    async def main():
        e = Engine()
        s = GroupScheduler("g", e.run_pass)
        e.gate.clear()
        running = asyncio.ensure_future(s.submit_pass(key(), 0))
        await started(e, 1)
        queued = asyncio.ensure_future(s.submit_pass(key("b"), 1))
        await asyncio.sleep(0)
        closing = asyncio.ensure_future(s.close("last tenant deleted"))
        await asyncio.sleep(0)
        with pytest.raises(BackendError, match="last tenant deleted"):
            await queued
        e.gate.set()
        assert await running == "a:0"            # the op in progress finishes
        await closing
        with pytest.raises(BackendError, match="is closed: last tenant deleted"):
            await s.submit_pass(key(), 2)
    run(main())


def test_wrong_result_count_fails_the_run_not_the_worker():
    async def main():
        async def bad(key, payloads):
            return payloads[:-1]
        s = GroupScheduler("g", bad)
        for i in (1, 2):                         # the worker survives the first failure
            with pytest.raises(BackendError, match="returned 0 results for 1 requests"):
                await asyncio.wait_for(s.submit_pass(key(), i), timeout=5)
        await s.close("test done")
    run(main())
