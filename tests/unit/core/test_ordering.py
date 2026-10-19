"""services.ordering: passes overlap, barriers wait, program order is kept per model."""
import asyncio

import pytest

from tinkercloud.training.services.ordering import BARRIER, PASS, ModelQueues


def _op(log, name, gate=None, delay=0.0):
    async def run():
        log.append(f"start {name}")
        if gate is not None:
            await gate.wait()
        if delay:
            await asyncio.sleep(delay)
        log.append(f"end {name}")
        return name
    return run


def test_barrier_waits_for_earlier_passes_to_complete():
    async def main():
        q = ModelQueues(); log = []; gate = asyncio.Event()
        t1 = asyncio.create_task(q.run("m", PASS, _op(log, "fb1", gate)))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(q.run("m", BARRIER, _op(log, "optim")))
        await asyncio.sleep(0.01)
        assert log == ["start fb1"]  # optim has not started: fb1 still in flight
        gate.set()
        await asyncio.gather(t1, t2)
        assert log == ["start fb1", "end fb1", "start optim", "end optim"]
    asyncio.run(main())


def test_pass_after_barrier_starts_only_after_barrier_started():
    async def main():
        q = ModelQueues(); log = []; gate_fb1 = asyncio.Event(); gate_optim = asyncio.Event()
        t1 = asyncio.create_task(q.run("m", PASS, _op(log, "fb1", gate_fb1)))
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(q.run("m", BARRIER, _op(log, "optim", gate_optim)))
        await asyncio.sleep(0.01)
        t3 = asyncio.create_task(q.run("m", PASS, _op(log, "fb2")))
        await asyncio.sleep(0.01)
        assert log == ["start fb1"]  # fb2 held behind the not-yet-started barrier
        gate_fb1.set(); await asyncio.sleep(0.01)
        # optim started; fb2 may now run and overlap with it
        assert log[:3] == ["start fb1", "end fb1", "start optim"] and "end optim" not in log
        assert "start fb2" in log
        gate_optim.set()
        await asyncio.gather(t1, t2, t3)
        assert log.index("start fb2") > log.index("start optim")
    asyncio.run(main())


def test_passes_overlap_and_models_are_independent():
    async def main():
        q = ModelQueues(); log = []; gate = asyncio.Event()
        a = asyncio.create_task(q.run("m", PASS, _op(log, "a", gate)))
        b = asyncio.create_task(q.run("m", PASS, _op(log, "b", gate)))
        other = asyncio.create_task(q.run("other", BARRIER, _op(log, "other-optim")))
        await asyncio.sleep(0.01)
        assert {"start a", "start b", "start other-optim", "end other-optim"} <= set(log)
        gate.set()
        await asyncio.gather(a, b, other)
        assert q.inflight_count("m") == 0
    asyncio.run(main())


def test_failed_op_does_not_block_the_queue():
    async def main():
        q = ModelQueues(); log = []
        async def boom():
            raise RuntimeError("x")
        with pytest.raises(RuntimeError):
            await q.run("m", PASS, boom)
        assert await q.run("m", BARRIER, _op(log, "after")) == "after"
    asyncio.run(main())
