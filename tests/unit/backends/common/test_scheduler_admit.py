"""GroupScheduler under an admission rule of its own: what DEFER and END do to the order, what the
rule is shown, and what happens when it raises.

The rule here reads the next operation's key: "join", "defer", "raise", anything else ends the run.
Every case queues its operations behind a held barrier, so the worker sees them all at once."""
import asyncio

import pytest

from tests.fakes.held_queue import queue_behind_barrier, run_behind_barrier
from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.common.scheduler import Admit, GroupScheduler


def by_key(run, nxt):
    if nxt.key == "raise":
        raise RuntimeError("bad rule")
    return {"join": Admit.JOIN, "defer": Admit.DEFER}.get(nxt.key, Admit.END)


class Rig:
    """`log` holds what ran, in order: a tuple of payloads for a pass, a name for a barrier.
    `offered` holds every call of the rule: the run's payloads, the next operation's kind and key."""

    def __init__(self):
        self.log = []
        self.offered = []
        self.pass_gate = asyncio.Event()
        self.pass_gate.set()
        self.s = GroupScheduler("g", self._admit, self._run_pass, "test")

    def _admit(self, run, nxt):
        self.offered.append((tuple(p.payload for p in run.passes), type(nxt).__name__, nxt.key))
        return by_key(run, nxt)

    async def _run_pass(self, key, payloads):
        self.log.append(tuple(payloads))
        await self.pass_gate.wait()
        return list(payloads)

    def p(self, payload, key="join"):
        return self.s.submit_pass(key, payload)

    def b(self, name, key=None):
        async def fn():
            self.log.append(name)
            return name
        return self.s.submit_barrier(fn, key)

    async def queue(self, *submissions):
        return await queue_behind_barrier(self.s.submit_barrier, submissions)

    async def run(self, *submissions):
        return await run_behind_barrier(self.s.submit_barrier, submissions)


def test_deferred_operations_run_after_the_run_in_arrival_order():
    """Two deferred barriers wait while the passes behind them join, then run in the order they came."""
    async def main():
        r = Rig()
        results = await r.run(r.p("a"), r.b("d1", "defer"), r.p("b"), r.b("d2", "defer"), r.p("c"))
        assert results == ["a", "d1", "b", "d2", "c"]
        assert r.log == [("a", "b", "c"), "d1", "d2"]
    asyncio.run(main())


def test_an_operation_that_ends_a_run_runs_behind_the_deferred_ones():
    """END leaves its operation behind what the run deferred. Here the deferred one is a barrier, so
    nothing passes the two before they run."""
    async def main():
        r = Rig()
        await r.run(r.p("a"), r.b("d1", "defer"), r.b("e", "end"), r.p("b"))
        assert r.log == [("a",), "d1", "e", ("b",)]
    asyncio.run(main())


def test_a_waiting_operation_is_not_offered_to_admit_again():
    """A deferred pass and the pass that ended the run each head a run of their own; the rule sees
    each of them once, as the next operation of the first run."""
    async def main():
        r = Rig()
        await r.run(r.p("a"), r.p("p", "defer"), r.p("x", "end"), r.p("y"))
        assert r.log == [("a",), ("p", "y"), ("x",)]
        assert [(run, key) for run, _, key in r.offered] == [(("a",), "defer"), (("a",), "end"), (("p",), "join")]
    asyncio.run(main())


def test_a_barriers_key_reaches_admit():
    """The rule is shown a barrier with the key its submitter gave, and one without a key as None."""
    async def main():
        r = Rig()
        await r.run(r.p("a"), r.b("owned", "tenant-7"), r.p("b"), r.b("plain"))
        assert r.offered == [(("a",), "Barrier", "tenant-7"), (("b",), "Barrier", None)]
    asyncio.run(main())


def test_abandon_fails_a_deferred_operation():
    """An operation that waits because a run deferred it fails with the reason, as a queued one does."""
    async def main():
        r = Rig()
        r.pass_gate.clear()
        release, (a, d1) = await r.queue(r.p("a"), r.b("d1", "defer"))
        await release()
        while r.log != [("a",)]:            # the pass is in progress; d1 waits behind it
            await asyncio.sleep(0)
        r.s.abandon("group lost")
        r.pass_gate.set()
        assert await asyncio.wait_for(a, 5) == "a"
        with pytest.raises(BackendError, match="group lost"):
            await asyncio.wait_for(d1, 5)
        assert r.log == [("a",)]
    asyncio.run(main())


def test_an_admit_that_raises_fails_the_requests_it_held_and_the_queue_goes_on():
    """The run so far and the operation under ruling get the rule's error; no pass runs for them, and
    the operation queued behind them completes."""
    async def main():
        r = Rig()
        a, b, c, d = await r.run(r.p("a"), r.p("b"), r.p("c", "raise"), r.p("d"))
        assert [repr(x) for x in (a, b, c)] == ["RuntimeError('bad rule')"] * 3
        assert d == "d"
        assert r.log == [("d",)]
    asyncio.run(main())
