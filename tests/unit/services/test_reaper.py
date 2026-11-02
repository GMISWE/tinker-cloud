"""ModelReaper: a tick schedules one task per expired model and never waits;
a model being freed is not freed twice; shutdown cancels the frees in flight,
then frees every live model at once."""
import asyncio

import pytest

from tinkercloud.training.services.reaper import ModelReaper

T = 0.2          # seconds one free takes in the shutdown test
INTERVAL = 1.0   # the reap interval the tick must beat


class Frees:
    """The free function: records (model, reason); blocks forever on the
    models in `stuck`, else takes `delay` seconds."""

    def __init__(self, stuck=(), delay=0.0):
        self.calls = []
        self.stuck = set(stuck)
        self.delay = delay
        self.done = []

    async def __call__(self, model_id, reason):
        self.calls.append((model_id, reason))
        if model_id in self.stuck:
            await asyncio.Event().wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        self.done.append(model_id)


def expired(*pairs):
    return [(session, list(models)) for session, models in pairs]


def test_tick_frees_each_model_in_its_own_task():
    async def main():
        frees = Frees(stuck={"a"})
        reaper = ModelReaper(frees)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        n = reaper.tick(expired(("sa", ["a"]), ("sb", ["b"])), live={"a", "b"})
        assert n == 2 and loop.time() - t0 < INTERVAL / 10, "the tick returned without waiting on a free"
        await asyncio.wait_for(reaper._frees["b"], timeout=INTERVAL)
        assert frees.done == ["b"], "b was freed while a's free is still blocked"
        assert [m for m, _ in frees.calls] == ["a", "b"]
        assert reaper.inflight() == 1
        assert frees.calls[0][1] == "session sa expired"
    asyncio.run(main())


def test_a_model_being_freed_is_not_freed_twice():
    async def main():
        frees = Frees(stuck={"a"})
        reaper = ModelReaper(frees)
        for _ in range(3):   # three ticks expire a while its first free is blocked
            reaper.tick(expired(("sa", ["a"])), live={"a"})
            await asyncio.sleep(0)
        assert [m for m, _ in frees.calls] == ["a"]
        assert reaper.inflight() == 1
        # a model not in `live` is skipped: nothing to free
        assert reaper.tick(expired(("sx", ["gone"])), live={"a"}) == 0
    asyncio.run(main())


def test_shutdown_cancels_inflight_frees_then_frees_all_at_once():
    async def main():
        frees = Frees(delay=T)
        reaper = ModelReaper(frees)
        blocked = asyncio.Event()

        async def stuck_free(model_id, reason):
            frees.calls.append((model_id, reason))
            blocked.set()
            await asyncio.Event().wait()

        reaper._free = stuck_free
        first = reaper.free("a", "session sa expired")
        await blocked.wait()
        reaper._free = frees

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await reaper.shutdown(["a", "b", "c"])
        elapsed = loop.time() - t0
        assert first.cancelled(), "a's blocked free was cancelled"
        assert elapsed < 2 * T, f"three frees of {T}s ran concurrently, not in {elapsed:.2f}s"
        assert [m for m, _ in frees.calls] == ["a", "a", "b", "c"], "a freed once more, with b and c"
        assert all(r == "shutdown" for _, r in frees.calls[1:])
        assert sorted(frees.done) == ["a", "b", "c"]
        assert reaper.inflight() == 0
    asyncio.run(main())


def test_run_forever_ticks_on_the_interval_and_survives_a_failing_tick():
    async def main():
        frees = Frees()
        reaper = ModelReaper(frees)
        rounds = []

        def expired_fn():
            rounds.append(1)
            if len(rounds) == 1:
                raise RuntimeError("storage hiccup")
            return expired(("s", ["m"]))

        task = asyncio.create_task(reaper.run_forever(0.01, expired_fn, lambda: {"m"}))
        for _ in range(200):
            if frees.done:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        assert len(rounds) >= 2 and frees.done == ["m"], "the failing first tick did not stop the loop"
    asyncio.run(main())


def test_shutdown_runs_every_free_then_reports_a_free_that_raised():
    async def main():
        frees = Frees(delay=T / 4)
        good = frees

        async def free(model_id, reason):
            if model_id == "b":
                raise RuntimeError("the backend refused")
            await good(model_id, reason)

        reaper = ModelReaper(free)
        with pytest.raises(RuntimeError, match="refused"):
            await reaper.shutdown(["a", "b", "c"])
        assert sorted(frees.done) == ["a", "c"], "the other frees ran to completion before the error surfaced"
        assert reaper.inflight() == 0
    asyncio.run(main())


def test_shutdown_reports_an_inflight_free_that_raised_on_cancel_after_freeing_the_rest():
    async def main():
        frees = Frees(delay=T / 4)
        started = asyncio.Event()

        async def raises_on_cancel(model_id, reason):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("teardown broke") from None

        reaper = ModelReaper(raises_on_cancel)
        reaper.free("a", "session sa expired")
        await started.wait()
        reaper._free = frees
        with pytest.raises(RuntimeError, match="teardown broke"):
            await reaper.shutdown(["a", "b", "c"])
        assert sorted(frees.done) == ["a", "b", "c"], "every live model was freed before the error surfaced"
        assert reaper.inflight() == 0
    asyncio.run(main())
