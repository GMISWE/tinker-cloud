"""storage.sample_futures: in-memory sample futures, per-session cursor log, TTLs, cancel."""
import asyncio

import pytest

from tinkercloud.training.storage import sample_futures as sf
from tinkercloud.training.storage.sample_futures import SampleFutureStore


def _reg(store, rid, session="sess", seq=None, model="m"):
    return store.register(rid, "asample", model, session, seq)


def test_register_complete_fail_and_done_event():
    s = SampleFutureStore()
    assert _reg(s, "r1", seq=1) == "r1"
    fut = s.get("r1")
    assert fut.status == "pending" and not fut.done.is_set() and fut.session == ("sess", 0)
    s.complete("r1", {"sequences": []}, b"proto")
    assert fut.status == "completed" and fut.done.is_set() and fut.payload_size == 5
    _reg(s, "r2", seq=2)
    s.fail("r2", "boom")
    assert s.get("r2").status == "failed" and s.get("r2").error == "boom"
    s.complete("r2", {}, b"")   # terminal is terminal
    assert s.get("r2").status == "failed"


def test_seq_id_retry_returns_the_owner():
    s = SampleFutureStore()
    assert _reg(s, "r1", seq=7) == "r1"
    assert _reg(s, "r9", seq=7) == "r1"          # retry: same owner, no second future
    assert s.get("r9") is None
    assert _reg(s, "r3", session="other", seq=7) == "r3"   # seq_ids are per session
    assert _reg(s, "r4", session=None, seq=None) == "r4"   # no session: no idempotency, no log


def test_clone_id_comes_from_the_seq_id_block():
    s = SampleFutureStore()
    _reg(s, "r1", seq=3_000_000_005)
    assert s.get("r1").session == ("sess", 3)
    assert sf.clone_of(None) == 0


def test_poll_cursor_ack_and_lost_response_replay():
    async def main():
        s = SampleFutureStore()
        for i in range(3):
            _reg(s, f"r{i}", seq=i)
        s.complete("r0", {}, b"aa")
        s.fail("r1", "x")
        s.complete("r2", {}, b"cccc")
        key = ("sess", 0)
        entries, cursor = await s.poll(key, 0, 0.01)
        assert [e[0] for e in entries] == ["r0", "r1", "r2"] and cursor == 3
        assert entries[0][1:] == ("finished", 2) and entries[1][1:] == ("failed", None)
        # lost response: the same prev_cursor returns the same batch
        entries, cursor = await s.poll(key, 0, 0.01)
        assert [e[0] for e in entries] == ["r0", "r1", "r2"] and cursor == 3
        # client advanced to 2: entries 0 and 1 are acknowledged and dropped
        entries, cursor = await s.poll(key, 2, 0.01)
        assert [e[0] for e in entries] == ["r2"] and cursor == 3
        entries, _ = await s.poll(key, 0, 0.01)
        assert [e[0] for e in entries] == ["r2"]   # trimmed entries do not come back
        # nothing new: waits, then returns empty with the cursor unchanged
        entries, cursor = await s.poll(key, 3, 0.02)
        assert entries == [] and cursor == 3
    asyncio.run(main())


def test_poll_wakes_on_completion():
    async def main():
        s = SampleFutureStore()
        _reg(s, "r1", seq=1)
        waiter = asyncio.create_task(s.poll(("sess", 0), 0, 5.0))
        await asyncio.sleep(0.01)
        s.complete("r1", {}, b"p")
        entries, cursor = await asyncio.wait_for(waiter, 1.0)
        assert [e[0] for e in entries] == ["r1"] and cursor == 1
    asyncio.run(main())


def test_client_ahead_of_a_restarted_server_is_renumbered():
    async def main():
        s = SampleFutureStore()
        _reg(s, "r1", seq=41)
        s.complete("r1", {}, b"p")
        entries, cursor = await s.poll(("sess", 0), 40, 0.01)   # SDK cursor from before our restart
        assert [e[0] for e in entries] == ["r1"] and cursor == 41
    asyncio.run(main())


def test_cancel_pending_and_terminal():
    async def main():
        s = SampleFutureStore()
        _reg(s, "r1", seq=1)
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(slow())
        s.attach_task("r1", task)
        await started.wait()
        assert s.cancel("r1") is True
        await asyncio.sleep(0)
        assert task.cancelled() and s.get("r1").status == "failed"
        assert s.cancel("r1") is False
        with pytest.raises(KeyError):
            s.cancel("nope")
        _reg(s, "r2", seq=2, model="gone")
        _reg(s, "r3", seq=3, model="kept")
        assert s.cancel_model("gone") == 1
        assert s.get("r2").status == "failed" and "deleted" in s.get("r2").error
        assert s.get("r3").status == "pending"
        assert s.get("r1").error == "cancelled by client"
    asyncio.run(main())


def test_sweep_ttls():
    s = SampleFutureStore()
    t0 = 1000.0
    _reg(s, "pending", seq=1)
    s.get("pending").created_at = t0
    _reg(s, "done", seq=2)
    s.complete("done", {}, b"p")
    s.get("done").completed_at = t0
    _reg(s, "taken", seq=3)
    s.complete("taken", {}, b"p")
    s.get("taken").completed_at = t0
    s.mark_retrieved("taken")
    s.get("taken").retrieved_at = t0
    assert s.sweep(now=t0 + 100) == 0
    assert s.sweep(now=t0 + sf.RETRIEVED_TTL_S + 1) == 1 and s.get("taken") is None
    assert s.sweep(now=t0 + sf.COMPLETED_TTL_S + 1) == 1 and s.get("done") is None
    # a stuck pending sample is FAILED (logged, done set), not dropped; it ages out afterwards
    assert s.sweep(now=t0 + sf.PENDING_TTL_S + 1) == 0
    stuck = s.get("pending")
    assert stuck.status == "failed" and stuck.done.is_set() and "pending longer" in stuck.error
    stuck.completed_at = t0
    assert s.sweep(now=t0 + sf.COMPLETED_TTL_S + 1) == 1 and s.get("pending") is None
    assert s.stats()["total"] == 0 and s.stats()["sessions"] == 0
    assert _reg(s, "again", seq=2) == "again"   # a swept owner no longer claims the seq_id


def test_cursor_survives_log_drop_and_expired_entries_trim_from_the_left():
    async def main():
        s = SampleFutureStore()
        key = ("sess", 0)
        _reg(s, "r1", seq=1)
        s.complete("r1", {}, b"p")
        entries, cursor = await s.poll(key, 0, 0.01)      # SDK now holds cursor 1, then idles (no ack poll)
        assert cursor == 1
        s.mark_retrieved("r1")
        s.get("r1").retrieved_at = 0.0
        assert s.sweep(now=sf.RETRIEVED_TTL_S + 1) == 1     # r1 gone -> log dropped
        assert s.stats()["sessions"] == 0
        # next sample recreates the log; it must continue from cursor 1, not restart at 0
        _reg(s, "r2", seq=2)
        s.complete("r2", {}, b"q")
        entries, cursor = await s.poll(key, 1, 0.01)
        assert [e[0] for e in entries] == ["r2"] and cursor == 2
        # expired entries are trimmed from the left even without an ack
        _reg(s, "r3", seq=3)
        s.complete("r3", {}, b"r")
        s.get("r2").completed_at = 0.0
        s.sweep(now=sf.COMPLETED_TTL_S + 1)                 # r2 expires, r3 stays
        entries, cursor = await s.poll(key, 1, 0.01)
        assert [e[0] for e in entries] == ["r3"] and cursor == 3
    asyncio.run(main())
