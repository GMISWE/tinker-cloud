"""FuturesStorage: one connection, payload never persisted, seq_id idempotency, legacy rebuild."""
import sqlite3
from datetime import datetime, timedelta

import pytest

from tinkercloud.training.storage import futures
from tinkercloud.training.storage.futures import DuplicateSeqId, FuturesStorage


@pytest.fixture
def store(tmp_path):
    s = FuturesStorage(tmp_path / "futures.db")
    yield s
    s.close()


def _payload(n):
    return {"model_id": "m", "forward_backward_input": {"data": [{"tokens": list(range(n))}]}}


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = datetime(2026, 1, 1, 12)

        @classmethod
        def utcnow(cls):
            return cls.now

    monkeypatch.setattr(futures, "datetime", Clock)
    return Clock


def test_payload_not_persisted(store, tmp_path):
    store.save_future("r1", "forward_backward", _payload(1000), model_id="m", seq_id=1)
    row = sqlite3.connect(tmp_path / "futures.db").execute(
        "SELECT payload_hash, payload_bytes FROM futures WHERE request_id='r1'").fetchone()
    assert row[0] == FuturesStorage.payload_hash(_payload(1000))
    assert 0 < row[1] < 10_000
    cols = {r[1] for r in sqlite3.connect(tmp_path / "futures.db").execute("PRAGMA table_info(futures)")}
    assert "payload" not in cols
    assert "payload" not in store.get_future("r1")


def test_seq_id_retry_and_conflict(store):
    assert store.save_future("r1", "forward_backward", _payload(3), model_id="m", seq_id=7) == "r1"
    # identical retry -> the original request_id, no second row
    assert store.save_future("r2", "forward_backward", _payload(3), model_id="m", seq_id=7) == "r1"
    assert store.get_future("r2") is None
    with pytest.raises(DuplicateSeqId, match="different payload"):
        store.save_future("r3", "forward_backward", _payload(4), model_id="m", seq_id=7)
    with pytest.raises(DuplicateSeqId, match="different operation"):
        store.save_future("r4", "optim_step", _payload(3), model_id="m", seq_id=7)
    # seq_ids are per model
    assert store.save_future("r5", "forward_backward", _payload(3), model_id="other", seq_id=7) == "r5"


def test_status_roundtrip_and_stats(store):
    store.save_future("r1", "optim_step", {}, model_id="m")
    assert store.get_future("r1")["status"] == "pending"
    assert store.update_status("r1", "completed", {"metrics": {"grad_norm": 1.0}})
    fut = store.get_future("r1")
    assert fut["status"] == "completed" and fut["result"] == {"metrics": {"grad_norm": 1.0}}
    assert not store.update_status("nope", "failed", {"error": "x"})
    assert store.get_stats()["completed"] == 1
    assert store.has_training_requests("m") and not store.has_training_requests("other")


def test_cache_is_bounded_and_falls_back_to_db(tmp_path):
    s = FuturesStorage(tmp_path / "f.db", cache_size=4)
    for i in range(10):
        s.save_future(f"r{i}", "forward", {"i": i}, model_id="m")
    assert s.get_stats()["in_memory"] == 4
    assert s.get_future("r0")["status"] == "pending"          # evicted -> reloaded from SQLite
    assert s.update_status("r1", "failed", {"error": "boom"})  # evicted -> updated via SQLite
    assert s.get_future("r1")["result"] == {"error": "boom"}
    s.close()


def test_legacy_schema_is_rebuilt(tmp_path):
    db = tmp_path / "futures.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE futures (request_id TEXT PRIMARY KEY, model_id TEXT, operation TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL)""")
    conn.execute("INSERT INTO futures VALUES ('old','m','forward','{}','pending',NULL,'t','t')")
    conn.commit()
    conn.close()
    s = FuturesStorage(db)
    assert s.get_future("old") is None
    s.save_future("new", "forward", {}, model_id="m", seq_id=1)
    assert s.get_future("new")["seq_id"] == 1
    s.close()
    # a second open of the new schema keeps its rows
    s2 = FuturesStorage(db)
    assert s2.get_future("new") is not None
    s2.close()


def test_cleanup_ages_out_finished_futures_and_keeps_pending_ones(store):
    """A pending future's operation still runs, and a create-via-load resend is matched against it:
    cleanup at any age, 0 included, used to delete it, and the resend then filed a second create."""
    store.save_future("done", "forward", {}, model_id="m")
    store.update_status("done", "completed", {"ok": True})
    store.save_future("running", "load_weights", {"p": 1}, model_id="s:train:0", seq_id=0)
    assert store.cleanup_old_futures(max_age_hours=24) == 0
    assert store.cleanup_old_futures(max_age_hours=0) == 1
    assert store.get_future("done") is None
    assert store.filed("s:train:0", 0, "load_weights", {"p": 1}).request_id == "running"


def test_a_retry_after_cleanup_is_still_the_first_request(store):
    """The pending row owns (model_id, seq_id). A cleanup that removed it would let the retry run the
    operation a second time and the first result could no longer be read."""
    first = store.save_future("r1", "optim_step", _payload(3), model_id="m", seq_id=7)
    assert store.cleanup_old_futures(max_age_hours=0) == 0
    assert store.save_future("r2", "optim_step", _payload(3), model_id="m", seq_id=7) == first == "r1"
    assert store.get_future("r2") is None   # the retry filed no second row


@pytest.mark.parametrize("cached", [True, False])
def test_a_just_finished_future_survives_cleanup_and_a_resend(store, clock, cached):
    first = store.save_future("r1", "optim_step", _payload(3), model_id="m", seq_id=7)
    clock.now += timedelta(hours=4)
    result = {"metrics": {"grad_norm": 1.0}}
    assert store.update_status(first, "completed", result)
    if not cached:
        store._cache.clear()

    removed = store.cleanup_old_futures(max_age_hours=3)
    assert store.save_future("r2", "optim_step", _payload(3), model_id="m", seq_id=7) == first
    assert removed == 0
    assert store.get_stats()["in_memory"] == int(cached)
    assert store.get_future(first)["result"] == result
    assert store.get_future("r2") is None
    with pytest.raises(DuplicateSeqId, match="different payload"):
        store.save_future("r3", "optim_step", _payload(4), model_id="m", seq_id=7)


@pytest.mark.parametrize("cached", [True, False])
def test_cleanup_removes_old_finished_futures_but_keeps_newly_finished_ones(store, clock, cached):
    store.save_future("old", "optim_step", {}, model_id="old-model", seq_id=1)
    store.save_future("recent", "optim_step", {}, model_id="recent-model", seq_id=1)
    store.save_future("pending", "load_weights", {"p": 1}, model_id="s:train:0", seq_id=0)
    clock.now += timedelta(hours=26)
    assert store.update_status("old", "completed", {"ok": True})
    clock.now += timedelta(hours=4)
    assert store.update_status("recent", "completed", {"ok": True})
    if not cached:
        store._cache.clear()

    assert store.cleanup_old_futures(max_age_hours=3) == 1
    assert store.get_stats()["in_memory"] == (2 if cached else 0)
    assert store.get_future("old") is None
    assert store.filed("old-model", 1, "optim_step", {}) is None
    assert store.get_future("recent")["result"] == {"ok": True}
    assert store.filed("s:train:0", 0, "load_weights", {"p": 1}).request_id == "pending"
    assert store.cleanup_old_futures(max_age_hours=3) == 0


@pytest.mark.parametrize("cached", [True, False])
def test_a_30_hour_operation_keeps_its_result_after_default_cleanup(store, clock, cached):
    store.save_future("long", "forward_backward", _payload(3), model_id="m", seq_id=7)
    clock.now += timedelta(hours=30)
    result = {"metrics": {"loss": 0.5}}
    assert store.update_status("long", "completed", result)
    if not cached:
        store._cache.clear()

    assert store.cleanup_old_futures() == 0
    assert store.get_stats()["in_memory"] == int(cached)
    assert store.get_future("long")["result"] == result
    assert store.save_future("retry", "forward_backward", _payload(3), model_id="m", seq_id=7) == "long"


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_cleanup_keeps_a_finished_future_at_the_exact_age_boundary(store, clock, status):
    store.save_future("r1", "optim_step", {}, model_id="m", seq_id=7)
    assert store.update_status("r1", status, {"ok": status == "completed"})
    clock.now += timedelta(hours=3)
    assert store.cleanup_old_futures(max_age_hours=3) == 0
    assert store.get_future("r1")["status"] == status
    clock.now += timedelta(microseconds=1)
    assert store.cleanup_old_futures(max_age_hours=3) == 1
    assert store.get_stats()["in_memory"] == 0
    assert store.get_future("r1") is None


def test_filed_returns_the_status_and_the_failed_create_mark_with_the_id(store):
    """The create-via-load dedup reads the earlier request's status in the same locked read as its id."""
    store.save_future("r1", "load_weights", {"p": 1}, model_id="s:train:0", seq_id=0)
    assert store.filed("s:train:0", 0, "load_weights", {"p": 1}) == ("r1", "pending", False)
    store.mark_create_failed("r1")
    assert store.filed("s:train:0", 0, "load_weights", {"p": 1}) == ("r1", "pending", True)
    store.update_status("r1", "failed", {"error": "x"})
    assert store.filed("s:train:0", 0, "load_weights", {"p": 1}) == ("r1", "failed", True)
    assert store.filed("s:train:0", 1, "load_weights", {"p": 1}) is None
    with pytest.raises(DuplicateSeqId) as e:
        store.filed("s:train:0", 0, "load_weights", {"p": 2})
    assert (e.value.request_id, e.value.status) == ("r1", "failed")


def test_result_proto_is_stored_beside_the_json(store, tmp_path):
    store.save_future("r1", "forward_backward", _payload(3), model_id="m", seq_id=1)
    assert store.get_future("r1")["result_proto"] is None
    store.update_status("r1", "completed", {"metrics": {}}, result_proto=b"\x0a\x01x")
    assert store.get_future("r1")["result_proto"] == b"\x0a\x01x"
    assert store.get_future("r1")["result"] == {"metrics": {}}
    # survives cache eviction (read back from the row)
    store._cache.clear()
    assert store.get_future("r1")["result_proto"] == b"\x0a\x01x"
    # attached after completion (lazy build on first proto retrieve)
    store.save_future("r2", "sample", {}, model_id="m")
    store.update_status("r2", "completed", {"sequences": []})
    store.set_result_proto("r2", b"pb")
    store._cache.clear()
    assert store.get_future("r2")["result_proto"] == b"pb"


def test_pre_proto_schema_is_rebuilt(tmp_path):
    db = tmp_path / "futures.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE futures (request_id TEXT PRIMARY KEY, model_id TEXT, operation TEXT NOT NULL,
                    status TEXT NOT NULL, result TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    seq_id INTEGER, payload_hash TEXT NOT NULL, payload_bytes INTEGER NOT NULL)""")
    conn.execute("INSERT INTO futures VALUES ('old','m','forward','pending',NULL,'t','t',1,'h',0)")
    conn.commit()
    conn.close()
    s = FuturesStorage(db)
    assert s.get_future("old") is None
    s.save_future("new", "forward", {}, model_id="m", seq_id=1)
    assert s.get_future("new")["result_proto"] is None
    s.close()
