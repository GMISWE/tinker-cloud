"""FuturesStorage: one connection, payload never persisted, seq_id idempotency, legacy rebuild."""
import sqlite3

import pytest

from tinkercloud.training.storage.futures import DuplicateSeqId, FuturesStorage


@pytest.fixture
def store(tmp_path):
    s = FuturesStorage(tmp_path / "futures.db")
    yield s
    s.close()


def _payload(n):
    return {"model_id": "m", "forward_backward_input": {"data": [{"tokens": list(range(n))}]}}


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


def test_cleanup(store):
    store.save_future("r1", "forward", {}, model_id="m")
    assert store.cleanup_old_futures(max_age_hours=24) == 0
    assert store.cleanup_old_futures(max_age_hours=0) == 1
    assert store.get_future("r1") is None


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
