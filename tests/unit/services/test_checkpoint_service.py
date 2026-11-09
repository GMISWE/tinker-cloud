"""CheckpointService's read side: every run's checkpoints a page at a time, and the newest
loadable checkpoint of each kind a TrainingRun reports. Store on disk, no backend calls."""
from datetime import datetime, timedelta, timezone

import pytest

from tinkercloud.training.checkpoints import CheckpointKind, CheckpointSaveOptions, CheckpointStore
from tinkercloud.training.services.checkpoint_service import CheckpointService
from tinkercloud.training.storage.metadata import MetadataStorage

ADAMW = {"type": "adamw"}
SAVE_OPTIONS = CheckpointSaveOptions(ttl_seconds=None, overwrite=True, user_metadata=None)
W, S, E = CheckpointKind.WEIGHTS, CheckpointKind.SAMPLER_WEIGHTS, CheckpointKind.EXTERNAL_WEIGHTS


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "ckpt", MetadataStorage(tmp_path / "meta"))


@pytest.fixture
def service(store):
    return CheckpointService(backend=None, store=store)


def _save(store, model_id, kind, name, created_at, payload=b"", complete=True, persist=True):
    t = store.begin_save(model_id, kind, name, options=SAVE_OPTIONS, persist=persist, optimizer_config=ADAMW)
    if persist and payload:
        (t.root / "w.bin").write_bytes(payload)
    if complete:
        store.complete(t.ref)
    rec = store.get(t.ref)
    rec["created_at"] = created_at   # deterministic order; the store keeps created_at across writes
    store._write(t.ref, rec)
    return t.ref


class TestListAll:
    def test_pages_every_run_newest_first_with_a_total(self, store, service):
        _save(store, "run_a", W, "a1", "2026-01-01T00:00:01", b"x")
        _save(store, "run_b", W, "b1", "2026-01-01T00:00:03", b"xyz")
        _save(store, "run_a", S, "a1", "2026-01-01T00:00:02", b"xy")
        _save(store, "run_b", S, "eph", "2026-01-01T00:00:04", persist=False)   # ephemeral: never listed

        page, total = service.list_all_checkpoints(limit=2, offset=0)
        assert total == 3
        assert [c.tinker_path for c in page] == ["tinker://run_b/weights/b1", "tinker://run_a/sampler_weights/a1"]
        assert [c.size_bytes for c in page] == [3, 2]
        assert page[1].checkpoint_id == "sampler_weights/a1" and page[1].checkpoint_type == "sampler"

        rest, total = service.list_all_checkpoints(limit=2, offset=2)
        assert total == 3 and [c.tinker_path for c in rest] == ["tinker://run_a/weights/a1"]
        assert service.list_all_checkpoints(limit=5, offset=9) == ([], 3)

    def test_sizes_are_walked_for_the_page_only(self, store, service, monkeypatch):
        for i in range(5):
            _save(store, "run", W, f"c{i}", f"2026-01-01T00:00:0{i}", b"x")
        walked = []
        size_bytes = store.size_bytes
        monkeypatch.setattr(store, "size_bytes", lambda ref: walked.append(ref.name) or size_bytes(ref))
        page, total = service.list_all_checkpoints(limit=2, offset=1)
        assert total == 5 and walked == ["c3", "c2"] and [c.size_bytes for c in page] == [1, 1]


class TestLatest:
    def test_newest_loadable_of_each_kind(self, store, service):
        _save(store, "run", W, "w1", "2026-01-01T00:00:01")
        _save(store, "run", W, "w2", "2026-01-01T00:00:03")
        _save(store, "run", W, "w3", "2026-01-01T00:00:05", complete=False)   # pending: not loadable
        _save(store, "run", S, "s1", "2026-01-01T00:00:02")
        _save(store, "run", E, "hf", "2026-01-01T00:00:09")                   # not a kind TrainingRun names
        latest = service.latest_checkpoints("run")
        assert latest["last_checkpoint"].tinker_path == "tinker://run/weights/w2"
        assert latest["last_checkpoint"].checkpoint_type == "training"
        assert latest["last_sampler_checkpoint"].tinker_path == "tinker://run/sampler_weights/s1"

    def test_failed_and_expired_are_skipped(self, store, service):
        ok = _save(store, "run", W, "ok", "2026-01-01T00:00:01")
        bad = _save(store, "run", W, "bad", "2026-01-01T00:00:02", complete=False)
        store.fail(bad, "boom")
        old = _save(store, "run", W, "old", "2026-01-01T00:00:03")
        rec = store.get(old)
        rec["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        store._write(old, rec)
        assert service.latest_checkpoints("run")["last_checkpoint"].tinker_path == ok.uri
