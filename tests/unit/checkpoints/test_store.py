"""The checkpoint store: identity <-> directory, the per-model counter, the
pending/completed/failed record and what a reader is allowed to see.

Pure filesystem tests; no backend, no server."""
import json

import pytest

from tinkercloud.training.checkpoints import (
    CheckpointFailed,
    CheckpointKind,
    CheckpointKindMismatch,
    CheckpointNotFound,
    CheckpointPending,
    CheckpointRef,
    CheckpointStore,
    InvalidCheckpointPath,
)
from tinkercloud.training.storage.metadata import MetadataStorage

W, S = CheckpointKind.WEIGHTS, CheckpointKind.SAMPLER_WEIGHTS


class TestRef:
    def test_round_trip(self):
        ref = CheckpointRef.parse("tinker://model_a/weights/step-3")
        assert (ref.model_id, ref.kind, ref.name) == ("model_a", W, "step-3")
        assert ref.uri == "tinker://model_a/weights/step-3"
        assert CheckpointRef.parse("tinker://m/sampler_weights/final").kind is S

    @pytest.mark.parametrize("bad", [
        "/data/checkpoints/m/x", "tinker://m", "tinker://m/x", "tinker://m/weights",
        "tinker://m/weights/a/b", "tinker://m/native/x", "tinker://m/weights/..", "tinker:///weights/x",
        "", None,
    ])
    def test_everything_else_is_refused(self, bad):
        with pytest.raises(InvalidCheckpointPath):
            CheckpointRef.parse(bad)

    def test_make_validates_segments(self):
        with pytest.raises(InvalidCheckpointPath):
            CheckpointRef.make("m", W, "a/b")


class TestLayout:
    def test_roots_by_kind_and_native_area(self, store):
        assert store.root(CheckpointRef.make("m", W, "c1")) == store.base / "m" / "weights" / "c1"
        assert store.root(CheckpointRef.make("m", S, "c1")) == store.base / "m" / "sampler_weights" / "c1"
        native = store.native_root("m")
        assert native == store.base / "m" / "native" and native.is_dir()


class TestSaveLifecycle:
    def test_begin_creates_an_empty_root_and_a_pending_record(self, store):
        t = store.begin_save("m", W, "c1", weight_version=4)
        assert t.root.is_dir() and not any(t.root.iterdir()) and t.step == 1
        rec = store.get(t.ref)
        assert rec["status"] == "pending" and rec["weight_version"] == 4 and rec["ephemeral"] is False
        with pytest.raises(CheckpointPending):
            store.require(t.ref)
        store.complete(t.ref)
        assert store.require(t.ref) == t.root
        assert store.get(t.ref)["completed_at"]

    def test_failed_save_is_visible_as_failed(self, store):
        t = store.begin_save("m", W, "c1")
        store.fail(t.ref, "disk full")
        with pytest.raises(CheckpointFailed, match="disk full"):
            store.require(t.ref)

    def test_counter_is_monotonic_across_failures_deletes_and_overwrites(self, store):
        a = store.begin_save("m", W, "a"); store.complete(a.ref)
        b = store.begin_save("m", W, "b"); store.fail(b.ref, "x")
        assert (a.step, b.step) == (1, 2)
        store.delete(a.ref)
        c = store.begin_save("m", W, "c"); store.complete(c.ref)
        assert c.step == 3                      # deleted a's 1 and failed b's 2 are never reused
        again = store.begin_save("m", W, "c")   # same name saved twice: new counter, old bytes gone
        assert again.step == 4 and again.root == c.root and not any(again.root.iterdir())
        s = store.begin_save("m", S, "c"); store.complete(s.ref)
        assert s.step == 5                      # one counter per model, both kinds
        assert store.next_step("other") == 1    # per model

    def test_begin_replaces_stale_bytes_under_the_same_name(self, store):
        t = store.begin_save("m", W, "c1")
        (t.root / "old").write_text("x")
        store.complete(t.ref)
        t2 = store.begin_save("m", W, "c1")
        assert not (t2.root / "old").exists()

    def test_ephemeral_save_has_no_step_no_root_and_is_not_listed(self, store):
        t = store.begin_save("m", S, "m_1_abcd", persist=False, weight_version=2)
        assert t.step is None and not t.root.exists()
        store.complete(t.ref)
        assert store.get(t.ref)["ephemeral"] is True
        assert store.list("m") == []
        assert store.next_step("m") == 1        # consumed nothing


class TestReads:
    def test_require_checks_kind(self, store):
        t = store.begin_save("m", S, "s1"); store.complete(t.ref)
        with pytest.raises(CheckpointKindMismatch):
            store.require(t.ref, kind=W)
        with pytest.raises(CheckpointKindMismatch):
            store.resolve_resume(t.ref.uri)
        assert store.require(t.ref, kind=S) == t.root

    def test_unknown_is_not_found(self, store):
        with pytest.raises(CheckpointNotFound):
            store.require(CheckpointRef.make("m", W, "nope"))
        assert store.delete(CheckpointRef.make("m", W, "nope")) is False

    def test_list_reports_uri_status_and_size(self, store):
        t = store.begin_save("m", W, "c1")
        (t.root / "blob").write_bytes(b"12345")
        store.complete(t.ref)
        p = store.begin_save("m", W, "c2")      # still pending: listed, marked
        [c2, c1] = store.list("m")
        assert c1["uri"] == t.ref.uri and c1["size_bytes"] == 5 and c1["status"] == "completed"
        assert c2["uri"] == p.ref.uri and c2["status"] == "pending"


class TestDeletion:
    def test_delete_removes_record_and_bytes_but_not_native(self, store):
        native = store.native_root("m")
        (native / "iter_0000001").mkdir()
        t = store.begin_save("m", W, "c1")
        (t.root / "blob").write_text("x")
        store.complete(t.ref)
        assert store.delete(t.ref) is True
        assert not t.root.exists() and store.get(t.ref) is None
        assert (native / "iter_0000001").is_dir()

    def test_release_model_drops_only_ephemeral_records(self, store):
        keep = store.begin_save("m", W, "c1"); store.complete(keep.ref)
        eph = store.begin_save("m", S, "m_1_x", persist=False); store.complete(eph.ref)
        assert store.release_model("m") == 1
        assert store.get(keep.ref) and store.get(eph.ref) is None
        assert store.native_root("m").is_dir()


class TestBootSweep:
    def test_pending_rows_become_failed(self, store, tmp_path):
        t = store.begin_save("m", W, "c1")
        done = store.begin_save("m", W, "c2"); store.complete(done.ref)
        fresh = CheckpointStore(store.base, MetadataStorage(tmp_path / "meta"))  # a restart
        assert fresh.sweep_pending() == 1
        with pytest.raises(CheckpointFailed, match="restarted"):
            fresh.require(t.ref)
        assert fresh.require(done.ref) == done.root
        assert fresh.sweep_pending() == 0

    def test_records_of_other_shapes_are_ignored(self, store):
        # a foreign json under the model's metadata dir is not a checkpoint record
        d = store.metadata.checkpoints_dir / "m"
        d.mkdir(parents=True)
        (d / "stray.json").write_text(json.dumps({"path": "tinker://m/weights/x"}))
        assert store.records("m") == [] and store.sweep_pending() == 0
