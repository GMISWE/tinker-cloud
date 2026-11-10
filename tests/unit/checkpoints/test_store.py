"""The checkpoint store: identity <-> directory, the per-model counter, the
pending/completed/failed record and what a reader is allowed to see.

Pure filesystem tests; no backend, no server."""
import json
import shutil

import pytest

from tinkercloud.training.checkpoints import (
    CheckpointDeleteFailed,
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

ADAMW = {"type": "adamw"}   # the family record every save carries

W, S = CheckpointKind.WEIGHTS, CheckpointKind.SAMPLER_WEIGHTS


def _rmtree_stuck_on(name):
    """A shutil.rmtree that cannot remove the tree called `name`, behaving as the real one does: it raises,
    or with ignore_errors=True it quietly leaves the files. Other trees go through the real function."""
    real = shutil.rmtree

    def rmtree(path, ignore_errors=False, **kwargs):
        if path.name != name:
            return real(path, ignore_errors=ignore_errors, **kwargs)
        if ignore_errors:
            return
        raise PermissionError(13, "Permission denied", str(path))
    return rmtree


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

    def test_make_refuses_names_that_exceed_255_bytes(self):
        # Name alone is too long.
        with pytest.raises(InvalidCheckpointPath, match="exceeds 255 UTF-8 bytes"):
            CheckpointRef.make("m", W, "x" * 256)
        # Name fits alone but the record key (kind--name) does not.
        just_under = "x" * (255 - len("weights--"))
        CheckpointRef.make("m", W, just_under)  # longest name that fits a weights record key
        with pytest.raises(InvalidCheckpointPath, match="record key"):
            CheckpointRef.make("m", W, just_under + "x")

    def test_long_name_does_not_remove_existing_checkpoint(self, store):
        # The bug: begin_save removed the old directory before the write failed.
        ok = CheckpointRef.make("m", W, "ok-name")
        store.begin_save("m", W, "ok-name", optimizer_config=ADAMW)
        store.complete(ok)
        assert store.require(ok).is_dir()
        with pytest.raises(InvalidCheckpointPath):
            store.begin_save("m", W, "x" * 256, optimizer_config=ADAMW)
        assert store.require(ok).is_dir()  # old checkpoint still there

    def test_make_refuses_name_measured_in_bytes_not_characters(self):
        # 130 two-byte characters == 260 bytes, refused because the name alone is too long.
        with pytest.raises(InvalidCheckpointPath, match="exceeds 255 UTF-8 bytes"):
            CheckpointRef.make("m", W, "é" * 130)
        # 123 two-byte characters == 246 bytes, the longest that fits a weights record key.
        CheckpointRef.make("m", W, "é" * 123)


class TestLayout:
    def test_roots_by_kind_and_native_area(self, store):
        assert store.root(CheckpointRef.make("m", W, "c1")) == store.base / "m" / "weights" / "c1"
        assert store.root(CheckpointRef.make("m", S, "c1")) == store.base / "m" / "sampler_weights" / "c1"
        native = store.native_root("m")
        assert native == store.base / "m" / "native" and native.is_dir()


class TestSaveLifecycle:
    def test_begin_creates_an_empty_root_and_a_pending_record(self, store):
        t = store.begin_save("m", W, "c1", weight_version=4, optimizer_config=ADAMW)
        assert t.root.is_dir() and not any(t.root.iterdir()) and t.step == 1
        rec = store.get(t.ref)
        assert rec["status"] == "pending" and rec["weight_version"] == 4 and rec["ephemeral"] is False
        with pytest.raises(CheckpointPending):
            store.require(t.ref)
        store.complete(t.ref)
        assert store.require(t.ref) == t.root
        assert store.get(t.ref)["completed_at"]

    def test_failed_save_is_visible_as_failed(self, store):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        store.fail(t.ref, "disk full")
        with pytest.raises(CheckpointFailed, match="disk full"):
            store.require(t.ref)

    def test_counter_is_monotonic_across_failures_deletes_and_overwrites(self, store):
        a = store.begin_save("m", W, "a", optimizer_config=ADAMW); store.complete(a.ref)
        b = store.begin_save("m", W, "b", optimizer_config=ADAMW); store.fail(b.ref, "x")
        assert (a.step, b.step) == (1, 2)
        store.delete(a.ref)
        c = store.begin_save("m", W, "c", optimizer_config=ADAMW); store.complete(c.ref)
        assert c.step == 3                      # deleted a's 1 and failed b's 2 are never reused
        again = store.begin_save("m", W, "c", optimizer_config=ADAMW)   # same name saved twice: new counter, old bytes gone
        assert again.step == 4 and again.root == c.root and not any(again.root.iterdir())
        s = store.begin_save("m", S, "c", optimizer_config=ADAMW); store.complete(s.ref)
        assert s.step == 5                      # one counter per model, both kinds
        assert store.next_step("other") == 1    # per model

    def test_begin_replaces_stale_bytes_under_the_same_name(self, store):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "old").write_text("x")
        store.complete(t.ref)
        t2 = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        assert not (t2.root / "old").exists()

    def test_ephemeral_save_has_no_step_no_root_and_is_not_listed(self, store):
        t = store.begin_save("m", S, "m_1_abcd", persist=False, weight_version=2, optimizer_config=ADAMW)
        assert t.step is None and not t.root.exists()
        store.complete(t.ref)
        assert store.get(t.ref)["ephemeral"] is True
        assert store.list("m") == []
        assert store.next_step("m") == 1        # consumed nothing


class TestReads:
    def test_require_checks_kind(self, store):
        t = store.begin_save("m", S, "s1", optimizer_config=ADAMW); store.complete(t.ref)
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
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_bytes(b"12345")
        store.complete(t.ref)
        p = store.begin_save("m", W, "c2", optimizer_config=ADAMW)      # still pending: listed, marked
        [c2, c1] = store.list("m")
        assert c1["uri"] == t.ref.uri and c1["size_bytes"] == 5 and c1["status"] == "completed"
        assert c2["uri"] == p.ref.uri and c2["status"] == "pending"


class TestDeletion:
    def test_delete_removes_record_and_bytes_but_not_native(self, store):
        native = store.native_root("m")
        (native / "iter_0000001").mkdir()
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_text("x")
        store.complete(t.ref)
        assert store.delete(t.ref) is True
        assert not t.root.exists() and store.get(t.ref) is None
        assert (native / "iter_0000001").is_dir()

    def test_a_delete_that_cannot_remove_the_bytes_keeps_the_record_and_says_why(self, store, monkeypatch):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_text("x")
        store.complete(t.ref)
        completed_at = store.get(t.ref)["completed_at"]
        monkeypatch.setattr(shutil, "rmtree", _rmtree_stuck_on("c1"))
        with pytest.raises(CheckpointDeleteFailed, match="Could not delete checkpoint .*Permission denied") as caught:
            store.delete(t.ref)   # a CheckpointError, so the client is told why instead of a bare 500
        assert caught.value.status_code == 500 and isinstance(caught.value.__cause__, PermissionError)
        rec = store.get(t.ref)   # the record is kept, so the checkpoint is still listed while its bytes remain
        assert rec["status"] == "failed" and "delete did not finish" in rec["error"]
        assert rec["completed_at"] == completed_at   # when the save finished is still true
        assert (t.root / "blob").is_file()
        with pytest.raises(CheckpointFailed, match="partly deleted"):   # not "creation failed": the save worked
            store.require(t.ref)
        monkeypatch.undo()
        assert store.delete(t.ref) is True   # a retry removes the bytes and then the record
        assert not t.root.exists() and store.get(t.ref) is None

    def test_a_checkpoint_still_being_saved_is_not_deleted(self, store):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_text("half")
        with pytest.raises(CheckpointPending, match="cannot be deleted yet"):
            store.delete(t.ref)
        assert (t.root / "blob").is_file() and store.get(t.ref)["status"] == "pending"
        store.complete(t.ref)   # the backend finishing afterwards finds the record it expects
        assert store.require(t.ref) == t.root
        assert store.delete(t.ref) is True

    def test_the_expiry_sweep_survives_a_checkpoint_it_cannot_delete(self, store, monkeypatch):
        stuck = store.begin_save("m", W, "stuck", ttl_seconds=0, optimizer_config=ADAMW)
        store.complete(stuck.ref)
        fine = store.begin_save("m", W, "fine", ttl_seconds=0, optimizer_config=ADAMW)
        store.complete(fine.ref)
        monkeypatch.setattr(shutil, "rmtree", _rmtree_stuck_on("stuck"))
        assert store.sweep_expired() == 1   # "fine" went; "stuck" is logged and left for the next sweep
        assert store.get(fine.ref) is None and store.get(stuck.ref) is not None
        monkeypatch.undo()
        assert store.sweep_expired() == 1 and store.get(stuck.ref) is None

    def test_a_failing_metadata_volume_is_reported_as_a_failed_delete_not_a_raw_oserror(self, store, monkeypatch):
        """The volume that holds the records can be the thing that fails (read-only remount, I/O error)."""
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_text("x")
        store.complete(t.ref)

        def broken(*_a, **_k):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(shutil, "rmtree", _rmtree_stuck_on("c1"))
        monkeypatch.setattr(MetadataStorage, "save_checkpoint", broken)   # the failure mark cannot be written either
        with pytest.raises(CheckpointDeleteFailed, match="Read-only file system|Permission denied"):
            store.delete(t.ref)

    def test_a_record_that_cannot_be_removed_after_the_bytes_went_is_a_failed_delete(self, store, monkeypatch):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        store.complete(t.ref)

        def broken(*_a, **_k):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(MetadataStorage, "delete_checkpoint", broken)
        with pytest.raises(CheckpointDeleteFailed, match="Input/output error"):
            store.delete(t.ref)
        assert not t.root.exists()   # the bytes are gone; only the record is left, and a retry removes it
        monkeypatch.undo()
        assert store.delete(t.ref) is True and store.get(t.ref) is None

    def test_the_expiry_sweep_survives_a_metadata_volume_that_is_failing(self, store, monkeypatch):
        t = store.begin_save("m", W, "c1", ttl_seconds=0, optimizer_config=ADAMW)
        store.complete(t.ref)

        def broken(*_a, **_k):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(MetadataStorage, "delete_checkpoint", broken)
        assert store.sweep_expired() == 0   # logged, not raised: sweep_forever has no try and would end for good

    def test_the_stored_error_and_the_response_do_not_carry_a_server_path(self, store, monkeypatch):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        (t.root / "blob").write_text("x")
        store.complete(t.ref)
        monkeypatch.setattr(shutil, "rmtree", _rmtree_stuck_on("c1"))
        with pytest.raises(CheckpointDeleteFailed) as caught:
            store.delete(t.ref)
        for text in (str(caught.value), store.get(t.ref)["error"]):
            assert str(store.base) not in text and "Permission denied" in text

    def test_a_checkpoint_that_stays_stuck_logs_one_line_per_sweep_not_a_traceback(self, store, monkeypatch, caplog):
        t = store.begin_save("m", W, "stuck", ttl_seconds=0, optimizer_config=ADAMW)
        store.complete(t.ref)
        monkeypatch.setattr(shutil, "rmtree", _rmtree_stuck_on("stuck"))
        with caplog.at_level("WARNING"):
            store.sweep_expired()
            store.sweep_expired()   # the second sweep finds a record that already says delete_failed
        assert not any(r.exc_info for r in caplog.records)
        assert any("Could not delete expired checkpoint" in r.getMessage() for r in caplog.records)

    def test_release_model_drops_only_ephemeral_records(self, store):
        keep = store.begin_save("m", W, "c1", optimizer_config=ADAMW); store.complete(keep.ref)
        eph = store.begin_save("m", S, "m_1_x", persist=False, optimizer_config=ADAMW); store.complete(eph.ref)
        assert store.release_model("m") == 1
        assert store.get(keep.ref) and store.get(eph.ref) is None
        assert store.native_root("m").is_dir()


class TestBootSweep:
    def test_pending_rows_become_failed(self, store, tmp_path):
        t = store.begin_save("m", W, "c1", optimizer_config=ADAMW)
        done = store.begin_save("m", W, "c2", optimizer_config=ADAMW); store.complete(done.ref)
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
