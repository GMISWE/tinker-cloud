"""External weights (save_weights_external): the store's third kind and its
expiry, the URL signer, and the base-class export over the fake backend.

Pure filesystem tests; no server."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.checkpoints import (
    CheckpointKind,
    CheckpointNotFound,
    CheckpointRef,
    InvalidCheckpointPath,
)
from tinkercloud.training.checkpoints import store as store_module
from tinkercloud.training.checkpoints.store import is_expired
from tinkercloud.training.utils import signed_urls

ADAMW = {"type": "adamw"}   # the family record every save carries

E = CheckpointKind.EXTERNAL_WEIGHTS


class TestStoreKind:
    def test_external_is_a_parsed_kind_with_its_own_root(self, store):
        ref = CheckpointRef.parse("tinker://m/external_weights/hf1")
        assert (ref.kind, ref.record_key) == (E, "external_weights--hf1")
        assert store.root(ref) == store.base / "m" / "external_weights" / "hf1"
        with pytest.raises(InvalidCheckpointPath):
            CheckpointRef.parse("tinker://m/external/hf1")  # the SDK type name is not a kind

    def test_files_lists_regular_files_relative_to_root(self, store):
        t = store.begin_save("m", E, "hf1", optimizer_config=ADAMW)
        (t.root / "a.json").write_text("{}")
        (t.root / "sub").mkdir()
        (t.root / "sub" / "b.bin").write_bytes(b"xy")
        store.complete(t.ref)
        files = store.files(t.ref, kind=E)
        assert set(files) == {"a.json", "sub/b.bin"} and files["sub/b.bin"].read_bytes() == b"xy"
        with pytest.raises(CheckpointNotFound):
            store.files(CheckpointRef.make("m", E, "nope"))


class TestTtl:
    def test_ttl_is_recorded_and_listed(self, store):
        t = store.begin_save("m", E, "hf1", ttl_seconds=3600, optimizer_config=ADAMW)
        store.complete(t.ref)
        rec = store.get(t.ref)
        delta = datetime.fromisoformat(rec["expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(minutes=59) < delta <= timedelta(hours=1)
        assert store.list("m")[0]["expires_at"] == rec["expires_at"]
        assert store.begin_save("m", E, "forever", optimizer_config=ADAMW).ref and store.get(CheckpointRef.make("m", E, "forever"))["expires_at"] is None

    def test_expired_reads_as_not_found_and_is_swept(self, store):
        t = store.begin_save("m", E, "hf1", ttl_seconds=3600, optimizer_config=ADAMW)
        (t.root / "weights.bin").write_bytes(b"weights")
        store.complete(t.ref)
        rec = store.get(t.ref)
        assert not is_expired(rec)
        assert store.sweep_expired() == 0
        assert is_expired(rec, now=datetime.now(timezone.utc) + timedelta(hours=2))
        rec["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        store._write(t.ref, rec)
        with pytest.raises(CheckpointNotFound, match="expired"):
            store.require(t.ref)
        assert store.sweep_expired() == 1
        assert not t.root.exists() and store.get(t.ref) is None
        assert store.list("m") == [] and store.usage() == (0, 0)
        assert not is_expired({"status": "completed"})  # pre-ttl record: never expires

    def test_set_ttl_moves_or_clears_expiry_of_a_persistent_checkpoint(self, store):
        w = store.begin_save("m", CheckpointKind.WEIGHTS, "w1", optimizer_config=ADAMW)
        store.complete(w.ref)
        assert store.get(w.ref)["expires_at"] is None
        assert store.set_ttl(w.ref, 7200)
        delta = datetime.fromisoformat(store.get(w.ref)["expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(minutes=119) < delta <= timedelta(hours=2)
        created = store.get(w.ref)["created_at"]
        assert store.set_ttl(w.ref, None) and store.get(w.ref)["expires_at"] is None
        assert store.get(w.ref)["created_at"] == created   # listing order is not disturbed
        assert store.get(w.ref)["status"] == "completed"

    def test_set_ttl_refuses_unknown_ephemeral_and_already_expired(self, store):
        assert not store.set_ttl(CheckpointRef.make("m", E, "nope"), 3600)
        eph = store.begin_save("m", CheckpointKind.SAMPLER_WEIGHTS, "s", persist=False, optimizer_config=ADAMW)
        assert not store.set_ttl(eph.ref, 3600)
        t = store.begin_save("m", E, "hf1", ttl_seconds=3600, optimizer_config=ADAMW)
        store.complete(t.ref)
        rec = store.get(t.ref)
        rec["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        store._write(t.ref, rec)
        assert not store.set_ttl(t.ref, None)   # gone to readers; the sweep still deletes it
        assert store.sweep_expired() == 1

    def test_a_shortened_ttl_is_swept_once_it_has_passed(self, store, monkeypatch):
        t = store.begin_save("m", E, "hf1", optimizer_config=ADAMW)
        (t.root / "weights.bin").write_bytes(b"weights")
        store.complete(t.ref)
        assert store.set_ttl(t.ref, 3600) and store.sweep_expired() == 0

        class _AnHourLater(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + timedelta(hours=1, seconds=1)

        monkeypatch.setattr(store_module, "datetime", _AnHourLater)
        assert store.sweep_expired() == 1
        assert not t.root.exists() and store.get(t.ref) is None and store.list("m") == []


class TestSigner:
    def test_round_trip_tamper_and_expiry(self):
        exp = 2_000_000_000
        sig = signed_urls.sign("k", "m", "hf1", "adapter_config.json", exp)

        def ok(**kw):
            return signed_urls.verify(**{"key": "k", "model_id": "m", "name": "hf1", "relpath": "adapter_config.json",
                                         "exp": exp, "sig": sig, "now": exp - 10, **kw})

        assert ok()
        assert not ok(sig=sig[:-1] + ("0" if sig[-1] != "0" else "1"))
        assert not ok(relpath="adapter_model.safetensors")   # bound to one file
        assert not ok(name="hf2") and not ok(model_id="m2")  # ... of one checkpoint
        assert not ok(exp=exp + 1)                           # exp is inside the signature
        assert not ok(now=exp) and not ok(now=exp + 1)       # expired
        assert not ok(key="other")


class TestExport:
    def _model(self, b, tmp_path, rank):
        return asyncio.run(b.create_model("m", "r", "fake/tiny", 0, lora_config={"rank": rank} if rank else None,
                                          native_root=tmp_path / "native", optimizer="adamw"))

    def test_lora_export_moves_the_adapter_to_the_root(self, tmp_path):
        b = FakeBackend()
        h = self._model(b, tmp_path, rank=4)
        h.w = 0.5
        root = tmp_path / "external_weights" / "hf1"
        root.mkdir(parents=True)
        asyncio.run(b.export_external_weights(h, root, step=3))
        assert sorted(p.name for p in root.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
        assert not (root / ".native").exists()
        assert (root / "adapter_model.safetensors").read_bytes() == b"fake-adapter w=0.5 v=0\n"

    def test_full_finetune_has_no_export_and_leaves_nothing(self, tmp_path):
        b = FakeBackend()
        h = self._model(b, tmp_path, rank=0)
        root = tmp_path / "external_weights" / "hf1"
        root.mkdir(parents=True)
        with pytest.raises(BackendError, match="no HF export"):
            asyncio.run(b.export_external_weights(h, root, step=1))
        assert list(root.iterdir()) == []
