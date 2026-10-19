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
    CheckpointStore,
    InvalidCheckpointPath,
)
from tinkercloud.training.checkpoints.store import is_expired
from tinkercloud.training.storage.metadata import MetadataStorage
from tinkercloud.training.utils import signed_urls

E = CheckpointKind.EXTERNAL_WEIGHTS


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "ckpt", MetadataStorage(tmp_path / "meta"))


class TestStoreKind:
    def test_external_is_a_parsed_kind_with_its_own_root(self, store):
        ref = CheckpointRef.parse("tinker://m/external_weights/hf1")
        assert (ref.kind, ref.record_key) == (E, "external_weights--hf1")
        assert store.root(ref) == store.base / "m" / "external_weights" / "hf1"
        with pytest.raises(InvalidCheckpointPath):
            CheckpointRef.parse("tinker://m/external/hf1")  # the SDK type name is not a kind

    def test_files_lists_regular_files_relative_to_root(self, store):
        t = store.begin_save("m", E, "hf1")
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
        t = store.begin_save("m", E, "hf1", ttl_seconds=3600)
        store.complete(t.ref)
        rec = store.get(t.ref)
        delta = datetime.fromisoformat(rec["expires_at"]) - datetime.now(timezone.utc)
        assert timedelta(minutes=59) < delta <= timedelta(hours=1)
        assert store.list("m")[0]["expires_at"] == rec["expires_at"]
        assert store.begin_save("m", E, "forever").ref and store.get(CheckpointRef.make("m", E, "forever"))["expires_at"] is None

    def test_expired_reads_as_not_found(self, store):
        t = store.begin_save("m", E, "hf1", ttl_seconds=3600)
        store.complete(t.ref)
        rec = store.get(t.ref)
        assert not is_expired(rec)
        assert is_expired(rec, now=datetime.now(timezone.utc) + timedelta(hours=2))
        rec["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        store._write(t.ref, rec)
        with pytest.raises(CheckpointNotFound, match="expired"):
            store.require(t.ref)
        assert not is_expired({"status": "completed"})  # pre-ttl record: never expires


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
                                          native_root=tmp_path / "native"))

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
