"""Miles resumes from the Megatron iter_* directory a checkpoint was published
from. The iter number is the store's per-model save counter, so the directory
is recorded beside the interchange adapter at publish time and resolved
through that record -- for load_weights (the train group) and for
create_model(resume_from) (the builder's --load). A root without the record
is not a Miles-native checkpoint, never a guess."""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinkercloud.training.backends.miles import backend as miles_backend, model_setup
from tinkercloud.training.backends.miles.backend import MilesBackend


def _handle():
    return SimpleNamespace(
        adapter_slot=None, lock=asyncio.Lock(), rollout_manager=None, created_from_checkpoint=False,
        args=SimpleNamespace(save="/tmp/save"), train_group=SimpleNamespace(load_checkpoint=AsyncMock()),
    )


def test_publish_records_the_native_dir(tmp_path, monkeypatch):
    save = tmp_path / "native"
    old, new = save / "iter_0000001" / "adapter", save / "iter_0000002" / "adapter"
    for d in (old, new):
        d.mkdir(parents=True)
    os.utime(old, (1, 1))
    monkeypatch.setattr(miles_backend, "export_hf_adapter", lambda src, dst: None)
    root = tmp_path / "ckpt" / "m" / "weights" / "resume"
    root.mkdir(parents=True)
    miles_backend._publish_native_adapter(SimpleNamespace(save=str(save)), str(root))
    assert (root / model_setup.NATIVE_CHECKPOINT_POINTER).read_text() == str(new.parent)
    assert model_setup.resolve_native_checkpoint(str(root)) == str(new.parent)


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_checkpoint_uses_the_record_and_passes_the_flag(tmp_path, optimizer):
    native = tmp_path / "native" / "iter_0000007"
    native.mkdir(parents=True)
    root = tmp_path / "ckpt" / "resume"
    root.mkdir(parents=True)
    model_setup.record_native_checkpoint(str(root), str(native))
    h = _handle()
    asyncio.run(MilesBackend().load_checkpoint(h, root, optimizer=optimizer))
    h.train_group.load_checkpoint.assert_awaited_once_with(str(native), load_optimizer=optimizer)
    assert h.created_from_checkpoint is True


def test_missing_native_dir_is_an_error_not_a_guess(tmp_path):
    root = tmp_path / "ckpt" / "resume"
    root.mkdir(parents=True)
    model_setup.record_native_checkpoint(str(root), str(tmp_path / "gone"))
    with pytest.raises(Exception, match="is gone"):
        asyncio.run(MilesBackend().load_checkpoint(_handle(), root))


def test_root_without_record_is_not_miles_native(tmp_path):
    root = tmp_path / "ckpt" / "foreign"
    root.mkdir(parents=True)
    with pytest.raises(Exception, match="not a Miles-native checkpoint"):
        asyncio.run(MilesBackend().load_checkpoint(_handle(), root))
