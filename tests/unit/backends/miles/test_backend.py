"""MilesBackend without a cluster: the boot-time clip guard, the resume path, and the datum lengths it reads.

Miles reads eps_clip / eps_clip_high once at actor boot and both RL losses share them, so one actor group serves either
importance_sampling or ppo and _check_miles_clip_config refuses the other (D11). A resume loads from the Megatron iter_*
directory a checkpoint was published from; a missing record is an error, not a guess."""
import asyncio
import os
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinkercloud.training.backends.base import UnsupportedFeatureError
from tinkercloud.training.backends.miles import backend as miles_backend, model_setup
from tinkercloud.training.backends.miles.backend import MilesBackend, MilesHandle, _check_miles_clip_config
from tinkercloud.training.backends.miles.config import NO_CLIP_EPS, NO_CLIP_EPS_HIGH


def _clip_handle(eps_clip, eps_clip_high):
    return MilesHandle(model_id="m", backend_type="miles",
                       args=Namespace(eps_clip=eps_clip, eps_clip_high=eps_clip_high))


UNCLIPPED = _clip_handle(NO_CLIP_EPS, NO_CLIP_EPS_HIGH)   # the default boot


PPO_BOOT = _clip_handle(0.2, 0.2)                          # SLIME_EPS_CLIP=0.2 SLIME_EPS_CLIP_HIGH=0.2


def test_cross_entropy_never_consults_the_clip():
    _check_miles_clip_config(_clip_handle(0.2, 0.28), "cross_entropy", None)


def test_unclipped_boot_serves_importance_sampling():
    _check_miles_clip_config(UNCLIPPED, "importance_sampling", None)


def test_unclipped_boot_refuses_ppo_even_without_config():
    with pytest.raises(UnsupportedFeatureError, match="ppo clip thresholds \\(0.8, 1.2\\)"):
        _check_miles_clip_config(UNCLIPPED, "ppo", None)


def test_ppo_boot_refuses_importance_sampling():
    with pytest.raises(UnsupportedFeatureError, match="clipped actor group"):
        _check_miles_clip_config(PPO_BOOT, "importance_sampling", None)


def test_ppo_boot_serves_ppo_at_the_boot_range_only():
    _check_miles_clip_config(PPO_BOOT, "ppo", None)                                   # defaults 0.8 / 1.2
    _check_miles_clip_config(PPO_BOOT, "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
    with pytest.raises(UnsupportedFeatureError, match="set SLIME_EPS_CLIP=0.2 SLIME_EPS_CLIP_HIGH=0.3"):
        _check_miles_clip_config(PPO_BOOT, "ppo", {"clip_high_threshold": 1.3})


def test_legacy_default_boot_is_refused_for_importance_sampling():
    # The pre-fix boot (0.2 / 0.28) must no longer pass silently.
    with pytest.raises(UnsupportedFeatureError, match="\\(0.8, 1.28\\)"):
        _check_miles_clip_config(_clip_handle(0.2, 0.28), "importance_sampling", None)


def _resume_handle():
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
    h = _resume_handle()
    asyncio.run(MilesBackend().load_checkpoint(h, root, optimizer=optimizer))
    h.train_group.load_checkpoint.assert_awaited_once_with(str(native), load_optimizer=optimizer)
    assert h.created_from_checkpoint is True


def test_missing_native_dir_is_an_error_not_a_guess(tmp_path):
    root = tmp_path / "ckpt" / "resume"
    root.mkdir(parents=True)
    model_setup.record_native_checkpoint(str(root), str(tmp_path / "gone"))
    with pytest.raises(Exception, match="is gone"):
        asyncio.run(MilesBackend().load_checkpoint(_resume_handle(), root))


def test_root_without_record_is_not_miles_native(tmp_path):
    root = tmp_path / "ckpt" / "foreign"
    root.mkdir(parents=True)
    with pytest.raises(Exception, match="not a Miles-native checkpoint"):
        asyncio.run(MilesBackend().load_checkpoint(_resume_handle(), root))


def test_model_input_lens_reads_the_router_datum():
    # The service hands backends the router's pydantic datums; the boundary
    # validation guarantees model_input, so there is no malformed case here.
    from tinkercloud.training.backends.miles.backend import _model_input_lens
    from tinkercloud.training.models.requests import ForwardBackwardDatum
    d = ForwardBackwardDatum.model_validate(
        {"model_input": {"chunks": [{"tokens": [1, 2, 3]}]}, "loss_fn_inputs": {}})
    assert _model_input_lens([d]) == [3]
