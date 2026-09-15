"""Miles reads eps_clip / eps_clip_high once at actor boot and both RL losses
share the clamp. The guard makes the choice explicit: importance_sampling is
served only by an unclipped group, ppo only by a group booted with exactly the
call's range (Tinker defaults 0.8 / 1.2 when the call sends none)."""
from argparse import Namespace

import pytest

from tinkercloud.training.backends.base import UnsupportedFeatureError
from tinkercloud.training.backends.miles.backend import MilesHandle, _check_miles_clip_config
from tinkercloud.training.backends.miles.config import NO_CLIP_EPS, NO_CLIP_EPS_HIGH


def _handle(eps_clip, eps_clip_high):
    return MilesHandle(model_id="m", backend_type="miles",
                       args=Namespace(eps_clip=eps_clip, eps_clip_high=eps_clip_high))


UNCLIPPED = _handle(NO_CLIP_EPS, NO_CLIP_EPS_HIGH)   # the default boot
PPO_BOOT = _handle(0.2, 0.2)                          # SLIME_EPS_CLIP=0.2 SLIME_EPS_CLIP_HIGH=0.2


def test_cross_entropy_never_consults_the_clip():
    _check_miles_clip_config(_handle(0.2, 0.28), "cross_entropy", None)


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
        _check_miles_clip_config(_handle(0.2, 0.28), "importance_sampling", None)
