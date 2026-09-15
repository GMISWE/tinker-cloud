"""Per-backend typed config: env -> overrides -> effective values, logged with sources."""
import pytest
from pydantic import ValidationError

from tinkercloud.training.backends.miles.config import NO_CLIP_EPS, MilesConfig
from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig


def test_defaults_when_nothing_is_set():
    cfg = MilesConfig.from_env(environ={})
    assert (cfg.multilora_slots, cfg.eps_clip, cfg.dyn_batch, cfg.cobatch_e0_tokens, cfg.tp) == (0, NO_CLIP_EPS, True, None, None)
    assert cfg.source_of("eps_clip") == "default"


def test_env_values_are_typed_and_empty_means_unset():
    env = {"TINKERCLOUD_MILES_MULTILORA_SLOTS": "4", "SLIME_EPS_CLIP": "0.15", "SLIME_DYN_BATCH": "0",
           "SLIME_NO_OFFLOAD": "1", "TINKERCLOUD_MILES_COBATCH_E0_TOKENS": "", "SLIME_TP": "2"}
    cfg = MilesConfig.from_env(environ=env)
    assert (cfg.multilora_slots, cfg.eps_clip, cfg.dyn_batch, cfg.no_offload, cfg.cobatch_e0_tokens, cfg.tp) == (4, 0.15, False, True, None, 2)
    assert cfg.source_of("eps_clip") == "env SLIME_EPS_CLIP"
    assert cfg.source_of("cobatch_e0_tokens") == "default"


def test_overrides_win_over_env_and_unknown_keys_are_rejected():
    cfg = MilesConfig.from_env({"eps_clip": 0.3}, environ={"SLIME_EPS_CLIP": "0.15"})
    assert cfg.eps_clip == 0.3 and cfg.source_of("eps_clip") == "override"
    with pytest.raises(ValidationError):
        MilesConfig.from_env({"eps_clipp": 0.3}, environ={})
    with pytest.raises(ValidationError):
        MilesConfig.from_env(environ={"SLIME_TP": "two"})


def test_describe_lists_every_field_with_source():
    text = MilesConfig.from_env({"train_gpus": 2}, environ={"SLIME_WEIGHT_DECAY": "0.01"}).describe()
    assert "train_gpus=2 (override)" in text
    assert "weight_decay=0.01 (env SLIME_WEIGHT_DECAY)" in text
    assert f"eps_clip={NO_CLIP_EPS!r} (default)" in text
    assert text.count("\n") + 1 == len(MilesConfig.model_fields)


def test_every_env_var_maps_to_a_field():
    for cls in (MilesConfig, NemoRLConfig):
        assert set(cls.ENV) == set(cls.model_fields), cls.__name__


def test_nemo_rl_splits_known_keys_from_raw_config_overrides():
    cfg, raw = NemoRLConfig.split_overrides({"train_mbs": 4, "policy": {"train_global_batch_size": 64}})
    assert cfg.train_mbs == 4 and cfg.source_of("train_mbs") == "override"
    assert raw == {"policy": {"train_global_batch_size": 64}}
    cfg, raw = NemoRLConfig.split_overrides(None)
    assert raw == {} and cfg.refit_buffer_memory_ratio == 0.3
