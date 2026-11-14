"""The verl config the backend composes for a model: the values a Tinker client assumes and never sends
are set by the backend, so a moved upstream default cannot change a run."""
import pytest

pytest.importorskip("verl", reason="composes verl's packaged config")
from omegaconf import OmegaConf

from tinkercloud.training.backends.verl.backend import (
    PINNED_ACTOR_CONFIG, _compose_actor_rollout_config, actor_rollout_overrides,
)
from tinkercloud.training.backends.verl.builder import VerlArgumentBuilder


@pytest.fixture(scope="module")
def cfg():
    return VerlArgumentBuilder().build_args("org/model", num_gpus=4, lora_config={"rank": 0})


@pytest.fixture(scope="module")
def composed(cfg):
    return _compose_actor_rollout_config(cfg)


@pytest.mark.parametrize("key", sorted(PINNED_ACTOR_CONFIG))
def test_the_backend_sets_the_value_itself(cfg, composed, key, monkeypatch):
    value = PINNED_ACTOR_CONFIG[key]
    assert f"actor_rollout_ref.actor.{key}={value}" in actor_rollout_overrides(cfg)
    assert OmegaConf.select(composed, f"actor_rollout_ref.actor.{key}", throw_on_missing=True) == value
    # The pin decides, not a default that happens to agree: a pin of another value is what the config holds.
    other = value + type(value)(1)   # "AdamW1", 1, 1.0: another value of the pin's own type
    monkeypatch.setitem(PINNED_ACTOR_CONFIG, key, other)
    moved = OmegaConf.select(_compose_actor_rollout_config(cfg), f"actor_rollout_ref.actor.{key}", throw_on_missing=True)
    assert moved == other


def test_the_sampler_runs_with_cuda_graphs(cfg):
    """The builder set eager mode for the sampler, behind an override that nothing passed. Guards
    against the setting's return, not against verl's default: that is False too."""
    assert "actor_rollout_ref.rollout.enforce_eager=False" in actor_rollout_overrides(cfg)
