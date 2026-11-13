"""The verl backend's micro-batch: one sequence per GPU in a training pass and in a forward-only pass,
set by the backend, and the padding of a request that follows from it."""
import pytest

pytest.importorskip("verl", reason="composes verl's packaged config")
from omegaconf import OmegaConf

from tinkercloud.training.backends.verl import builder
from tinkercloud.training.backends.verl.backend import (
    VerlBackend, VerlHandle, _compose_actor_rollout_config, actor_rollout_overrides,
)
from tinkercloud.training.backends.verl.builder import VerlArgumentBuilder

from .datums import nk_datum

# constant of the backend -> the verl key it sets
KEYS = {
    "TRAIN_MICRO_BATCH_PER_GPU": "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "INFER_MICRO_BATCH_PER_GPU": "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
}


def _cfg():
    return VerlArgumentBuilder().build_args("org/model", num_gpus=4, lora_config={"rank": 0})


@pytest.mark.parametrize("constant", sorted(KEYS))
def test_the_micro_batch_is_one_sequence_set_by_the_backend(constant, monkeypatch):
    """The value must be 1 (`tests/gpu/test_verl_fit.py` runs the pass that needs it) and must reach
    verl through the override list, where a moved upstream default cannot replace it."""
    assert getattr(builder, constant) == 1
    cfg = _cfg()
    assert f"{KEYS[constant]}=1" in actor_rollout_overrides(cfg)
    assert OmegaConf.select(_compose_actor_rollout_config(cfg), KEYS[constant], throw_on_missing=True) == 1
    # the constant decides, not a default that agrees with it
    monkeypatch.setattr(builder, constant, 3)
    assert OmegaConf.select(_compose_actor_rollout_config(_cfg()), KEYS[constant], throw_on_missing=True) == 3


@pytest.mark.parametrize("infer", [False, True], ids=["train", "infer"])
@pytest.mark.parametrize("sent, padded", [(1, 4), (4, 4), (5, 8)])
def test_a_request_is_padded_to_the_data_parallel_size_only(sent, padded, infer):
    """The engine needs a whole micro-batch on every rank, so a request is padded with copies of its
    first datum. With 4 sequences per GPU one datum became 16 copies, and a client could not split a
    request to get under the memory limit."""
    engine = _cfg()["engine"]
    handle = VerlHandle(model_id="m", backend_type="verl", config={"engine": engine}, trainer_config=None, dp_size=4)
    data = [nk_datum([1, 2, 3], [2, 3, 4], [1.0, 1.0, 1.0]) for _ in range(sent)]
    td = VerlBackend()._to_tensordict(handle, data, "cross_entropy", infer)
    assert td.batch_size[0] == padded
