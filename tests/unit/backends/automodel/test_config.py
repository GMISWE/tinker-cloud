"""AutomodelConfig: required knobs fail at startup, values arrive as strings."""
import pytest
from pydantic import ValidationError

from tinkercloud.training.backends.automodel.config import AutomodelConfig

ENV = {
    "AUTOMODEL_TRAIN_GPUS": "2",
    "AUTOMODEL_MAX_LORAS": "4",
    "AUTOMODEL_MAX_LORA_RANK": "32",
    "AUTOMODEL_MAX_CPU_LORAS": "8",
    "AUTOMODEL_MICRO_BATCH_ROWS": "2",
}


def test_from_env_parses_strings():
    cfg = AutomodelConfig.from_env(environ=ENV)
    assert (cfg.train_gpus, cfg.max_loras, cfg.max_lora_rank, cfg.max_cpu_loras, cfg.micro_batch_rows) == (2, 4, 32, 8, 2)
    assert cfg.source_of("max_lora_rank") == "env AUTOMODEL_MAX_LORA_RANK"


@pytest.mark.parametrize("var", sorted(ENV))
def test_every_knob_is_required(var):
    env = {k: v for k, v in ENV.items() if k != var}
    field = next(f for f, v in AutomodelConfig.ENV.items() if v == var)
    with pytest.raises(ValidationError, match=field):
        AutomodelConfig.from_env(environ=env)


def test_rank_vllm_cannot_serve_is_refused():
    with pytest.raises(ValidationError, match="max_lora_rank"):
        AutomodelConfig.from_env(environ={**ENV, "AUTOMODEL_MAX_LORA_RANK": "24"})


def test_cpu_cache_smaller_than_gpu_slots_is_refused():
    with pytest.raises(ValidationError, match="max_cpu_loras=2 must be >= max_loras=4"):
        AutomodelConfig.from_env(environ={**ENV, "AUTOMODEL_MAX_CPU_LORAS": "2"})


def test_override_wins_over_env():
    cfg = AutomodelConfig.from_env({"train_gpus": 3}, environ=ENV)
    assert cfg.train_gpus == 3 and cfg.source_of("train_gpus") == "override"
