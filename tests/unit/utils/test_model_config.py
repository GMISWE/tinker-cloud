"""utils.model_config: the GPU count comes from NUM_GPUS, Ray or nvidia-smi, never from a guess."""
import subprocess

import pytest


def test_gpu_count_with_no_source_raises(monkeypatch):
    from tinkercloud.training.utils import model_config
    monkeypatch.delenv("NUM_GPUS", raising=False)
    monkeypatch.delenv("SLIME_NUM_GPUS", raising=False)
    import ray
    monkeypatch.setattr(ray, "is_initialized", lambda: False)

    def no_binary(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", no_binary)
    with pytest.raises(RuntimeError, match="set NUM_GPUS"):
        model_config.detect_num_gpus()


def test_gpu_count_env_still_wins(monkeypatch):
    from tinkercloud.training.utils import model_config
    monkeypatch.setenv("NUM_GPUS", "4")
    assert model_config.detect_num_gpus() == 4


def test_declared_mtp_layers_reads_text_config_and_defaults_to_none():
    from tinkercloud.training.utils.model_config import declared_mtp_layers
    # Qwen3.5: composite config, MTP under text_config
    assert declared_mtp_layers({"text_config": {"mtp_num_hidden_layers": 1}}) == 1
    # DeepSeek-style top-level key wins over the nested one
    assert declared_mtp_layers({"num_nextn_predict_layers": 2, "text_config": {"mtp_num_hidden_layers": 1}}) == 2
    assert declared_mtp_layers({"num_hidden_layers": 28}) == 0
    with pytest.raises(ValueError, match="not a layer count"):
        declared_mtp_layers({"mtp_num_hidden_layers": "1"})


@pytest.mark.parametrize("value", [True, False, -1, 1.0])
def test_declared_mtp_layers_refuses_a_value_that_is_not_a_count(value):
    from tinkercloud.training.utils.model_config import declared_mtp_layers
    with pytest.raises(ValueError, match="not a layer count"):
        declared_mtp_layers({"text_config": {"mtp_num_hidden_layers": value}})
