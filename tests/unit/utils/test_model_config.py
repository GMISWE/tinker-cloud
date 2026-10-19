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
