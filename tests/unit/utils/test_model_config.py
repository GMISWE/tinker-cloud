"""utils.model_config: the GPU count comes from NUM_GPUS, Ray or nvidia-smi, never from a guess;
a model repository's code runs only for models the operator trusts."""
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


def test_no_model_runs_repository_code_unless_the_operator_lists_it(monkeypatch):
    from tinkercloud.training.utils import model_config
    monkeypatch.delenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", raising=False)
    assert not model_config.remote_code_trusted("org/model")
    monkeypatch.setenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", " org/model , /data/other ")
    assert model_config.remote_code_trusted("org/model")
    assert model_config.remote_code_trusted("/data/other")
    assert not model_config.remote_code_trusted("org/model-2")


def test_a_config_declaring_custom_code_is_refused_unless_the_model_is_trusted(monkeypatch, tmp_path):
    import json
    from tinkercloud.training.utils import model_config
    monkeypatch.delenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "evil", "auto_map": {"AutoConfig": "m.C"}}))
    with pytest.raises(ValueError, match="TINKERCLOUD_TRUST_REMOTE_CODE_MODELS"):
        model_config.read_raw_hf_config(str(tmp_path))
    monkeypatch.setenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", str(tmp_path))
    assert model_config.read_raw_hf_config(str(tmp_path))["auto_map"] == {"AutoConfig": "m.C"}


def test_a_config_without_custom_code_reads_as_written(monkeypatch, tmp_path):
    import json
    from tinkercloud.training.utils import model_config
    monkeypatch.delenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2", "max_position_embeddings": 4096}))
    assert model_config.read_raw_hf_config(str(tmp_path)) == {"model_type": "qwen2", "max_position_embeddings": 4096}


def test_no_server_loader_trusts_remote_code_unconditionally():
    from pathlib import Path
    from tinkercloud.training.utils import model_config
    training = Path(model_config.__file__).resolve().parents[1]
    offenders = [str(p.relative_to(training)) for p in training.rglob("*.py")
                 if not p.name.startswith("._") and "trust_remote_code=True" in p.read_text()]
    assert offenders == []
