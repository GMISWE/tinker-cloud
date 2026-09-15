"""Sites that used to swallow a failure or invent a value now raise. Each test
pins the raise so a fallback cannot creep back in."""
import subprocess

import pytest

from tinkercloud.training.backends.base import BackendError


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


def test_malformed_supported_models_is_a_startup_error(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    monkeypatch.setenv("SUPPORTED_MODELS", "{not json")
    with pytest.raises(ValueError, match="SUPPORTED_MODELS"):
        TrainingConfig._get_default_models()
    monkeypatch.setenv("SUPPORTED_MODELS", '[{"model_name": "m", "max_context_length": 8}]')
    assert TrainingConfig._get_default_models()[0].model_name == "m"


def test_model_input_lens_reads_the_router_datum():
    # The service hands backends the router's pydantic datums; the boundary
    # validation guarantees model_input, so there is no malformed case here.
    from tinkercloud.training.backends.miles.backend import _model_input_lens
    from tinkercloud.training.models.requests import ForwardBackwardDatum
    d = ForwardBackwardDatum.model_validate(
        {"model_input": {"chunks": [{"tokens": [1, 2, 3]}]}, "loss_fn_inputs": {}})
    assert _model_input_lens([d]) == [3]


def test_learning_rate_rpc_failure_is_a_backend_error():
    from tinkercloud.training.backends.nemo_rl.backend import _set_learning_rate

    class _WorkerGroup:
        def run_all_workers_single_data(self, *a, **k):
            raise RuntimeError("actor died")

    class _Policy:
        worker_group = _WorkerGroup()

    with pytest.raises(BackendError, match="could not set learning rate 0.5"):
        _set_learning_rate(_Policy(), 0.5)


def test_miles_parallel_config_keys_are_required():
    from test_miles_pack_length import MODEL_CONFIG, _load_builder_module
    from argparse import Namespace
    b = _load_builder_module().MilesArgumentBuilder(default_save_dir="/tmp/ckpt")
    with pytest.raises(KeyError, match="tensor_parallel_size"):
        b._configure_model_args(
            Namespace(), base_model="/tmp/model", megatron_checkpoint_path="/tmp/mcore",
            lora_config={"rank": 8}, debug_train_only=False, load_dir=None,
            model_config=MODEL_CONFIG, parallel_config={"num_gpus": 4}, save_dir=None,
        )
