"""utils.model_config: the GPU count comes from NUM_GPUS, Ray or nvidia-smi, never from a guess;
a model repository's code runs only for models the operator trusts, checked before a backend sees one."""
import asyncio
import json
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


CLEAN_CONFIG = {"model_type": "qwen2", "max_position_embeddings": 4096}
CUSTOM_TOKENIZER = {"auto_map": {"AutoTokenizer": ["tokenization_x.XTokenizer", None]}}


def _trusting(monkeypatch, *models):
    """The process config, trusting `models`; remote_code_trusted reads it, not the environment."""
    from tinkercloud.training import config
    monkeypatch.setattr(config, "_config", config.TrainingConfig(trust_remote_code_models=list(models)))


def _repo(tmp_path, files):
    """A model directory holding `files` (name -> JSON document)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, doc in files.items():
        (tmp_path / name).write_text(json.dumps(doc))
    return str(tmp_path)


def test_no_model_runs_repository_code_unless_the_operator_lists_it(monkeypatch):
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    assert not model_config.remote_code_trusted("org/model")
    _trusting(monkeypatch, "org/model", "/data/other")
    monkeypatch.setenv("TINKERCLOUD_TRUST_REMOTE_CODE_MODELS", "org/model-2")   # read at startup, not per call
    assert model_config.remote_code_trusted("org/model") and model_config.remote_code_trusted("/data/other")
    assert not model_config.remote_code_trusted("org/model-2")


@pytest.mark.parametrize("name, doc", [
    ("config.json", {**CLEAN_CONFIG, "auto_map": {"AutoConfig": "configuration_x.XConfig"}}),
    ("tokenizer_config.json", CUSTOM_TOKENIZER),   # beside a clean config.json
    ("processor_config.json", {"auto_map": {"AutoProcessor": "processing_x.XProcessor"}}),
    ("preprocessor_config.json", {"auto_map": {"AutoImageProcessor": "image_processing_x.XImageProcessor"}}),
    ("video_preprocessor_config.json", {"auto_map": {"AutoVideoProcessor": "video_processing_x.XVideoProcessor"}}),
], ids=["config", "tokenizer", "processor", "preprocessor", "video_preprocessor"])
def test_a_repository_declaring_custom_code_anywhere_is_refused_unless_trusted(monkeypatch, tmp_path, name, doc):
    from tinkercloud.training.utils import model_config
    repo = _repo(tmp_path, {"config.json": CLEAN_CONFIG, name: doc})
    _trusting(monkeypatch)
    with pytest.raises(ValueError, match=f"auto_map in {name}.*TINKERCLOUD_TRUST_REMOTE_CODE_MODELS"):
        model_config.check_remote_code(repo)
    _trusting(monkeypatch, repo)
    model_config.check_remote_code(repo)


def test_a_repository_without_custom_code_passes(monkeypatch, tmp_path):
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    model_config.check_remote_code(_repo(tmp_path, {"config.json": CLEAN_CONFIG, "tokenizer_config.json": {"model_max_length": 4096}}))


def test_a_repository_is_checked_exactly_as_the_backend_loads_it(monkeypatch, tmp_path):
    """A clean m beside an m_torch_dist with a custom tokenizer: a backend that loads base_model as
    sent (NeMo RL) gets m_torch_dist checked, not m; a backend that resolves base_model elsewhere
    (miles, verl) gets what it names checked."""
    from tinkercloud.training.backends.base import TrainingBackend
    from tinkercloud.training.backends.fake.backend import FakeBackend
    from tinkercloud.training.utils import model_config
    clean = _repo(tmp_path / "m", {"config.json": CLEAN_CONFIG})
    custom = _repo(tmp_path / "m_torch_dist", {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER})
    _trusting(monkeypatch, clean)
    backend = FakeBackend()
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: TrainingBackend.model_repositories(backend, base_model))
    with pytest.raises(ValueError, match=f"'{custom}' declares custom code"):
        asyncio.run(model_config.check_model_code(backend, custom))
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: [base_model.replace("_torch_dist", "")])
    asyncio.run(model_config.check_model_code(backend, custom))
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: [clean, custom])
    with pytest.raises(ValueError, match="auto_map in tokenizer_config.json"):
        asyncio.run(model_config.check_model_code(backend, "org/model"))
    asyncio.run(model_config.check_model_code(FakeBackend(), custom))   # the fake loads no model


def test_a_hub_repository_is_checked_file_by_file(monkeypatch, tmp_path):
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    files = {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER}
    _repo(tmp_path, files)

    def download(repo, name):
        if repo != "org/model":
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)   # its constructor wants an HTTP response
        if name not in files:
            raise EntryNotFoundError.__new__(EntryNotFoundError)
        return str(tmp_path / name)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(ValueError, match="auto_map in tokenizer_config.json"):
        model_config.check_remote_code("org/model")
    model_config.check_remote_code("org/missing")   # no such repository: nothing of it can load


def test_a_hub_outage_fails_the_check(monkeypatch, tmp_path):
    """Unreachable and not cached, a file raises LocalEntryNotFoundError, an EntryNotFoundError too:
    nothing is known about the file, so the check fails rather than skip it as missing."""
    hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub.utils import LocalEntryNotFoundError
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    _repo(tmp_path, {"config.json": CLEAN_CONFIG})

    def download(repo, name):
        if name != "config.json":
            raise LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
        return str(tmp_path / name)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(LocalEntryNotFoundError):
        model_config.check_remote_code("org/model")


def test_a_create_is_checked_before_the_backend_sees_the_model(monkeypatch, tmp_path):
    from tinkercloud.training.backends.fake.backend import FakeBackend
    from tinkercloud.training.models.requests import OptimizerConfig
    from tinkercloud.training.services.model_service import ModelService
    repo = _repo(tmp_path, {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER})
    _trusting(monkeypatch)
    backend = FakeBackend()
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: [base_model])   # as every backend but the fake

    async def dispatched(**kwargs):
        raise AssertionError("the backend saw an untrusted model")
    monkeypatch.setattr(backend, "create_model", dispatched)
    service = ModelService(backend=backend, store=None, sample_futures=None)
    with pytest.raises(ValueError, match="auto_map in tokenizer_config.json"):
        asyncio.run(service.create_model(model_id="m", request_id="r", base_model=repo, lora_config=None,
                                         parallelism_config=None, metadata_storage=None, training_clients={},
                                         training_runs_metadata={}, optimizer_config=OptimizerConfig()))


def test_a_base_model_sampler_is_checked_before_its_engine_starts(monkeypatch, tmp_path):
    from fastapi import HTTPException
    from tinkercloud.training.backends.fake.backend import FakeBackend
    from tinkercloud.training.models.requests import ResolveSamplerTargetRequest
    from tinkercloud.training.routers.sampling import resolve_sampler_target
    from tinkercloud.training.services.sampling_service import SamplingService
    repo = _repo(tmp_path, {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER})
    backend = FakeBackend({"base_models": [repo]})
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: [base_model])
    service = SamplingService(backend=backend)
    request = ResolveSamplerTargetRequest(base_model=repo)
    _trusting(monkeypatch)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(resolve_sampler_target(request, None, {}, None, service))
    assert refused.value.status_code == 400 and "auto_map in tokenizer_config.json" in refused.value.detail
    _trusting(monkeypatch, repo)
    assert asyncio.run(resolve_sampler_target(request, None, {}, None, service)).base_model == repo


def test_no_server_loader_trusts_remote_code_unconditionally():
    from pathlib import Path
    from tinkercloud.training.utils import model_config
    training = Path(model_config.__file__).resolve().parents[1]
    offenders = [str(p.relative_to(training)) for p in training.rglob("*.py")
                 if not p.name.startswith("._") and "trust_remote_code=True" in p.read_text()]
    assert offenders == []
