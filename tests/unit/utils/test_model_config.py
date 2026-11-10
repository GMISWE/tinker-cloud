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


def _hub(monkeypatch, *, offline, cache=None):
    """huggingface_hub in the given mode (the library reads HF_HUB_OFFLINE once, at import), on `cache`."""
    import huggingface_hub as hub
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    if cache is not None:
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    return hub


def _cached(cache, repo, files):
    """`repo` in the hub cache layout under `cache`: refs/main naming one snapshot that holds `files`."""
    from huggingface_hub.file_download import repo_folder_name
    root = cache / repo_folder_name(repo_id=repo, repo_type="model")
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("0" * 40)
    _repo(root / "snapshots" / ("0" * 40), files)


def _refusing(monkeypatch, hub, error):
    """Every download from the hub raises `error`, built without its constructor (there is no HTTP response to give it)."""
    def download(repo, name):
        raise error.__new__(error)
    monkeypatch.setattr(hub, "hf_hub_download", download)


def _loads_as_sent(monkeypatch, backend):
    """The fake backend naming base_model as the repository it loads, as TrainingBackend's default does."""
    monkeypatch.setattr(backend, "model_repositories", lambda base_model: [base_model])


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


@pytest.mark.parametrize("doc", [
    {"image_processor": {"auto_map": {"AutoImageProcessor": "image_processing_x.XImageProcessor"}}},
    {"processors": [{"name": "audio"}, {"auto_map": {"AutoFeatureExtractor": "feature_x.XExtractor"}}]},
    {"a": {"b": [{"c": {"auto_map": {"AutoProcessor": "processing_x.XProcessor"}}}]}},
], ids=["nested_object", "inside_a_list", "deep"])
def test_an_auto_map_at_any_depth_declares_code(monkeypatch, tmp_path, doc):
    """A processor config can carry a component's own config, auto_map included, under a key of its
    own; a check of the top-level key alone passed such a repository."""
    from tinkercloud.training.utils import model_config
    repo = _repo(tmp_path, {"config.json": CLEAN_CONFIG, "processor_config.json": doc})
    _trusting(monkeypatch)
    with pytest.raises(ValueError, match="auto_map in processor_config.json"):
        model_config.check_remote_code(repo)


def _nested(depth):
    return '{"k":' * depth + '{"auto_map": {"AutoProcessor": "processing_x.XProcessor"}}' + "}" * depth


@pytest.mark.parametrize("text, refusal", [
    (_nested(500), "auto_map in processor_config.json"),
    (_nested(100_000), "processor_config.json is not readable JSON .RecursionError."),
    ('{"auto_map": ', "processor_config.json is not readable JSON .JSONDecodeError."),
], ids=["deep", "deeper_than_the_parser", "not_json"])
def test_an_auto_map_at_any_depth_is_found_and_an_unreadable_file_is_refused_by_name(monkeypatch, tmp_path, text, refusal):
    """500 levels parse and overflowed a recursive walk (a RecursionError, no refusal naming the
    file); a file the parser cannot read at all cannot pass either, and the refusal names it."""
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    repo = _repo(tmp_path, {"config.json": CLEAN_CONFIG})
    (tmp_path / "processor_config.json").write_text(text)
    with pytest.raises(ValueError, match=f"^'{repo}'.*{refusal}"):
        model_config.check_remote_code(repo)


def test_only_a_non_empty_auto_map_at_any_depth_declares_code(monkeypatch, tmp_path):
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    model_config.check_remote_code(_repo(tmp_path, {
        "config.json": {**CLEAN_CONFIG, "auto_map": {}},
        "processor_config.json": {"image_processor": {"auto_map": {}, "automap": {"AutoProcessor": "x.X"}},
                                  "note": "auto_map"}}))


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


def test_the_real_backends_name_the_directories_they_load(monkeypatch, tmp_path):
    """miles resolves an id to <HF_HOME>/<name> and its Megatron twin <name>_torch_dist; verl to
    <TINKERCLOUD_MODEL_ROOT>/<id>. With either method gone, the default checks the id as sent, a hub
    id here, and never opens these directories."""
    from tinkercloud.training.backends.miles.backend import MilesBackend
    from tinkercloud.training.backends.verl.backend import VerlBackend
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    miles_home, verl_root = tmp_path / "miles", tmp_path / "verl"
    _repo(miles_home / "m", {"config.json": CLEAN_CONFIG})
    megatron = _repo(miles_home / "m_torch_dist", {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER})
    monkeypatch.setenv("HF_HOME", str(miles_home))
    with pytest.raises(ValueError, match=f"'{megatron}' declares custom code"):
        asyncio.run(model_config.check_model_code(MilesBackend(), "org/m"))

    verl_dir = _repo(verl_root / "org" / "m", {"config.json": {**CLEAN_CONFIG, "auto_map": {"AutoConfig": "c.C"}}})
    monkeypatch.setenv("TINKERCLOUD_MODEL_ROOT", str(verl_root))
    with pytest.raises(ValueError, match=f"'{verl_dir}' declares custom code"):
        asyncio.run(model_config.check_model_code(VerlBackend(), "org/m"))


def test_a_hub_repository_is_checked_file_by_file(monkeypatch, tmp_path):
    hub = _hub(monkeypatch, offline=False)
    from huggingface_hub.utils import EntryNotFoundError
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    files = {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER}
    _repo(tmp_path, files)

    def download(repo, name):
        if name not in files:
            raise EntryNotFoundError.__new__(EntryNotFoundError)
        return str(tmp_path / name)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(ValueError, match="auto_map in tokenizer_config.json"):
        model_config.check_remote_code("org/model")


@pytest.mark.parametrize("error", ["RepositoryNotFoundError", "GatedRepoError", "HFValidationError"])
def test_a_repository_the_server_cannot_read_is_refused(monkeypatch, error):
    """Missing, gated or not an id at all: read as "declares nothing", such a repository passed the
    check, and a base-model sampler for it was answered 200. It is a ValueError, so a 400."""
    hub = _hub(monkeypatch, offline=False)
    from fastapi import HTTPException
    from huggingface_hub import utils
    from tinkercloud.training.backends.fake.backend import FakeBackend
    from tinkercloud.training.models.requests import ResolveSamplerTargetRequest
    from tinkercloud.training.routers.sampling import resolve_sampler_target
    from tinkercloud.training.services.sampling_service import SamplingService
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    _refusing(monkeypatch, hub, getattr(utils, error))   # the name is the test's parameter
    with pytest.raises(ValueError, match="'org/unreadable' names no model repository this server can read"):
        model_config.check_remote_code("org/unreadable")
    backend = FakeBackend({"base_models": ["org/unreadable"]})
    _loads_as_sent(monkeypatch, backend)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(resolve_sampler_target(ResolveSamplerTargetRequest(base_model="org/unreadable"), None, {}, None,
                                           SamplingService(backend=backend)))
    assert refused.value.status_code == 400 and "'org/unreadable' names no model repository" in refused.value.detail


def test_an_empty_id_is_a_repository_the_server_cannot_read(monkeypatch):
    """Path("") is the working directory, so "" was checked as that directory and passed; the
    loaders take "" for a hub id, which it is not."""
    _hub(monkeypatch, offline=False)
    from tinkercloud.training.utils import model_config
    _trusting(monkeypatch)
    with pytest.raises(ValueError, match="'' names no model repository this server can read"):
        model_config.check_remote_code("")


def test_offline_a_hub_id_is_checked_through_its_cached_snapshot(monkeypatch, tmp_path):
    """Offline, the per-file download of a file the cached snapshot lacks (a text model's processor
    config) raised the same error as an outage, so a hub id whose snapshot lacks any of the files
    was refused. The cached snapshot is checked as a directory instead."""
    _hub(monkeypatch, offline=True, cache=tmp_path)
    from tinkercloud.training.utils import model_config
    _cached(tmp_path, "org/clean", {"config.json": CLEAN_CONFIG, "tokenizer_config.json": {"model_max_length": 4096}})
    _cached(tmp_path, "org/custom", {"config.json": CLEAN_CONFIG, "tokenizer_config.json": CUSTOM_TOKENIZER})
    _trusting(monkeypatch)
    model_config.check_remote_code("org/clean")
    with pytest.raises(ValueError, match="'org/custom' declares custom code .auto_map in tokenizer_config.json"):
        model_config.check_remote_code("org/custom")
    with pytest.raises(ValueError, match="'org/absent' is not in this server's hub cache"):
        model_config.check_remote_code("org/absent")


def test_a_hub_outage_fails_the_check(monkeypatch, tmp_path):
    """Unreachable and not cached, a file raises LocalEntryNotFoundError, an EntryNotFoundError too:
    nothing is known about the file, so the check fails rather than skip it as missing."""
    hub = _hub(monkeypatch, offline=False)
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
    _loads_as_sent(monkeypatch, backend)

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
    _loads_as_sent(monkeypatch, backend)
    service = SamplingService(backend=backend)
    request = ResolveSamplerTargetRequest(base_model=repo)
    _trusting(monkeypatch)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(resolve_sampler_target(request, None, {}, None, service))
    assert refused.value.status_code == 400 and "auto_map in tokenizer_config.json" in refused.value.detail
    _trusting(monkeypatch, repo)
    assert asyncio.run(resolve_sampler_target(request, None, {}, None, service)).base_model == repo


def test_no_server_loader_trusts_remote_code_unconditionally():
    """Every .py file under training/, in the keyword form and the dict form a kwargs literal takes."""
    import re
    from pathlib import Path
    from tinkercloud.training.utils import model_config
    unconditional = re.compile(r"""trust_remote_code["']?\s*[=:]\s*True""")
    assert unconditional.search("from_pretrained(path, trust_remote_code=True)")
    assert unconditional.search('kwargs = {"trust_remote_code": True}')
    assert not unconditional.search("trust_remote_code=remote_code_trusted(path)")
    training = Path(model_config.__file__).resolve().parents[1]
    offenders = [str(p.relative_to(training)) for p in training.rglob("*.py") if unconditional.search(p.read_text())]
    assert offenders == []
