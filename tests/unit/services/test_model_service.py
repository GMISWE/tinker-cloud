"""ModelService.create_model: a create that fails after the backend allocated the model undoes all of it."""
import asyncio
from types import SimpleNamespace

import pytest

from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.models.requests import OptimizerConfig
from tinkercloud.training.services.model_service import ModelService
from tinkercloud.training.storage.metadata import MetadataStorage


class _FullDisk(MetadataStorage):
    """A save that raises; a record write is whole or not at all, so nothing is on disk after it."""

    def save_training_run(self, training_run_id, metadata):
        raise OSError(28, "No space left on device")


class _Unreadable(MetadataStorage):
    """Every read of a record raises."""

    def _refuse(self, *args, **kwargs):
        raise PermissionError("metadata directory cannot be read")

    load_training_run = list_training_runs = load_checkpoint = list_checkpoints = _refuse


def _create(service, metadata_storage, clients, runs):
    return service.create_model(
        model_id="m", request_id="r", base_model="fake/tiny", lora_config=None, parallelism_config=None,
        metadata_storage=metadata_storage, training_clients=clients, training_runs_metadata=runs,
        optimizer_config=OptimizerConfig(),
    )


def _backend_serving(endpoints):
    """A FakeBackend whose models claim to be served from `endpoints`."""
    class Serving(FakeBackend):
        async def create_model(self, *args, **kwargs):
            handle = await super().create_model(*args, **kwargs)
            handle.inference_endpoints = endpoints
            return handle
    return Serving()


@pytest.fixture
def make_service(tmp_path, monkeypatch):
    monkeypatch.setenv("NUM_GPUS", "1")   # detect_num_gpus() answers from the env before it asks Ray or nvidia-smi
    store = SimpleNamespace(native_root=lambda _model_id: tmp_path / "native")
    return lambda backend=None: ModelService(backend=backend or FakeBackend(), store=store, sample_futures=None)


def test_a_create_that_works_registers_the_model(make_service, tmp_path):
    service, clients, runs = make_service(), {}, {}
    storage = MetadataStorage(tmp_path / "meta")
    asyncio.run(_create(service, storage, clients, runs))
    assert set(clients) == {"m"} and set(runs) == {"m"} and set(service.backend._models) == {"m"}
    assert storage.load_training_run("m") is not None


def test_metadata_that_cannot_be_saved_releases_the_backend_model(make_service, tmp_path, monkeypatch):
    service, clients, runs = make_service(), {}, {}
    storage = MetadataStorage(tmp_path / "meta")

    def full(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("tinkercloud.training.storage.metadata.os.replace", full)   # the write's last step
    with pytest.raises(OSError, match="No space left"):
        asyncio.run(_create(service, storage, clients, runs))
    monkeypatch.undo()
    assert service.backend._models == {}   # it used to stay allocated, in no registry the reaper or delete_model reads
    assert clients == {} and runs == {}
    assert storage.list_training_runs(limit=10) == []   # no partial record


def test_a_create_reads_no_training_run_record(make_service, tmp_path):
    """A read between the backend's create and the undo's `try` stranded the model when the read raised."""
    service, clients, runs = make_service(), {}, {}
    asyncio.run(_create(service, _Unreadable(tmp_path / "meta"), clients, runs))
    assert set(clients) == {"m"} and set(runs) == {"m"} and set(service.backend._models) == {"m"}


def test_a_route_that_is_refused_undoes_everything_the_create_recorded(make_service, tmp_path):
    """InferenceEndpoint raises ValueError for a URL that is not http(s); the route is checked before the record is saved."""
    service, clients, runs = make_service(_backend_serving(["engine:8000"])), {}, {}
    storage = MetadataStorage(tmp_path / "meta")
    with pytest.raises(ValueError):  # noqa: PT011  (#193)
        asyncio.run(_create(service, storage, clients, runs))
    assert service.backend._models == {}
    assert clients == {} and runs == {}
    assert storage.load_training_run("m") is None   # list_training_runs must not list a run whose create failed
    assert storage.list_training_runs(limit=10) == []


def test_a_failed_release_does_not_hide_the_create_error(make_service, tmp_path, monkeypatch):
    async def broken(self, handle):
        raise RuntimeError("engine gone")

    monkeypatch.setattr(FakeBackend, "delete_model", broken)
    with pytest.raises(OSError, match="No space left"):
        asyncio.run(_create(make_service(), _FullDisk(tmp_path / "meta"), {}, {}))
