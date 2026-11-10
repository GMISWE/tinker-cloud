"""ModelService.create_model: a create that fails after the backend allocated the model undoes all of it."""
import asyncio
from types import SimpleNamespace

import pytest

from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.models.requests import OptimizerConfig
from tinkercloud.training.services.model_service import ModelService
from tinkercloud.training.storage.metadata import MetadataStorage


class _FullDisk(MetadataStorage):
    """The realistic disk-full: the file is created and cut short, then the write raises."""

    def save_training_run(self, training_run_id, metadata):
        path = self.training_runs_dir / f"{training_run_id}.json"
        path.write_text('{"training_run_id": "m", "base_')
        raise OSError(28, "No space left on device")


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


def test_metadata_that_cannot_be_saved_releases_the_backend_model(make_service, tmp_path):
    service, clients, runs = make_service(), {}, {}
    with pytest.raises(OSError, match="No space left"):
        asyncio.run(_create(service, _FullDisk(tmp_path / "meta"), clients, runs))
    assert service.backend._models == {}   # it used to stay allocated, in no registry the reaper or delete_model reads
    assert clients == {} and runs == {}


def test_a_route_that_is_refused_undoes_everything_the_create_recorded(make_service, tmp_path):
    """InferenceEndpoint raises ValueError for a URL that is not http(s): the step after the registries are filled."""
    service, clients, runs = make_service(_backend_serving(["engine:8000"])), {}, {}
    storage = MetadataStorage(tmp_path / "meta")
    with pytest.raises(ValueError):
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


def test_a_training_run_record_that_cannot_be_removed_does_not_hide_the_create_error(make_service, tmp_path, monkeypatch):
    def broken(self, training_run_id):
        raise PermissionError("read-only metadata")

    monkeypatch.setattr(MetadataStorage, "delete_training_run", broken)
    service = make_service(_backend_serving(["engine:8000"]))
    with pytest.raises(ValueError):
        asyncio.run(_create(service, MetadataStorage(tmp_path / "meta"), {}, {}))
    assert service.backend._models == {}   # the handle was still released


def _kept_record(storage):
    """The record an unloaded model keeps for resume (delete_model preserves it), with a checkpoint pointer in it."""
    storage.save_training_run("m", {"base_model": "fake/tiny", "last_checkpoint": "weights--c1", "marker": "kept"})
    return storage.load_training_run("m")


def test_a_failed_create_under_a_kept_id_restores_the_record_it_overwrote(make_service, tmp_path):
    """create-via-load reuses the id. The save overwrites the kept record, so the undo must put it back, not delete it."""
    storage = MetadataStorage(tmp_path / "meta")
    kept = _kept_record(storage)
    service = make_service(_backend_serving(["engine:8000"]))   # the route step fails after the save
    with pytest.raises(ValueError):
        asyncio.run(_create(service, storage, {}, {}))
    assert storage.load_training_run("m") == kept   # checkpoint listing and loads still find it


def test_a_create_that_fails_before_it_saves_leaves_a_kept_record_alone(make_service, tmp_path, monkeypatch):
    storage = MetadataStorage(tmp_path / "meta")
    kept = _kept_record(storage)

    def refuse(self):
        raise ValueError("unsupported optimizer record")

    monkeypatch.setattr(OptimizerConfig, "record", refuse)
    service = make_service()
    with pytest.raises(ValueError, match="unsupported optimizer record"):
        asyncio.run(_create(service, storage, {}, {}))
    assert service.backend._models == {}
    assert storage.load_training_run("m") == kept   # this create never wrote, so it must not delete


def test_a_cut_short_write_does_not_leave_a_partial_record_behind(make_service, tmp_path):
    storage = _FullDisk(tmp_path / "meta")
    with pytest.raises(OSError, match="No space left"):
        asyncio.run(_create(make_service(), storage, {}, {}))
    assert not (storage.training_runs_dir / "m.json").exists()
    assert storage.list_training_runs(limit=10) == []
