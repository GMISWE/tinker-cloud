import pytest

from tinkercloud.training.checkpoints import CheckpointStore
from tinkercloud.training.storage.metadata import MetadataStorage


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "ckpt", MetadataStorage(tmp_path / "meta"))
