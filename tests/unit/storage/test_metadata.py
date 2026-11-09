"""MetadataStorage builds every path from ids that arrive in requests: each stays under its directory."""
import pytest

from tinkercloud.training.storage.metadata import MetadataStorage


@pytest.fixture
def storage(tmp_path):
    return MetadataStorage(tmp_path / "meta")


@pytest.mark.parametrize("call", [
    lambda s: s.load_training_run("../../etc/passwd"),
    lambda s: s.save_training_run("../outside", {}),
    lambda s: s.list_checkpoints(".."),
    lambda s: s.load_checkpoint("m", "../../x"),
    lambda s: s.save_checkpoint("../m", "c", {}),
    lambda s: s.delete_checkpoint("m/../..", "c"),
])
def test_an_id_that_leaves_its_directory_is_refused(storage, tmp_path, call):
    with pytest.raises(ValueError, match="leaves"):
        call(storage)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta"]   # nothing written beside it


def test_ids_that_stay_inside_still_work(storage):
    storage.save_training_run("model_x_0123456789abcdef", {"base_model": "b"})
    assert storage.load_training_run("model_x_0123456789abcdef")["base_model"] == "b"
    assert storage.load_training_run("..") is None   # "...json": a file name inside the directory
    storage.save_checkpoint("model_x_0123456789abcdef", "weights--c1", {})
    [rec] = storage.list_checkpoints("model_x_0123456789abcdef")
    assert rec["checkpoint_name"] == "weights--c1"
    assert storage.delete_checkpoint("model_x_0123456789abcdef", "weights--c1")
