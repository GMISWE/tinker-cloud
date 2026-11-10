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


# A record is replaced whole or not at all: it is how a checkpoint is found again.

@pytest.mark.parametrize("save, load", [
    (lambda s, d: s.save_checkpoint("m", "c", d), lambda s: s.load_checkpoint("m", "c")),
    (lambda s, d: s.save_training_run("m", d), lambda s: s.load_training_run("m")),
])
def test_a_write_that_fails_keeps_the_record_it_replaces(storage, save, load):
    save(storage, {"ttl": 60})
    with pytest.raises(TypeError):
        save(storage, {"ttl": object()})   # not JSON
    assert load(storage)["ttl"] == 60


def test_a_failed_rename_leaves_no_temp_file_and_the_old_record(storage, monkeypatch, tmp_path):
    storage.save_checkpoint("m", "c", {"ttl": 60})

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("tinkercloud.training.storage.metadata.os.replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        storage.save_checkpoint("m", "c", {"ttl": 1})
    monkeypatch.undo()
    assert storage.load_checkpoint("m", "c")["ttl"] == 60
    assert sorted(p.name for p in (tmp_path / "meta" / "checkpoints" / "m").iterdir()) == ["c.json"]


def test_a_saved_record_leaves_only_its_own_file(storage, tmp_path):
    storage.save_checkpoint("m", "c", {"ttl": 60})
    storage.save_checkpoint("m", "c", {"ttl": 30})
    assert sorted(p.name for p in (tmp_path / "meta" / "checkpoints" / "m").iterdir()) == ["c.json"]
    assert [r["ttl"] for r in storage.list_checkpoints("m")] == [30]


def test_a_temp_file_left_by_a_killed_write_is_swept_at_boot(storage, tmp_path):
    storage.save_training_run("model_x_0123456789abcdef", {"base_model": "b"})
    storage.save_checkpoint("m", "keep", {"v": 1})
    runs, ckpts = tmp_path / "meta" / "training_runs", tmp_path / "meta" / "checkpoints"
    (runs / ".deadbeef.tmp").write_text("{")
    (ckpts / "m" / ".cafe.tmp").write_text("{")
    (ckpts / "gone").mkdir()
    (ckpts / "gone" / ".f00d.tmp").write_text("{")   # the only thing in its model directory
    assert storage.sweep_temp_files() == 3
    assert sorted(p.name for p in runs.iterdir()) == ["model_x_0123456789abcdef.json"]
    assert sorted(p.name for p in ckpts.iterdir()) == ["m"]   # "gone" can be listed no more
    assert storage.load_checkpoint("m", "keep")["v"] == 1
    assert storage.sweep_temp_files() == 0


def test_a_record_whose_name_nearly_fills_the_filesystem_limit_can_still_be_written(storage):
    # the temp file must not be longer than the record's own name, or a name that fit before cannot be saved
    name = "c" * (255 - len(".json"))
    storage.save_checkpoint("m", name, {"ttl": 60})
    storage.save_checkpoint("m", name, {"ttl": 30})
    assert storage.load_checkpoint("m", name)["ttl"] == 30


def test_after_the_boot_sweep_a_model_that_was_listed_only_for_a_leftover_is_gone(storage, tmp_path):
    d = tmp_path / "meta" / "checkpoints" / "m"
    d.mkdir(parents=True)
    (d / ".beef.tmp").write_text("{")
    assert storage.list_checkpoints("m") == [] and d.exists()   # the directory exists only for the leftover
    storage.sweep_temp_files()
    assert not d.exists()
