"""Where the verl backend loads a model from when the operator names a model root."""
import pytest

from tinkercloud.training.backends.verl.builder import VerlArgumentBuilder


def test_a_model_the_root_holds_resolves_to_its_directory(monkeypatch, tmp_path):
    """The root is the first place read: its copy wins over the hub id."""
    (tmp_path / "org" / "model").mkdir(parents=True)
    monkeypatch.setenv("TINKERCLOUD_MODEL_ROOT", str(tmp_path))
    assert VerlArgumentBuilder.resolve_hf_path("org/model") == str(tmp_path / "org" / "model")


def test_a_model_the_root_lacks_is_refused_by_name(monkeypatch, tmp_path):
    """It used to resolve to the hub id without a word, and a run then loaded another copy of the
    model than the one the operator had placed."""
    monkeypatch.setenv("TINKERCLOUD_MODEL_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="holds no directory org/model"):
        VerlArgumentBuilder.resolve_hf_path("org/model")


def test_without_a_root_a_hub_id_stays_itself(monkeypatch):
    """No root named: the name goes to the loader as sent."""
    monkeypatch.delenv("TINKERCLOUD_MODEL_ROOT", raising=False)
    assert VerlArgumentBuilder.resolve_hf_path("org/a-model-no-directory-holds") == "org/a-model-no-directory-holds"


def test_an_absolute_path_is_never_looked_up_under_the_root(monkeypatch, tmp_path):
    """A caller that names a directory means that directory: the root's refusal is for names."""
    monkeypatch.setenv("TINKERCLOUD_MODEL_ROOT", str(tmp_path))
    assert VerlArgumentBuilder.resolve_hf_path("/somewhere/else/model") == "/somewhere/else/model"
