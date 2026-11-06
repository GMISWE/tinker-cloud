"""adapter.py's per-family optimizer state files: the dimuon
file holds `<PEFT key>.momentum`, the family is read from the file name, and a
directory holding another family's file is refused."""
import pytest
import torch
from safetensors.torch import load_file, save_file

from tinkercloud.training.backends.automodel import adapter as am

A = "model.layers.0.self_attn.q_proj.lora_A.weight"


def test_dimuon_state_round_trips_keyed_as_the_adapter(tmp_path):
    key = am.peft_key(A)
    state = {key: {"momentum": torch.full((4, 8), 0.5)}}
    am.write_optimizer_state("dimuon", state, tmp_path / "optimizer")
    assert sorted(p.name for p in (tmp_path / "optimizer").iterdir()) == [am.OPTIMIZER_FILES["dimuon"]]
    flat = load_file(str(tmp_path / "optimizer" / am.OPTIMIZER_FILES["dimuon"]))
    assert set(flat) == {f"{key}.momentum"}
    family, back = am.read_optimizer_state(tmp_path / "optimizer")
    assert family == "dimuon" and set(back) == {key} and torch.equal(back[key]["momentum"], state[key]["momentum"])
    # before the first step: a file with no tensors, read back as no state, still named for its family
    am.write_optimizer_state("dimuon", {}, tmp_path / "optimizer")
    assert am.read_optimizer_state(tmp_path / "optimizer") == ("dimuon", {})


def test_the_adamw_file_is_the_adamw_family(tmp_path):
    key = am.peft_key(A)
    state = {key: {"exp_avg": torch.zeros(2, 2), "exp_avg_sq": torch.ones(2, 2), "step": torch.tensor(1.0)}}
    am.write_optimizer_state("adamw", state, tmp_path / "optimizer")
    family, back = am.read_optimizer_state(tmp_path / "optimizer")
    assert family == "adamw" and set(back[key]) == set(am.STATE_KEYS["adamw"])


def test_a_dimuon_file_with_adamw_fields_is_refused(tmp_path):
    (tmp_path / "optimizer").mkdir()
    save_file({f"{am.peft_key(A)}.exp_avg": torch.zeros(1)}, str(tmp_path / "optimizer" / am.OPTIMIZER_FILES["dimuon"]))
    with pytest.raises(ValueError, match="is not <PEFT key>"):
        am.read_optimizer_state(tmp_path / "optimizer")


def test_zero_or_two_state_files_are_refused(tmp_path):
    (tmp_path / "optimizer").mkdir()
    with pytest.raises(ValueError, match="holds 0 optimizer state files"):
        am.optimizer_state_family(tmp_path / "optimizer")
    for name in am.OPTIMIZER_FILES.values():
        save_file({}, str(tmp_path / "optimizer" / name))
    with pytest.raises(ValueError, match="holds 2 optimizer state files"):
        am.optimizer_state_family(tmp_path / "optimizer")


def test_write_replaces_the_other_familys_file(tmp_path):
    am.write_optimizer_state("adamw", {}, tmp_path / "optimizer")
    am.write_optimizer_state("dimuon", {}, tmp_path / "optimizer")
    assert am.optimizer_state_family(tmp_path / "optimizer") == "dimuon"
