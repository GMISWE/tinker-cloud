"""adapter.py on the filesystem: the PEFT pair and the AdamW state file a
checkpoint root holds, their keys, and the swap-in that keeps a reader
from seeing a half-written directory."""
import json

import pytest
import torch
from safetensors.torch import load_file

from tinkercloud.training.backends.automodel import adapter as am
from tinkercloud.training.backends.automodel.lora import LoraSpec
from tinkercloud.training.checkpoints import interchange as ci

LORA = LoraSpec(rank=4, alpha=8, target_modules=("q_proj",), dropout=0.0)
A = "model.layers.0.self_attn.q_proj.lora_A.weight"
B = "model._orig_mod.layers.0.self_attn.q_proj.lora_B.weight"


def test_peft_adapter_is_the_interchange_layout(tmp_path):
    state = {A: torch.ones(4, 8), B: torch.zeros(8, 4)}
    am.write_peft_adapter(state, LORA, "base/model", tmp_path / "hf_adapter")
    assert ci.find_hf_adapter(str(tmp_path)) == str(tmp_path / "hf_adapter")
    weights = load_file(str(tmp_path / "hf_adapter" / ci.ADAPTER_WEIGHTS_FILE))
    assert set(weights) == {ci.PEFT_PREFIX + A, ci.PEFT_PREFIX + "model.layers.0.self_attn.q_proj.lora_B.weight"}
    config = json.loads((tmp_path / "hf_adapter" / ci.ADAPTER_CONFIG_FILE).read_text())
    assert (config["r"], config["lora_alpha"], config["target_modules"]) == (4, 8, ["q_proj"])
    assert not (tmp_path / "hf_adapter.tmp").exists()


def test_adam_state_round_trips_keyed_as_the_adapter(tmp_path):
    key = am.peft_key(A)
    state = {key: {"exp_avg": torch.full((4, 8), 0.5), "exp_avg_sq": torch.full((4, 8), 0.25), "step": torch.tensor(3.0)}}
    am.write_adam_state(state, tmp_path / "optimizer")
    flat = load_file(str(tmp_path / "optimizer" / am.OPTIMIZER_FILE))
    assert set(flat) == {f"{key}.exp_avg", f"{key}.exp_avg_sq", f"{key}.step"}
    back = am.read_adam_state(tmp_path / "optimizer")
    assert set(back) == {key} and all(torch.equal(back[key][f], state[key][f]) for f in am.ADAM_STATE_KEYS)
    # before the first step: a file with no tensors, read back as no state
    am.write_adam_state({}, tmp_path / "optimizer")
    assert am.read_adam_state(tmp_path / "optimizer") == {}


def test_adam_state_file_with_a_foreign_key_is_refused(tmp_path):
    from safetensors.torch import save_file
    (tmp_path / "optimizer").mkdir()
    save_file({"exp_avg": torch.zeros(1)}, str(tmp_path / "optimizer" / am.OPTIMIZER_FILE))
    with pytest.raises(ValueError, match="is not <PEFT key>"):
        am.read_adam_state(tmp_path / "optimizer")
    save_file({f"{am.peft_key(A)}.exp_avg": torch.zeros(1)}, str(tmp_path / "optimizer" / am.OPTIMIZER_FILE))
    with pytest.raises(ValueError, match="has fields"):
        am.read_adam_state(tmp_path / "optimizer")
