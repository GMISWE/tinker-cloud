"""The PEFT export of a model with a per-slice module: one standard adapter of rank
k·r and alpha k·α (k = the most slices a module has), the sliced module's A stacked and
its B block-diagonal, every other module zero-padded; it applies the trainer's own
deltas, and reads back into the per-slice parameters."""
import json

import pytest
import torch
from safetensors.torch import load_file

from tinkercloud.training.backends.automodel import adapter as am
from tinkercloud.training.backends.automodel.lora import LoraSpec, spec_mismatch
from tinkercloud.training.checkpoints import interchange as ci

WIDTHS = (3, 3, 5)
IN = 8
LORA = LoraSpec(rank=2, alpha=4, target_modules=("q_proj", "in_proj_qkv", "in_proj_z"), dropout=0.0,
                optimizer="adamw", slices=(("in_proj_qkv", WIDTHS),))
QKV = "model.language_model.layers.0.linear_attn.in_proj_qkv"
Z = "model.language_model.layers.0.linear_attn.in_proj_z"
Q = "model.language_model.layers.3.self_attn.q_proj"


def _state() -> dict:
    torch.manual_seed(0)
    state = {f"{Z}.lora_A.weight": torch.randn(2, IN), f"{Z}.lora_B.weight": torch.randn(6, 2),
             f"{Q}.lora_A.weight": torch.randn(2, IN), f"{Q}.lora_B.weight": torch.randn(4, 2)}
    for i, w in enumerate(WIDTHS):
        state[f"{QKV}.lora_A.{i}.weight"] = torch.randn(2, IN)
        state[f"{QKV}.lora_B.{i}.weight"] = torch.randn(w, 2)
    return state


def _trainer_delta(state: dict, module: str) -> torch.Tensor:
    s = LORA.alpha / LORA.rank
    if f"{module}.lora_A.weight" in state:
        return s * state[f"{module}.lora_B.weight"] @ state[f"{module}.lora_A.weight"]
    return s * torch.cat([state[f"{module}.lora_B.{i}.weight"] @ state[f"{module}.lora_A.{i}.weight"]
                          for i in range(len(WIDTHS))])


def test_sliced_export_is_rank_kr_and_applies_the_trainers_deltas(tmp_path):
    state = _state()
    am.write_peft_adapter(state, LORA, "base/model", tmp_path / "hf_adapter")
    config = json.loads((tmp_path / "hf_adapter" / ci.ADAPTER_CONFIG_FILE).read_text())
    assert (config["r"], config["lora_alpha"]) == (6, 12)
    assert set(config["target_modules"]) == {"q_proj", "in_proj_qkv", "in_proj_z"}
    assert spec_mismatch(config, LORA) is None
    weights = load_file(str(tmp_path / "hf_adapter" / ci.ADAPTER_WEIGHTS_FILE))
    p = ci.PEFT_PREFIX
    assert set(weights) == {f"{p}{m}.lora_{f}.weight" for m in (QKV, Z, Q) for f in "AB"}

    a, b = weights[f"{p}{QKV}.lora_A.weight"], weights[f"{p}{QKV}.lora_B.weight"]
    assert a.shape == (6, IN) and b.shape == (sum(WIDTHS), 6)
    assert torch.equal(a, torch.cat([state[f"{QKV}.lora_A.{i}.weight"] for i in range(3)]))
    row = 0
    for i, w in enumerate(WIDTHS):   # block i: slice i's rows, rank columns 2i:2i+2; zero elsewhere
        block = torch.zeros(w, 6)
        block[:, 2 * i:2 * i + 2] = state[f"{QKV}.lora_B.{i}.weight"]
        assert torch.equal(b[row:row + w], block)
        row += w
    for m in (Z, Q):   # zero-padded from rank 2 to 6
        assert weights[f"{p}{m}.lora_A.weight"].shape == (6, IN)
        assert torch.count_nonzero(weights[f"{p}{m}.lora_A.weight"][2:]) == 0
        assert torch.count_nonzero(weights[f"{p}{m}.lora_B.weight"][:, 2:]) == 0

    s = config["lora_alpha"] / config["r"]
    for m in (QKV, Z, Q):
        exported = s * weights[f"{p}{m}.lora_B.weight"] @ weights[f"{p}{m}.lora_A.weight"]
        assert (exported - _trainer_delta(state, m)).abs().max().item() <= 1e-6, m


def test_sliced_export_reads_back_into_the_per_slice_parameters(tmp_path):
    state = _state()
    am.write_peft_adapter(state, LORA, "base/model", tmp_path / "hf_adapter")
    weights = load_file(str(tmp_path / "hf_adapter" / ci.ADAPTER_WEIGHTS_FILE))
    back = am.from_peft(weights, list(state), LORA)
    assert set(back) == {am.peft_key(n) for n in state}
    assert all(torch.equal(back[am.peft_key(n)], t) for n, t in state.items())


def test_an_adapter_the_slices_cannot_hold_is_refused(tmp_path):
    state = _state()
    am.write_peft_adapter(state, LORA, "base/model", tmp_path / "hf_adapter")
    weights = load_file(str(tmp_path / "hf_adapter" / ci.ADAPTER_WEIGHTS_FILE))
    p = ci.PEFT_PREFIX
    off_block = dict(weights)
    off_block[f"{p}{QKV}.lora_B.weight"] = off_block[f"{p}{QKV}.lora_B.weight"].clone()
    off_block[f"{p}{QKV}.lora_B.weight"][0, 5] = 1.0   # q's row, v's rank columns
    with pytest.raises(ValueError, match="in_proj_qkv"):
        am.from_peft(off_block, list(state), LORA)
    padded = dict(weights)
    padded[f"{p}{Z}.lora_A.weight"] = padded[f"{p}{Z}.lora_A.weight"].clone()
    padded[f"{p}{Z}.lora_A.weight"][4] = 1.0   # a padding row
    with pytest.raises(ValueError, match="in_proj_z"):
        am.from_peft(padded, list(state), LORA)


def test_an_adapter_without_slices_exports_rank_r(tmp_path):
    plain = LoraSpec(rank=2, alpha=4, target_modules=("q_proj",), dropout=0.0, optimizer="adamw")
    state = {f"{Q}.lora_A.weight": torch.randn(2, IN), f"{Q}.lora_B.weight": torch.randn(4, 2)}
    am.write_peft_adapter(state, plain, "base/model", tmp_path / "hf_adapter")
    config = json.loads((tmp_path / "hf_adapter" / ci.ADAPTER_CONFIG_FILE).read_text())
    assert (config["r"], config["lora_alpha"]) == (2, 4)
    weights = load_file(str(tmp_path / "hf_adapter" / ci.ADAPTER_WEIGHTS_FILE))
    assert all(torch.equal(weights[am.peft_key(n)], t) for n, t in state.items())
    assert {am.peft_key(n): t for n, t in state.items()}.keys() == am.from_peft(weights, list(state), plain).keys()
