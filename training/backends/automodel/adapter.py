"""
The HF PEFT adapter directory that vLLM and PEFT load (R4, R12): the LoRA
weights under PEFT's keys and the adapter_config.json that names rank, alpha
and target modules. Written by rank 0 of a training group after the gather
(worker.export_adapter); read by the base model's vLLM through
/v1/load_lora_adapter, so the directory must be on a filesystem both see.

A checkpoint adds the optimizer state of the same parameters beside it, one
file named for the tenant's family:
`optimizer/adamw.safetensors`: `<PEFT key>.<exp_avg|exp_avg_sq|step>`, or
`optimizer/dimuon.safetensors`: `<PEFT key>.momentum`; unsharded, keyed as
the adapter file is. Empty before the first optimizer step; the file still
exists, so a load with optimizer=true restores "no step taken" instead of
being refused (that refusal is for roots another backend wrote).
"""
import json
import shutil
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

from ...checkpoints.interchange import ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, PEFT_PREFIX
from .lora import LoraSpec

OPTIMIZER_DIRNAME = "optimizer"
# per family: the state file under OPTIMIZER_DIRNAME, every per-parameter
# state key, and the keys laid out as the parameter is (the rest are scalars)
OPTIMIZER_FILES = {"adamw": "adamw.safetensors", "dimuon": "dimuon.safetensors"}
STATE_KEYS = {"adamw": ("exp_avg", "exp_avg_sq", "step"), "dimuon": ("momentum",)}
SHARD_KEYS = {"adamw": ("exp_avg", "exp_avg_sq"), "dimuon": ("momentum",)}
OPTIMIZER_FILE = OPTIMIZER_FILES["adamw"]
ADAM_STATE_KEYS = STATE_KEYS["adamw"]

# Wrapper segments torch.compile and activation checkpointing put into module
# names; PEFT and vLLM address the module underneath.
_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.")


def peft_key(name: str) -> str:
    """`model.layers.0.self_attn.q_proj.lora_A.weight` ->
    `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`."""
    for segment in _WRAPPER_SEGMENTS:
        name = name.replace(segment, "")
    return PEFT_PREFIX + name


def _swap_in(dest: Path, write: Callable[[Path], None]) -> None:
    """Run `write` on a sibling directory, then rename it over `dest`, so a
    reader never sees a half-written directory."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def write_peft_adapter(state: Dict[str, torch.Tensor], lora: LoraSpec, base_model: str, dest: Path) -> None:
    """Write the adapter at `dest`, replacing what is there."""
    def write(tmp: Path) -> None:
        save_file({peft_key(n): t.contiguous() for n, t in state.items()},
                  str(tmp / ADAPTER_WEIGHTS_FILE), metadata={"format": "pt"})
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": lora.rank,
            "lora_alpha": lora.alpha,
            "target_modules": list(lora.target_modules),
            "bias": "none",
            "base_model_name_or_path": base_model,
        }
        (tmp / ADAPTER_CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True))
    _swap_in(dest, write)


def write_optimizer_state(family: str, state: Dict[str, Dict[str, torch.Tensor]], dest: Path) -> None:
    """`dest/<family file>` from {PEFT key: {state key: tensor}}, replacing
    whatever family's file was there."""
    flat = {f"{name}.{field}": t.contiguous() for name, fields in state.items() for field, t in fields.items()}
    _swap_in(dest, lambda tmp: save_file(flat, str(tmp / OPTIMIZER_FILES[family]), metadata={"format": "pt"}))


def optimizer_state_family(src: Path) -> str:
    """The family whose state file `src` holds. Raises ValueError when it
    holds none or more than one."""
    present = [family for family, name in OPTIMIZER_FILES.items() if (src / name).is_file()]
    if len(present) != 1:
        raise ValueError(f"{src} holds {len(present)} optimizer state files, not one of "
                         f"{'|'.join(OPTIMIZER_FILES.values())}")
    return present[0]


def read_optimizer_state(src: Path) -> Tuple[str, Dict[str, Dict[str, torch.Tensor]]]:
    """The inverse of write_optimizer_state: (family, state). Raises ValueError
    on a key that is not `<PEFT key>.<field>` or a parameter missing a field."""
    family = optimizer_state_family(src)
    path = src / OPTIMIZER_FILES[family]
    keys = STATE_KEYS[family]
    state: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, t in load_file(str(path)).items():
        name, _, field = key.rpartition(".")
        if field not in keys or not name:
            raise ValueError(f"{path}: key {key!r} is not <PEFT key>.<{'|'.join(keys)}>")
        state.setdefault(name, {})[field] = t
    for name, fields in state.items():
        if set(fields) != set(keys):
            raise ValueError(f"{path}: {name} has fields {sorted(fields)}, not {sorted(keys)}")
    return family, state


def write_adam_state(state: Dict[str, Dict[str, torch.Tensor]], dest: Path) -> None:
    """write_optimizer_state for the AdamW family."""
    write_optimizer_state("adamw", state, dest)


def read_adam_state(src: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """read_optimizer_state of a directory holding the AdamW file."""
    family, state = read_optimizer_state(src)
    if family != "adamw":
        raise ValueError(f"{src} holds {family} state, not adamw")
    return state
