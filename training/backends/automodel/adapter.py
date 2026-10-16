"""
The HF PEFT adapter directory that vLLM and PEFT load (R4, R12): the LoRA
weights under PEFT's keys and the adapter_config.json that names rank, alpha
and target modules. Written by rank 0 of a training group after the gather
(worker.export_adapter); read by the base model's vLLM through
/v1/load_lora_adapter, so the directory must be on a filesystem both see.
"""
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file

from ...checkpoints.interchange import ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, PEFT_PREFIX
from .lora import LoraSpec

# Wrapper segments torch.compile and activation checkpointing put into module
# names; PEFT and vLLM address the module underneath.
_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.")


def peft_key(name: str) -> str:
    """`model.layers.0.self_attn.q_proj.lora_A.weight` ->
    `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`."""
    for segment in _WRAPPER_SEGMENTS:
        name = name.replace(segment, "")
    return PEFT_PREFIX + name


def write_peft_adapter(state: Dict[str, torch.Tensor], lora: LoraSpec, base_model: str, dest: Path) -> None:
    """Write the adapter at `dest`, replacing what is there. The write lands
    in a sibling directory and is renamed into place, so a reader never sees
    a half-written adapter."""
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
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
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
