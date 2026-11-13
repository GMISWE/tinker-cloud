"""Cross-backend checkpoint interchange.

One canonical artifact next to (never instead of) each backend's native
checkpoint::

    <checkpoint_root>/hf_adapter/
    ├── adapter_model.safetensors   # PEFT-standard keys
    └── adapter_config.json         # peft_type / r / lora_alpha / target_modules

Every backend writes it on save_checkpoint and reads it on resume, so a
program can move between engines; native formats stay for same-backend
fast resume. Where a checkpoint root lives is the store's business
(training/checkpoints/store.py); this module only knows the adapter's shape
inside a root. Design + measured seam gates: specs/007-q5-migration/HANDOFF.md.
"""
import json
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)

HF_ADAPTER_DIRNAME = "hf_adapter"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
ADAPTER_WEIGHTS_TORCH_FILE = "adapter_model.bin"
ADAPTER_CONFIG_FILE = "adapter_config.json"


def hf_adapter_dir(checkpoint_root: str) -> str:
    return os.path.join(checkpoint_root, HF_ADAPTER_DIRNAME)


def find_hf_adapter(checkpoint_root: str) -> Optional[str]:
    """Return the interchange dir if it holds a usable adapter, else None."""
    candidate = hf_adapter_dir(checkpoint_root)
    if os.path.isfile(os.path.join(candidate, ADAPTER_WEIGHTS_FILE)):
        return candidate
    return None


PEFT_PREFIX = "base_model.model."


def _write_peft_keyed(weights_src: str, weights_dst: str) -> None:
    """Copy the adapter weights, normalizing state-dict keys to the PEFT
    convention (`base_model.model.<module>.lora_{A,B}.weight`).

    Miles/Megatron-Bridge exports bare module paths (`model.layers.0...`).
    `PeftModel.from_pretrained` matches those against nothing, WARNS about
    missing adapter keys, and returns the base model — a silent zero-delta
    load. Measured 2026-08-03: unkeyed round-trip corr 0.806 / gap 0.986
    nats vs 0.99964 / 0.0033 after normalization.
    """
    try:
        from safetensors.torch import load_file, save_file
    except ImportError:  # noqa: TREE002  (#192)
        logger.warning("safetensors unavailable; publishing adapter keys unmodified")
        shutil.copy2(weights_src, weights_dst)
        return

    state = load_file(weights_src)
    if any(k.startswith(PEFT_PREFIX) for k in state):
        shutil.copy2(weights_src, weights_dst)   # already PEFT-keyed: keep bytes
        return
    logger.info("Re-keying %d adapter tensors to the PEFT convention", len(state))
    save_file(
        {f"{PEFT_PREFIX}{k}": v for k, v in state.items()},
        weights_dst,
        metadata={"format": "pt"},
    )


def _write_peft_keyed_from_torch(weights_src: str, weights_dst: str) -> None:
    """Convert a torch-serialized adapter (`adapter_model.bin`) to the
    interchange safetensors, normalizing keys as `_write_peft_keyed` does.

    Miles falls back to `.bin` because its fused-QKV export aliases the
    shared lora_A across q/k/v and safetensors refuses shared storage;
    tensors are cloned here so the file is loadable everywhere downstream.
    """
    import torch
    from safetensors.torch import save_file

    state = torch.load(weights_src, map_location="cpu", weights_only=True)
    logger.info("Converting %d torch-serialized adapter tensors to safetensors", len(state))
    save_file(
        {
            (k if k.startswith(PEFT_PREFIX) else f"{PEFT_PREFIX}{k}"):
                v.detach().clone().contiguous()
            for k, v in state.items()
        },
        weights_dst,
        metadata={"format": "pt"},
    )


def export_hf_adapter(src_dir: str, checkpoint_root: str) -> Optional[str]:
    """Publish an adapter a backend just wrote into the interchange dir.

    src_dir is the backend-native location of the PEFT pair. Returns the
    interchange dir, or None when the backend produced no adapter (full
    fine-tune, or a save that predates the first optimizer step).
    """
    weights_src = os.path.join(src_dir, ADAPTER_WEIGHTS_FILE)
    weights_torch_src = os.path.join(src_dir, ADAPTER_WEIGHTS_TORCH_FILE)
    if not os.path.isfile(weights_src) and not os.path.isfile(weights_torch_src):
        logger.info(
            "No %s or %s under %s; skipping interchange export",
            ADAPTER_WEIGHTS_FILE, ADAPTER_WEIGHTS_TORCH_FILE, src_dir,
        )
        return None

    dest = hf_adapter_dir(checkpoint_root)
    # Publish atomically: a reader on another pod must never see half a copy.
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    if os.path.isfile(weights_src):
        _write_peft_keyed(weights_src, os.path.join(tmp, ADAPTER_WEIGHTS_FILE))
    else:
        _write_peft_keyed_from_torch(weights_torch_src, os.path.join(tmp, ADAPTER_WEIGHTS_FILE))

    config_src = os.path.join(src_dir, ADAPTER_CONFIG_FILE)
    if os.path.isfile(config_src):
        shutil.copy2(config_src, os.path.join(tmp, ADAPTER_CONFIG_FILE))
    else:
        logger.warning("No %s under %s — importers must be told r/alpha", ADAPTER_CONFIG_FILE, src_dir)

    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    logger.info("Published interchange adapter %s -> %s", src_dir, dest)
    return dest


def stage_hf_adapter(checkpoint_root: str, dest_dir: str) -> Optional[str]:
    """Materialize the interchange adapter in a backend's own load layout.

    No-op (returns None) when there is nothing to stage or when dest_dir
    already holds an adapter — a backend's native artifact always wins over
    the interchange copy of itself.
    """
    src = find_hf_adapter(checkpoint_root)
    if src is None:
        return None
    if os.path.isfile(os.path.join(dest_dir, ADAPTER_WEIGHTS_FILE)):
        return None

    os.makedirs(dest_dir, exist_ok=True)
    for name in (ADAPTER_WEIGHTS_FILE, ADAPTER_CONFIG_FILE):
        srcfile = os.path.join(src, name)
        if os.path.isfile(srcfile):
            shutil.copy2(srcfile, os.path.join(dest_dir, name))
    logger.info("Staged interchange adapter %s -> %s", src, dest_dir)
    return dest_dir


def read_adapter_config(checkpoint_root: str) -> Optional[dict]:
    """r/alpha/target_modules of the interchange adapter, if published."""
    src = find_hf_adapter(checkpoint_root)
    if src is None:
        return None
    path = os.path.join(src, ADAPTER_CONFIG_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001  (#192)
        logger.warning("Unreadable %s", path, exc_info=True)
        return None
