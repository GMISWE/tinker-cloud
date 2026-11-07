"""
The HF PEFT adapter directory that vLLM and PEFT load (R4, R12): the LoRA
weights under PEFT's keys and the adapter_config.json that names rank, alpha
and target modules. Written by rank 0 of a training group after the gather
(worker.export_adapter); read by the base model's vLLM through
/v1/load_lora_adapter, so the directory must be on a filesystem both see.

A model with a per-slice module (LoraSpec.slices) trains one LoRA pair of rank r
per output slice, which a PEFT adapter cannot name: vLLM gives a module one rank
and shares one A across its slices. The export is therefore one standard
adapter of rank k·r and alpha k·α (k = LoraSpec.export_factor), so the scale
alpha/r is unchanged: the sliced module's A is its slices' A stacked, its B
block-diagonal (slice i's rows take rank columns i·r..(i+1)·r), and every other
module's factors are zero-padded to k·r. It applies the trainer's deltas
exactly; from_peft inverts it.

A checkpoint adds the optimizer state of the same parameters beside it, one
file named for the tenant's family:
`optimizer/adamw.safetensors`: `<PEFT key>.<exp_avg|exp_avg_sq|step>`, or
`optimizer/dimuon.safetensors`: `<PEFT key>.momentum`; unsharded, keyed as
the adapter file is. Empty before the first optimizer step; the file still
exists, so a load with optimizer=true restores "no step taken" instead of
being refused (that refusal is for roots another backend wrote).
"""
import json
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Sequence, Set, Tuple

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

# Wrapper segments torch.compile and activation checkpointing put into module
# names; PEFT and vLLM address the module underneath.
_WRAPPER_SEGMENTS = ("_orig_mod.", "_checkpoint_wrapped_module.")


def peft_key(name: str) -> str:
    """`model.layers.0.self_attn.q_proj.lora_A.weight` ->
    `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`."""
    for segment in _WRAPPER_SEGMENTS:
        name = name.replace(segment, "")
    return PEFT_PREFIX + name


# `<module>.lora_A.weight`, or `<module>.lora_A.<slice>.weight` on a per-slice module
_LORA_NAME = re.compile(r"^(?P<module>.+)\.lora_(?P<factor>[AB])(?:\.(?P<slice>\d+))?\.weight$")


def _modules(names: Sequence[str]) -> Dict[str, int]:
    """{module's PEFT key prefix: its slice count, 0 for an unsliced module} from LoRA
    parameter names. Raises ValueError on a name that is not a LoRA factor's."""
    modules: Dict[str, int] = {}
    for name in names:
        m = _LORA_NAME.match(peft_key(name))
        if m is None:
            raise ValueError(f"{name} is not a LoRA factor's name")
        slices = 0 if m["slice"] is None else int(m["slice"]) + 1
        modules[m["module"]] = max(modules.get(m["module"], 0), slices)
    return modules


def _slice_widths(module: str, slices: int, lora: LoraSpec, out: int) -> List[int]:
    """The output widths of the module's slices; one slice of width `out` when unsliced."""
    if slices == 0:
        return [out]
    widths = dict(lora.slices)[module.rsplit(".", 1)[-1]]
    assert len(widths) == slices, (module, widths, slices)
    return list(widths)


def _factor(module: str, factor: str, i: Optional[int]) -> str:
    return f"{module}.lora_{factor}.weight" if i is None else f"{module}.lora_{factor}.{i}.weight"


def to_peft(state: Dict[str, torch.Tensor], lora: LoraSpec) -> Dict[str, torch.Tensor]:
    """{LoRA parameter name: tensor} -> the exported adapter's {PEFT key: tensor}."""
    keyed = {peft_key(n): t for n, t in state.items()}
    if not lora.slices:
        return {k: t.contiguous() for k, t in keyed.items()}
    r, rank = lora.rank, lora.export_rank
    out: Dict[str, torch.Tensor] = {}
    for module, slices in _modules(list(state)).items():
        index: List[Optional[int]] = list(range(slices)) if slices else [None]
        a_parts = [keyed[_factor(module, "A", i)] for i in index]
        b_parts = [keyed[_factor(module, "B", i)] for i in index]
        a = a_parts[0].new_zeros(rank, a_parts[0].shape[1])
        b = b_parts[0].new_zeros(sum(p.shape[0] for p in b_parts), rank)
        row = 0
        for i, (pa, pb) in enumerate(zip(a_parts, b_parts)):
            a[i * r:(i + 1) * r] = pa
            b[row:row + pb.shape[0], i * r:(i + 1) * r] = pb
            row += pb.shape[0]
        out[_factor(module, "A", None)], out[_factor(module, "B", None)] = a, b
    return out


def from_peft(weights: Dict[str, torch.Tensor], names: Sequence[str], lora: LoraSpec) -> Dict[str, torch.Tensor]:
    """The inverse of to_peft for the model whose LoRA parameters are `names`:
    {PEFT key of each name: tensor}. Raises ValueError when the keys are not the
    export's, or an entry to_peft leaves zero (padding, off-block) is not: such an
    adapter holds a LoRA these parameters cannot represent."""
    if not lora.slices:
        plain = {peft_key(n) for n in names}
        if set(weights) != plain:
            _key_error(weights, plain)
        return {k: weights[k] for k in plain}
    modules = _modules(names)
    want = {_factor(m, f, None) for m in modules for f in "AB"}
    if set(weights) != want:
        _key_error(weights, want)
    r, rank = lora.rank, lora.export_rank
    out: Dict[str, torch.Tensor] = {}
    for module, slices in modules.items():
        a, b = weights[_factor(module, "A", None)], weights[_factor(module, "B", None)]
        if a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"{module}: rank {a.shape[0]}/{b.shape[1]}, not the export's {rank}")
        widths = _slice_widths(module, slices, lora, b.shape[0])
        rebuilt_a, rebuilt_b = torch.zeros_like(a), torch.zeros_like(b)
        row = 0
        for i, w in enumerate(widths):
            pa, pb = a[i * r:(i + 1) * r], b[row:row + w, i * r:(i + 1) * r]
            rebuilt_a[i * r:(i + 1) * r], rebuilt_b[row:row + w, i * r:(i + 1) * r] = pa, pb
            j = i if slices else None
            out[_factor(module, "A", j)], out[_factor(module, "B", j)] = pa.clone(), pb.clone()
            row += w
        if not (torch.equal(a, rebuilt_a) and torch.equal(b, rebuilt_b)):
            raise ValueError(f"{module}: nonzero entries outside its {len(widths)} rank-{r} block(s); "
                             "the adapter is not one this model's LoRA exports")
    return out


def _key_error(weights: Dict[str, torch.Tensor], want: Set[str]) -> NoReturn:
    missing, unknown = sorted(want - set(weights)), sorted(set(weights) - want)
    raise ValueError(f"adapter keys are not this model's LoRA: {len(missing)} missing (first: {missing[:1]}), "
                     f"{len(unknown)} unknown (first: {unknown[:1]})")


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
        save_file(to_peft(state, lora), str(tmp / ADAPTER_WEIGHTS_FILE), metadata={"format": "pt"})
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": lora.export_rank,
            "lora_alpha": lora.export_alpha,
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
