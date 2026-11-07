"""
A request's LoraConfig -> the adapter the training group builds (R2.4, R2.6).

Targets: train_attn -> q/k/v/o_proj, train_mlp -> gate/up/down_proj, the
module names of the Llama/Qwen family; the worker refuses a model in which a
target matches no module. A model with Gated DeltaNet layers (config
`layer_types` "linear_attention", Qwen3.5) adds `in_proj_qkv`, `in_proj_z` and
`out_proj` to train_attn, with one LoRA pair per q/k/v slice of `in_proj_qkv`
(model_lora). train_unembed is accepted and not applied: no backend trains a
LoRA on the unembedding, and its default (True) is what every client sends.

The signature (rank, alpha, targets) keys the shared training group; dropout
and the seed are per tenant.
"""
import hashlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ...models.requests import LoraConfig

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
# Gated DeltaNet's projections that train_attn adapts; in_proj_b/in_proj_a are not
DELTANET_MODULES = ("in_proj_qkv", "in_proj_z", "out_proj")
# layer types whose attention is q/k/v/o_proj: a sliding layer changes the mask, not the modules
# (nemo_automodel's GptOssAttention is one class for both, sliding_window set per layer type)
_SOFTMAX_LAYER_TYPES = ("full_attention", "sliding_attention")
# keys other hybrid families declare their layer kinds under (Jamba, Nemotron-H); not mapped here
_OTHER_LAYER_KEYS = ("layers_block_type", "hybrid_override_pattern")


@dataclass(frozen=True)
class LoraSpec:
    """What add_tenant receives: the group's signature (rank, alpha, targets)
    plus the tenant's own dropout and optimizer family."""
    rank: int
    alpha: int
    target_modules: Tuple[str, ...]   # module names, as PEFT and vLLM read them
    dropout: float
    optimizer: str   # "adamw" | "dimuon", one of the backend's SUPPORTED_OPTIMIZERS
    # (module name, output widths): a target that carries one LoRA pair of this rank per
    # slice of its output. Empty for a model with no such module.
    # () is the layout of a model with no per-slice module; model_lora adds Qwen3.5's
    slices: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()

    @property
    def signature(self) -> Tuple[int, int, Tuple[str, ...]]:
        return (self.rank, self.alpha, self.target_modules)

    @property
    def export_factor(self) -> int:
        """k: the exported adapter is one standard LoRA of rank k·rank and alpha k·alpha,
        k the most slices a module has (adapter.to_peft)."""
        return max((len(widths) for _, widths in self.slices), default=1)

    @property
    def export_rank(self) -> int:
        return self.export_factor * self.rank

    @property
    def export_alpha(self) -> int:
        return self.export_factor * self.alpha


def lora_spec(config: LoraConfig, optimizer: str) -> LoraSpec:
    """Raises ValueError for rank 0 (not a LoRA) or no target module."""
    if config.rank == 0:
        raise ValueError("lora_config.rank is 0: not a LoRA model")
    targets = (ATTN_MODULES if config.train_attn else ()) + (MLP_MODULES if config.train_mlp else ())
    if not targets:
        raise ValueError("lora_config trains neither attention nor MLP: the adapter would have no parameters")
    # LoraConfig documents alpha = rank when unset
    alpha = config.alpha if config.alpha is not None else config.rank
    return LoraSpec(rank=config.rank, alpha=alpha, target_modules=targets, dropout=config.dropout, optimizer=optimizer)


def model_lora(spec: LoraSpec, raw: Dict[str, Any]) -> LoraSpec:
    """`spec` with the modules train_attn means on this model (its raw config.json,
    text_config for a composite model). Raises ValueError for a layer type whose
    modules are not known here."""
    text = raw["text_config"] if "text_config" in raw else raw
    other = [k for k in _OTHER_LAYER_KEYS if k in text]
    if other:
        raise ValueError(f"layer kinds declared under {other} have no LoRA target mapping in the automodel backend")
    # a dense config (Llama, Qwen2, Qwen3) declares no layer kinds: every layer is softmax attention
    types = set(text["layer_types"]) if "layer_types" in text else {"full_attention"}
    unknown = types - set(_SOFTMAX_LAYER_TYPES) - {"linear_attention"}
    if unknown:
        raise ValueError(f"layer types {sorted(unknown)} have no LoRA target mapping in the automodel backend")
    if "linear_attention" not in types or ATTN_MODULES[0] not in spec.target_modules:
        return spec
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    targets = ATTN_MODULES + DELTANET_MODULES + tuple(m for m in spec.target_modules if m not in ATTN_MODULES)
    return replace(spec, target_modules=targets, slices=(("in_proj_qkv", (key, key, value)),))


def vllm_rank_factor(raw: Dict[str, Any]) -> int:
    """The largest export_factor a LoRA on this model can have: the vLLM serving it
    must hold adapters of max_lora_rank times this."""
    probe = LoraSpec(rank=1, alpha=1, target_modules=ATTN_MODULES + MLP_MODULES, dropout=0.0, optimizer="adamw")
    return model_lora(probe, raw).export_factor


def tenant_seed(model_id: str, seed: Optional[int]) -> int:
    """The request's seed, else one derived from the model id (R2.6)."""
    if seed is not None:
        return seed
    return int.from_bytes(hashlib.sha256(model_id.encode()).digest()[:8], "little") >> 1


def spec_mismatch(adapter_config: object, lora: LoraSpec) -> Optional[str]:
    """The first field of a PEFT adapter_config.json that does not describe
    `lora` as exported (export_rank, export_alpha), as "field: checkpoint vs model"; a field of the wrong type is a
    mismatch, never a crash. None when the adapter fits."""
    if not isinstance(adapter_config, dict):
        return "adapter_config.json is not a JSON object"
    for key in ("r", "lora_alpha", "target_modules"):
        if key not in adapter_config:
            return f"{key}: absent from adapter_config.json"
    r, alpha, targets = adapter_config["r"], adapter_config["lora_alpha"], adapter_config["target_modules"]
    if not _is_int(r) or r != lora.export_rank:
        return f"r: {r!r} vs {lora.export_rank}"
    if not (_is_int(alpha) or isinstance(alpha, float)) or float(alpha) != float(lora.export_alpha):
        return f"lora_alpha: {alpha!r} vs {lora.export_alpha}"
    if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)) \
            or set(targets) != set(lora.target_modules):
        return f"target_modules: {targets!r} vs {sorted(lora.target_modules)}"
    return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)   # JSON true is not a rank
