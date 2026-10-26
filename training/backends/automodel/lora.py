"""
A request's LoraConfig -> the adapter the training group builds (R2.4, R2.6).

Targets: train_attn -> q/k/v/o_proj, train_mlp -> gate/up/down_proj, the
module names of the Llama/Qwen family; the worker refuses a model in which a
target matches no module. train_unembed is accepted and not applied: no
backend trains a LoRA on the unembedding, and its default (True) is what every
client sends.

The signature (rank, alpha, targets) keys the shared training group; dropout
and the seed are per tenant.
"""
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

from ...models.requests import LoraConfig

ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: int
    target_modules: Tuple[str, ...]   # module names, as PEFT and vLLM read them
    dropout: float

    @property
    def signature(self) -> Tuple[int, int, Tuple[str, ...]]:
        return (self.rank, self.alpha, self.target_modules)


def lora_spec(config: LoraConfig) -> LoraSpec:
    """Raises ValueError for rank 0 (not a LoRA) or no target module."""
    if config.rank == 0:
        raise ValueError("lora_config.rank is 0: not a LoRA model")
    targets = (ATTN_MODULES if config.train_attn else ()) + (MLP_MODULES if config.train_mlp else ())
    if not targets:
        raise ValueError("lora_config trains neither attention nor MLP: the adapter would have no parameters")
    # LoraConfig documents alpha = rank when unset
    alpha = config.alpha if config.alpha is not None else config.rank
    return LoraSpec(rank=config.rank, alpha=alpha, target_modules=targets, dropout=config.dropout)


def tenant_seed(model_id: str, seed: Optional[int]) -> int:
    """The request's seed, else one derived from the model id (R2.6)."""
    if seed is not None:
        return seed
    return int.from_bytes(hashlib.sha256(model_id.encode()).digest()[:8], "little") >> 1


def spec_mismatch(adapter_config: object, lora: LoraSpec) -> Optional[str]:
    """The first field of a PEFT adapter_config.json that does not describe
    `lora`, as "field: checkpoint vs model"; a field of the wrong type is a
    mismatch, never a crash. None when the adapter fits."""
    if not isinstance(adapter_config, dict):
        return "adapter_config.json is not a JSON object"
    for key in ("r", "lora_alpha", "target_modules"):
        if key not in adapter_config:
            return f"{key}: absent from adapter_config.json"
    r, alpha, targets = adapter_config["r"], adapter_config["lora_alpha"], adapter_config["target_modules"]
    if not _is_int(r) or r != lora.rank:
        return f"r: {r!r} vs {lora.rank}"
    if not (_is_int(alpha) or isinstance(alpha, float)) or float(alpha) != float(lora.alpha):
        return f"lora_alpha: {alpha!r} vs {lora.alpha}"
    if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)) \
            or set(targets) != set(lora.target_modules):
        return f"target_modules: {targets!r} vs {sorted(lora.target_modules)}"
    return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)   # JSON true is not a rank
