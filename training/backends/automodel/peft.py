"""
How a LoraSpec becomes LoRA modules in Automodel's model.

Automodel's PeftConfig patches each matched nn.Linear into a LinearLoRA with one
lora_A/lora_B pair. A per-slice target (LoraSpec.slices: Qwen3.5's
`in_proj_qkv`, whose output is q ‖ k ‖ v) needs one pair per slice, which
PeftConfig cannot express. SlicedLinearLoRA is that module, and apply_lora puts
it on the per-slice targets after Automodel has patched the others.
NeMoAutoModelForCausalLM.from_pretrained applies the PeftConfig while it builds
the model, before sharding and weight loading, and offers no hook there;
lora_in_automodel replaces the function it calls for the duration of one build,
so both module kinds exist before FSDP2 shards them.
"""
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence, Tuple, cast

import torch
import torch.nn.functional as F
from nemo_automodel._transformers import infrastructure
from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from torch import nn

from ...utils.model_config import declared_mtp_layers
from .lora import LoraSpec

# bf16 LoRA weights would round away small AdamW updates
LORA_DTYPE = torch.float32


def build_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """from_pretrained keyword arguments for a model with this raw config. MTP layers
    train an auxiliary loss this backend never computes (the loss reads .logits) and
    Tinker adapts none of them: build none."""
    return {"num_nextn_predict_layers": 0} if declared_mtp_layers(raw) else {}


def peft_config(lora: LoraSpec) -> PeftConfig:
    """Automodel's config for the targets it patches itself: every target but the per-slice ones."""
    sliced = dict(lora.slices)
    return PeftConfig(
        target_modules=[f"*.{m}" for m in lora.target_modules if m not in sliced],
        dim=lora.rank,
        alpha=lora.alpha,
        dropout=lora.dropout,
        lora_dtype=LORA_DTYPE,
    )


class SlicedLinearLoRA(LinearLoRA):
    """An nn.Linear whose output is the concatenation of slices, with one LoRA pair of
    rank `dim` per slice: y = W x + scale · [B_0 A_0 x ‖ B_1 A_1 x ‖ …]. lora_A[i] and
    lora_B[i] are slice i's factors (parameter names `lora_A.<i>.weight`). A
    LinearLoRA subclass for its attributes (dim, scale, dropout_p); its forward and
    init are its own. Built by patch_sliced, never by the constructor."""
    lora_A: nn.ModuleList
    lora_B: nn.ModuleList
    dim: int
    scale: float
    dropout_p: float
    dropout_position: str
    use_dora: bool
    use_memory_efficient_lora: bool

    def _pairs(self) -> List[Tuple[nn.Linear, nn.Linear]]:
        return [(cast(nn.Linear, a), cast(nn.Linear, b)) for a, b in zip(self.lora_A, self.lora_B)]

    @torch.no_grad()
    def init_lora_weights(self, init_method: str) -> None:
        """LinearLoRA.init_lora_weights per slice; Automodel calls it once the model is materialized."""
        for a, b in self._pairs():
            if init_method == "xavier":
                nn.init.xavier_normal_(a.weight)
            else:
                nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
            b.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.linear(x, self.weight, self.bias)
        if self.dropout_position == "pre":
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        lora = torch.cat([b(a(x.to(a.weight.dtype)) * self.scale) for a, b in self._pairs()], dim=-1)
        if self.dropout_position == "post":
            lora = F.dropout(lora, p=self.dropout_p, training=self.training)
        return res + lora.to(res.dtype)


def patch_sliced(linear: nn.Linear, widths: Sequence[int], lora: LoraSpec, lora_dtype: torch.dtype) -> SlicedLinearLoRA:
    """Make `linear` (in place, as Automodel's patch_linear_module does) a
    SlicedLinearLoRA with slices of `widths`; its weight and bias freeze."""
    if type(linear) is not nn.Linear:
        raise TypeError(f"a per-slice LoRA patches an nn.Linear, not {type(linear).__name__}")
    if sum(widths) != linear.out_features:
        raise ValueError(f"slice widths {tuple(widths)} do not sum to the module's {linear.out_features} outputs")
    linear.weight.requires_grad_(False)
    if linear.bias is not None:
        linear.bias.requires_grad_(False)
    device, in_features = linear.weight.device, linear.in_features
    linear.__class__ = SlicedLinearLoRA
    sliced = cast(SlicedLinearLoRA, linear)
    sliced.lora_A = nn.ModuleList(
        nn.Linear(in_features, lora.rank, bias=False, dtype=lora_dtype, device=device) for _ in widths)
    sliced.lora_B = nn.ModuleList(nn.Linear(lora.rank, w, bias=False, dtype=lora_dtype, device=device) for w in widths)
    sliced.dim = lora.rank
    sliced.scale = lora.alpha / lora.rank
    sliced.dropout_p = lora.dropout
    sliced.dropout_position = "post"   # PeftConfig's default, which every other target here uses
    sliced.use_dora = False
    sliced.use_memory_efficient_lora = False
    sliced.init_lora_weights("xavier")
    return sliced


def _apply(model: nn.Module, config: PeftConfig, lora: LoraSpec, quantization_config: Any) -> None:
    apply_lora_to_linear_modules(model, config, quantization_config=quantization_config, skip_freeze=True)
    sliced = dict(lora.slices)
    found = {m: 0 for m in sliced}
    for name, module in list(model.named_modules()):
        short = name.rsplit(".", 1)[-1]
        if short in sliced:
            assert isinstance(module, nn.Linear), (name, type(module))
            patch_sliced(module, sliced[short], lora, LORA_DTYPE)
            found[short] += 1
    missing = [m for m, n in found.items() if n == 0]
    if missing:
        raise ValueError(f"per-slice LoRA targets {', '.join(missing)} match no module of this model")


def apply_lora(model: nn.Module, lora: LoraSpec) -> None:
    """Put `lora`'s modules into `model` in place: Automodel's LinearLoRA on the
    PeftConfig targets, SlicedLinearLoRA on the per-slice ones. Freezes nothing."""
    _apply(model, peft_config(lora), lora, None)


@contextmanager
def lora_in_automodel(lora: LoraSpec) -> Iterator[None]:
    """While open, a from_pretrained given a PeftConfig applies `lora` as apply_lora
    does, with that (possibly Automodel-adjusted) PeftConfig."""
    original = infrastructure.apply_lora_to_linear_modules

    def apply(model: nn.Module, config: PeftConfig, quantization_config: Any = None, skip_freeze: bool = False) -> int:
        assert skip_freeze, "Automodel freezes after loading; an early freeze would be a new call site"
        _apply(model, config, lora, quantization_config)
        return sum(isinstance(m, LinearLoRA) for m in model.modules())

    infrastructure.apply_lora_to_linear_modules = apply
    try:
        yield
    finally:
        infrastructure.apply_lora_to_linear_modules = original
