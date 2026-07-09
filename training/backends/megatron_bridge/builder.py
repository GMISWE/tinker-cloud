"""
Megatron-Bridge argument builder — maps Tinker create_model args to a
Megatron-Bridge classification config (Evo2 StripedHyena, NeMo2 checkpoints).

Scaffold: config assembly is stubbed. Reuse targets: megatron.bridge.AutoBridge,
bionemo-evo2 StripedHyena model def, NeMo2 dist-checkpoints, Transformer Engine.
See specs/004-bionemo-classification/plan.md.
"""
import logging
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder

logger = logging.getLogger(__name__)


class MegatronBridgeArgumentBuilder(ArgumentBuilder):
    """Builds Megatron-Bridge classification config from Tinker API parameters."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}

    def build_args(
        self,
        base_model: str,
        num_gpus: int = 4,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Resolve LoRA/head knobs to a dict.

        Unused by the backend today: MegatronBridgeBackend.create_model builds the
        recipe's ConfigContainer directly via evo2_1b_classifier_config(). Kept as
        the ArgumentBuilder seam; returns a flat config dict for callers that want
        one (mirrors AutomodelArgumentBuilder)."""
        cfg = dict(self.overrides)
        cfg.update({"base_model": base_model, "num_gpus": num_gpus})
        if lora_config:
            cfg["lora"] = lora_config
        if checkpoint_config:
            cfg["checkpoint"] = checkpoint_config
        return cfg
