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
        """Return Megatron-Bridge config for a classification model (no gen engine)."""
        # TODO(004-P5): assemble Megatron-Bridge config (TP/PP/CP, TE, NeMo2
        # checkpoint, Megatron LoRA adapters, num_labels head).
        raise NotImplementedError(
            "MegatronBridgeArgumentBuilder.build_args is a scaffold stub. "
            "Implement per specs/004-bionemo-classification/plan.md (P5)."
        )
