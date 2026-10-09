"""
Automodel argument builder — maps Tinker create_model args + HF config to a
NeMo-Automodel classification config (num_labels, LoRA/PEFT, head_config).

"""
import logging
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder

logger = logging.getLogger(__name__)


class AutomodelArgumentBuilder(ArgumentBuilder):
    """Builds NeMo-Automodel classification config from Tinker API parameters."""

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
        """Return (config_dict, hf_path) for a classification model.

        Classification has no generation engine (plan.md) — the create path
        must NOT stand up vLLM/SGLang for this objective. AutomodelBackend
        loads HF `AutoModelFor*Classification` directly; this builder resolves
        the LoRA/head knobs into a plain config dict for logging/metadata.
        """
        classification = kwargs["classification"]
        lora = lora_config or {}
        rank = lora.get("rank", 0)

        config = {
            "base_model": base_model,
            "objective": classification.objective.value,
            "num_labels": classification.num_labels,
            "lora": {
                "rank": rank,
                "alpha": lora.get("alpha") or (2 * rank),
                "dropout": lora.get("dropout", 0.0),
                "train_attn": lora.get("train_attn", True),
                "train_mlp": lora.get("train_mlp", True),
            } if rank else None,
        }
        config.update(self.overrides)
        # hf_path == base_model: HF resolves the checkpoint from the model id.
        return config, base_model
