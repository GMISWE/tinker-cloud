"""
Automodel argument builder — maps Tinker create_model args + HF config to a
NeMo-Automodel classification config (num_labels, LoRA/PEFT, head_config).

Scaffold: config assembly is stubbed. Reuse target:
  nemo_automodel/recipes/llm/train_seq_cls.py (seq-cls + LoRA recipe).
See specs/004-bionemo-classification/plan.md.
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

        Classification has no generation engine (plan.md:112) — the create
        path must NOT stand up vLLM/SGLang for this objective.
        """
        # TODO(004-P2): assemble NeMoAutoModelForSequenceClassification config
        # (num_labels, head_config, PEFT LoRA targets). See train_seq_cls.py.
        raise NotImplementedError(
            "AutomodelArgumentBuilder.build_args is a scaffold stub. "
            "Implement per specs/004-bionemo-classification/plan.md (P2 backend)."
        )
