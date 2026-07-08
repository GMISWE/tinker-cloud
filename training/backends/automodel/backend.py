"""
Automodel backend — HF-encoder classification (ESM2, Nucleotide Transformer).

Hosts supervised sequence- / token-classification via NeMo-Automodel + PEFT
LoRA. Model class reuse:
  - nemo_automodel NeMoAutoModelForSequenceClassification (seq-cls)
  - HF EsmForSequenceClassification / EsmForTokenClassification (ESM2)
  - recipe: nemo_automodel/recipes/llm/train_seq_cls.py

Scaffold only: model load / train / checkpoint internals are NotImplementedError
stubs. Generation methods are only valid for the language_modeling objective.
See specs/004-bionemo-classification/plan.md.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend
from ..objectives import Objective, is_classification

logger = logging.getLogger(__name__)

_PLAN = "specs/004-bionemo-classification/plan.md"


@dataclass
class AutomodelHandle(BackendHandle):
    """Automodel-specific runtime state (classification)."""

    base_model: str = ""
    objective: str = Objective.SEQUENCE_CLASSIFICATION.value
    num_labels: int = 0
    head_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    model: Any = None                 # NeMoAutoModelForSequenceClassification / EsmFor*
    tokenizer: Any = None
    config: Dict = field(default_factory=dict)
    data_buffer: List = field(default_factory=list)
    created_at: str = ""


class AutomodelBackend(TrainingBackend):
    """HF-encoder classification backend (no generation plane)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import ClassificationDataConverter
            self._converter = ClassificationDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import AutomodelArgumentBuilder
            self._builder = AutomodelArgumentBuilder(overrides=self.overrides)
        return self._builder

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False,
        checkpoint_path: Optional[str] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        objective: str = Objective.SEQUENCE_CLASSIFICATION.value,
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> AutomodelHandle:
        """Validate the classification objective + num_labels; record intent.

        Actual model instantiation is deferred (scaffold).
        """
        if not is_classification(objective):
            raise BackendError(
                f"Automodel backend only serves classification objectives, "
                f"got {objective!r}",
                backend="automodel", operation="create_model",
            )
        if not num_labels or num_labels < 2:
            raise BackendError(
                f"num_labels must be >= 2 for classification, got {num_labels!r}",
                backend="automodel", operation="create_model",
            )

        handle = AutomodelHandle(
            model_id=model_id,
            backend_type="automodel",
            base_model=base_model,
            objective=Objective(objective).value,
            num_labels=num_labels,
            head_config=head_config,
            lora_config=lora_config,
            created_at=datetime.now().isoformat(),
        )
        logger.info(
            "[%s] Automodel create_model %s: base=%s objective=%s num_labels=%d",
            request_id, model_id, base_model, handle.objective, num_labels,
        )

        # TODO(004-P2): load NeMoAutoModelForSequenceClassification /
        # EsmFor*Classification + PEFT LoRA (no generation engine). Reuse
        # nemo_automodel/recipes/llm/train_seq_cls.py.
        raise NotImplementedError(
            f"Automodel model load is a scaffold stub. Implement per {_PLAN} (P2)."
        )

    async def forward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """Forward-only logits pass (scaffold)."""
        self.converter.forward_to_backend(data, {"objective": _objective(handle)})
        raise NotImplementedError(
            f"Automodel forward is a scaffold stub. Implement per {_PLAN} (P2)."
        )

    async def forward_backward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """Accept a classification Datum (labels via loss_fn_inputs["labels"]),
        convert to a batch dict, then stub the CE compute (scaffold)."""
        batch = self.converter.forward_backward_to_backend(
            data, loss_fn, {"objective": _objective(handle)},
        )
        logger.info(
            "Automodel forward_backward: %d samples, input_ids=%s",
            len(data), tuple(batch["input_ids"].shape),
        )
        # TODO(004-P2): CE over class logits; LoRA + classification head trainable.
        raise NotImplementedError(
            f"Automodel classification_ce compute is a scaffold stub. "
            f"Implement per {_PLAN} (P2)."
        )

    async def apply_optimizer_step(
        self, handle: BackendHandle, learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimizer step over buffered classification data (scaffold)."""
        raise NotImplementedError(
            f"Automodel apply_optimizer_step is a scaffold stub. "
            f"Implement per {_PLAN} (P2)."
        )

    # --- generation plane: N/A for classification (plan.md Non-Goals) ---

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        raise _no_generation("update_inference_weights")

    async def sample(
        self, handle: BackendHandle, request_id: str, prompt_tokens: List[int],
        num_samples: int, sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
    ) -> Dict[str, Any]:
        raise _no_generation("sample")

    async def get_logprobs(
        self, handle: BackendHandle, data: List[Dict],
    ) -> List[Any]:
        raise _no_generation("get_logprobs")

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        raise _no_generation("prepare_for_generation")

    # --- checkpoint / teardown ---

    async def save_checkpoint(
        self, handle: BackendHandle, checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        """Save adapter + classification head (HF safetensors) (scaffold)."""
        raise NotImplementedError(
            f"Automodel save_checkpoint is a scaffold stub. Implement per {_PLAN} (P2)."
        )

    async def load_checkpoint(
        self, handle: BackendHandle, checkpoint_path: str,
    ) -> None:
        raise NotImplementedError(
            f"Automodel load_checkpoint is a scaffold stub. Implement per {_PLAN} (P2)."
        )

    async def delete_model(self, handle: BackendHandle) -> None:
        """Release model + buffered data (no Ray actors in scaffold)."""
        h: AutomodelHandle = handle  # type: ignore[assignment]
        h.data_buffer.clear()
        h.model = None
        logger.info("Automodel model %s deleted", h.model_id)


def _objective(handle: BackendHandle) -> str:
    return getattr(handle, "objective", Objective.SEQUENCE_CLASSIFICATION.value)


def _no_generation(operation: str) -> BackendError:
    return BackendError(
        "only valid for the language_modeling objective (no generation plane)",
        backend="automodel", operation=operation,
    )
