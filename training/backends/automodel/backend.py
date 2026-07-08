"""
Automodel backend — HF-encoder classification (ESM2, Nucleotide Transformer).

Hosts supervised sequence- / token-classification via HuggingFace
`AutoModelFor{Sequence,Token}Classification` + PEFT LoRA. This is the
P0-proven path (transformers + peft): ESM2 `EsmFor*Classification` load, LoRA
over `[query,key,value,dense]`, CE over `loss_fn_inputs["labels"]`.

Training model (unlike NeMo RL's R9 buffering): forward_backward runs the real
forward + `loss.backward()` immediately and returns the real loss;
apply_optimizer_step runs `optimizer.step()` over the accumulated grads. There
is NO generation plane — sample / get_logprobs are only valid for the
language_modeling objective.

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

# P0-proven LoRA targets for ESM/BERT-style encoders (plan.md P0 findings).
_DEFAULT_LORA_TARGETS = ["query", "key", "value", "dense"]
_DEFAULT_LR = 1e-4


@dataclass
class AutomodelHandle(BackendHandle):
    """Automodel-specific runtime state (classification)."""

    base_model: str = ""
    objective: str = Objective.SEQUENCE_CLASSIFICATION.value
    num_labels: int = 0
    head_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    model: Any = None                 # PeftModel wrapping EsmFor*Classification
    tokenizer: Any = None
    optimizer: Any = None
    device: Any = None
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
        """Load an HF classification model + PEFT LoRA (no generation engine)."""
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

        objective = Objective(objective).value
        handle = AutomodelHandle(
            model_id=model_id,
            backend_type="automodel",
            base_model=base_model,
            objective=objective,
            num_labels=num_labels,
            head_config=head_config,
            lora_config=lora_config,
            created_at=datetime.now().isoformat(),
        )
        logger.info(
            "[%s] Automodel create_model %s: base=%s objective=%s num_labels=%d",
            request_id, model_id, base_model, objective, num_labels,
        )

        import asyncio
        await asyncio.to_thread(
            self._build_model, handle, base_model, objective, num_labels,
            lora_config, head_config, checkpoint_path,
        )
        return handle

    def _build_model(
        self, handle: AutomodelHandle, base_model: str, objective: str,
        num_labels: int, lora_config: Optional[Dict[str, Any]],
        head_config: Optional[Dict[str, Any]], checkpoint_path: Optional[str],
    ) -> None:
        """Blocking model construction (runs in a thread)."""
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
        )

        hc = head_config or {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_dtype(hc.get("torch_dtype"))

        model_cls = (
            AutoModelForTokenClassification
            if Objective(objective) == Objective.TOKEN_CLASSIFICATION
            else AutoModelForSequenceClassification
        )
        try:
            # transformers >=4.56 renamed torch_dtype -> dtype (old name warns).
            model = model_cls.from_pretrained(
                base_model, num_labels=num_labels, dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as e:  # noqa: BLE001 — surface load failures as BackendError
            raise BackendError(
                f"failed to load {base_model!r} as {model_cls.__name__}: {e}",
                backend="automodel", operation="create_model", original_error=e,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        model = _apply_lora(model, objective, lora_config, hc)
        if hc.get("freeze_base"):
            # Linear-probe / head-only baseline: train only the classification
            # head, freeze the encoder. (plan.md CS2 head-only ablation.)
            for name, p in model.named_parameters():
                p.requires_grad = "classifier" in name or "score" in name
        model.to(device)
        model.train()

        lr = float(hc.get("learning_rate", _DEFAULT_LR))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr)

        handle.model = model
        handle.tokenizer = tokenizer
        handle.optimizer = optimizer
        handle.device = device
        handle.config = {"torch_dtype": str(dtype), "learning_rate": lr}

        if checkpoint_path:
            self._load_adapter(handle, checkpoint_path)

        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Automodel model %s ready: %s on %s, trainable %d/%d (%.2f%%)",
            handle.model_id, model_cls.__name__, device, n_train, n_total,
            100.0 * n_train / max(n_total, 1),
        )

    async def forward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """Forward-only logits pass (no gradient)."""
        import asyncio
        h: AutomodelHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_to_backend(
            data, {"objective": h.objective},
        )
        result = await asyncio.to_thread(self._forward_only, h, batch)
        return self.converter.backend_to_forward_result(result, data)

    def _forward_only(self, h: AutomodelHandle, batch) -> Dict[str, Any]:
        import torch
        model_in = _to_device(batch, h.device)
        with torch.no_grad():
            out = h.model(
                input_ids=model_in["input_ids"],
                attention_mask=model_in["attention_mask"],
            )
        return {"logits": out.logits.float()}

    async def forward_backward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """Convert a classification Datum, run forward + loss.backward()
        immediately, and return the real loss (deferred=False)."""
        import asyncio
        h: AutomodelHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_backward_to_backend(
            data, loss_fn, {"objective": h.objective},
        )
        logger.info(
            "Automodel forward_backward: %d samples, input_ids=%s",
            len(data), tuple(batch["input_ids"].shape),
        )
        result = await asyncio.to_thread(self._forward_backward, h, batch)
        return self.converter.backend_to_forward_backward_result(result, data)

    def _forward_backward(self, h: AutomodelHandle, batch) -> Dict[str, Any]:
        model_in = _to_device(batch, h.device)
        out = h.model(
            input_ids=model_in["input_ids"],
            attention_mask=model_in["attention_mask"],
            labels=model_in["labels"],
        )
        out.loss.backward()
        return {"loss": out.loss.detach().float().item()}

    async def apply_optimizer_step(
        self, handle: BackendHandle, learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Optimizer step over accumulated grads, then zero them."""
        import asyncio
        h: AutomodelHandle = handle  # type: ignore[assignment]
        return await asyncio.to_thread(self._optimizer_step, h, learning_rate)

    def _optimizer_step(
        self, h: AutomodelHandle, learning_rate: Optional[float],
    ) -> Dict[str, Any]:
        import torch
        if learning_rate is not None:
            for group in h.optimizer.param_groups:
                group["lr"] = float(learning_rate)
        trainable = [
            p for p in h.model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        grad_norm = 0.0
        if trainable:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float("inf"))
            )
        h.optimizer.step()
        h.optimizer.zero_grad(set_to_none=True)
        return {
            "success": True,
            "grad_norm": grad_norm,
            "metrics": {"grad_norm": grad_norm},
        }

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
        """Save the LoRA adapter + classification head (HF safetensors)."""
        import asyncio
        h: AutomodelHandle = handle  # type: ignore[assignment]

        def _save() -> str:
            import os
            os.makedirs(checkpoint_path, exist_ok=True)
            # PeftModel.save_pretrained writes the adapter; task_type SEQ_CLS/
            # TOKEN_CLS keeps the classifier in modules_to_save so it is saved too.
            h.model.save_pretrained(checkpoint_path)
            if h.tokenizer is not None:
                h.tokenizer.save_pretrained(checkpoint_path)
            return checkpoint_path

        path = await asyncio.to_thread(_save)
        logger.info("Automodel checkpoint saved: %s (step=%s)", path, step_id)
        return path

    async def load_checkpoint(
        self, handle: BackendHandle, checkpoint_path: str,
    ) -> None:
        import asyncio
        h: AutomodelHandle = handle  # type: ignore[assignment]
        await asyncio.to_thread(self._load_adapter, h, checkpoint_path)
        logger.info("Automodel checkpoint loaded: %s", checkpoint_path)

    def _load_adapter(self, h: AutomodelHandle, checkpoint_path: str) -> None:
        """Load adapter weights into the existing PeftModel."""
        try:
            h.model.load_adapter(checkpoint_path, adapter_name="default")
        except Exception as e:  # noqa: BLE001
            raise BackendError(
                f"failed to load adapter from {checkpoint_path!r}: {e}",
                backend="automodel", operation="load_checkpoint",
                original_error=e,
            )

    async def delete_model(self, handle: BackendHandle) -> None:
        """Release model + buffered data (no Ray actors in scaffold)."""
        h: AutomodelHandle = handle  # type: ignore[assignment]
        h.data_buffer.clear()
        h.model = None
        h.optimizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — teardown best-effort
            pass
        logger.info("Automodel model %s deleted", h.model_id)


def _apply_lora(model, objective: str, lora_config, head_config: Dict[str, Any]):
    """Wrap the model in a PEFT LoRA adapter for the classification task type.

    Returns the base model unchanged when no LoRA rank is requested.
    """
    rank = (lora_config or {}).get("rank", 0) if lora_config else 0
    if not rank:
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    task_type = (
        TaskType.TOKEN_CLS
        if Objective(objective) == Objective.TOKEN_CLASSIFICATION
        else TaskType.SEQ_CLS
    )
    targets = head_config.get("target_modules") or _DEFAULT_LORA_TARGETS
    alpha = (lora_config or {}).get("alpha") or (2 * rank)
    dropout = (lora_config or {}).get("dropout", 0.0)

    peft_config = LoraConfig(
        task_type=task_type,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    return get_peft_model(model, peft_config)


def _resolve_dtype(name: Optional[str]):
    import torch
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if not name:
        return torch.float32
    return mapping.get(str(name).lower(), torch.float32)


def _to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _no_generation(operation: str) -> BackendError:
    return BackendError(
        "only valid for the language_modeling objective (no generation plane)",
        backend="automodel", operation=operation,
    )
