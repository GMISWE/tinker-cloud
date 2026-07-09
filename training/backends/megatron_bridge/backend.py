"""
Megatron-Bridge backend — Evo2 (Striped Hyena) sequence classification.

Hosts the faithful BioNeMo CS2 recipe (bionemo-recipes/recipes/evo2_megatron/
examples/evo2_classifier.py) behind the tinker TrainingBackend API by
decomposing megatron-bridge's monolithic ``pretrain()`` into per-step primitives:
  - create_model         = evo2_1b_classifier_config + GlobalState + setup()
                           -> model / optimizer / scheduler
  - forward_backward      = get_forward_backward_func()(classifier_forward_step,
                           iter([batch]), model, num_microbatches=1)  (grads)
  - apply_optimizer_step  = optimizer.step() + scheduler.step()

Reproduces ~0.858 test acc on splice_sites_all (see P5-RESULTS.md). Runs ONLY in
the cu13 recipe venv (deploy_tinkercloud.sh --profile megatron_bridge); the
megatron.bridge + evo2_classifier imports are lazy so the module still loads
elsewhere. Generation plane raises (classification, no sampling).
See specs/004-bionemo-classification/P5-TINKER-BACKEND.md.
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend
from ..objectives import Objective, is_classification

logger = logging.getLogger(__name__)

_PLAN = "specs/004-bionemo-classification/P5-TINKER-BACKEND.md"

# Recipe locations inside the cu13 recipe env (deploy --profile megatron_bridge).
_RECIPE_DIR = os.environ.get("EVO2_RECIPE_DIR", "/workspace/evo2_megatron")
_RECIPE_EXAMPLES = os.path.join(_RECIPE_DIR, "examples")
_TOKENIZER_PATH = os.environ.get(
    "EVO2_TOKENIZER_PATH", os.path.join(_RECIPE_DIR, "tokenizers", "nucleotide_fast_tokenizer_512")
)


@dataclass
class MegatronBridgeHandle(BackendHandle):
    """Megatron-Bridge-specific runtime state (classification)."""

    base_model: str = ""
    objective: str = Objective.SEQUENCE_CLASSIFICATION.value
    num_labels: int = 0
    head_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    model: Any = None                 # list[MegatronModule] (Hyena classifier + LoRA + head)
    optimizer: Any = None             # MegatronOptimizer
    scheduler: Any = None             # OptimizerParamScheduler
    state: Any = None                 # megatron.bridge GlobalState
    forward_step: Any = None          # partial(classifier_forward_step, state)
    fbfunc: Any = None                # get_forward_backward_func()
    seq_length: int = 1024
    needs_zero: bool = True           # zero grads on the next forward_backward
    config: Dict = field(default_factory=dict)
    created_at: str = ""


class MegatronBridgeBackend(TrainingBackend):
    """Megatron-native classification backend (no generation plane)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import NeMo2ClassificationDataConverter
            self._converter = NeMo2ClassificationDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import MegatronBridgeArgumentBuilder
            self._builder = MegatronBridgeArgumentBuilder(overrides=self.overrides)
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
    ) -> MegatronBridgeHandle:
        """Build the Evo2 classifier + LoRA + optimizer via the recipe's
        config + megatron-bridge ``setup()`` (one-time; per-step control after)."""
        if not is_classification(objective):
            raise BackendError(
                f"Megatron-Bridge backend only serves classification objectives, "
                f"got {objective!r}",
                backend="megatron_bridge", operation="create_model",
            )
        if not num_labels or num_labels < 2:
            raise BackendError(
                f"num_labels must be >= 2 for classification, got {num_labels!r}",
                backend="megatron_bridge", operation="create_model",
            )

        hc = head_config or {}
        lc = lora_config or {}
        seq_length = int(hc.get("seq_length", max_seq_len if max_seq_len <= 8192 else 1024))
        handle = MegatronBridgeHandle(
            model_id=model_id,
            backend_type="megatron_bridge",
            base_model=base_model,
            objective=Objective(objective).value,
            num_labels=num_labels,
            head_config=head_config,
            lora_config=lora_config,
            seq_length=seq_length,
            created_at=datetime.now().isoformat(),
        )

        # Lazy imports — only present in the cu13 recipe venv.
        if _RECIPE_EXAMPLES not in sys.path:
            sys.path.insert(0, _RECIPE_EXAMPLES)
        try:
            import functools
            from megatron.bridge.training.state import GlobalState
            from megatron.bridge.training.setup import setup
            from megatron.bridge.data.utils import get_dataset_provider
            from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
            from evo2_classifier import evo2_1b_classifier_config, classifier_forward_step
        except Exception as e:  # pragma: no cover - env-specific
            raise BackendError(
                f"megatron-bridge/evo2 recipe not importable ({e!r}). Run under "
                f"deploy_tinkercloud.sh --profile megatron_bridge (cu13 recipe env).",
                backend="megatron_bridge", operation="create_model",
            )

        cfg = evo2_1b_classifier_config(
            base_ckpt_dir=hc.get("base_ckpt_dir", base_model),
            train_jsonl=hc.get("train_jsonl"), val_jsonl=hc.get("val_jsonl"),
            test_jsonl=hc.get("test_jsonl"),
            num_classes=num_labels,
            result_dir=checkpoint_path or hc.get("result_dir", f"/data/{model_id}"),
            experiment_name=model_id, model_size=hc.get("model_size", "evo2_1b_base"),
            tensor_model_parallel_size=(parallelism or {}).get("tp", 1),
            seq_length_tokens=seq_length, backbone_seq_length=seq_length,
            train_iters=hc.get("train_iters", 1000),
            global_batch_size=hc.get("global_batch_size", 32),
            micro_batch_size=hc.get("micro_batch_size", 8),
            lr=hc.get("lr", 5e-4), min_lr=hc.get("min_lr", 5e-5),
            warmup_iters=hc.get("warmup_iters", 30),
            pool=hc.get("pool", "mean"), classifier_dropout=hc.get("classifier_dropout", 0.1),
            use_lora=True, lora_dim=lc.get("rank", 16), lora_alpha=lc.get("alpha", 32),
            lora_dropout=lc.get("dropout", 0.1),
            tokenizer_path=hc.get("tokenizer_path", _TOKENIZER_PATH),
        )
        state = GlobalState()
        state.cfg = cfg
        so = setup(state, get_dataset_provider(cfg.dataset))
        handle.state = state
        handle.model = so.model
        handle.optimizer = so.optimizer
        handle.scheduler = so.scheduler
        handle.fbfunc = get_forward_backward_func()
        handle.forward_step = functools.partial(classifier_forward_step, state)
        handle.seq_length = cfg.model.seq_length
        handle.needs_zero = True
        logger.info(
            "[%s] megatron_bridge create_model %s: base=%s num_labels=%d seq_len=%d lora_r=%s",
            request_id, model_id, base_model, num_labels, handle.seq_length, lc.get("rank", 16),
        )
        return handle

    async def forward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """Forward-only pass -> per-sample class logits (no grad)."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_to_backend(data, {"seq_length": h.seq_length})
        losses = h.fbfunc(
            forward_step_func=h.forward_step, data_iterator=iter([batch]),
            model=h.model, num_microbatches=1, seq_length=h.seq_length,
            micro_batch_size=batch["input_ids"].shape[0], forward_only=True,
        )
        return {"loss": _reduce(losses, "ce loss"), "accuracy": _reduce(losses, "accuracy")}

    async def forward_backward(
        self, handle: BackendHandle, data: List[Dict], loss_fn: str,
    ) -> Dict[str, Any]:
        """One forward+backward microbatch (accumulates grads; no optim step).

        Grads are zeroed on the first fb after an optimizer step, so N fb calls
        before apply_optimizer_step accumulate (tinker-async friendly)."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_backward_to_backend(
            data, loss_fn, {"seq_length": h.seq_length},
        )
        if h.needs_zero:
            for mc in h.model:
                if hasattr(mc, "zero_grad_buffer"):
                    mc.zero_grad_buffer()
            h.optimizer.zero_grad()
            h.needs_zero = False
        losses = h.fbfunc(
            forward_step_func=h.forward_step, data_iterator=iter([batch]),
            model=h.model, num_microbatches=1, seq_length=h.seq_length,
            micro_batch_size=batch["input_ids"].shape[0], forward_only=False,
        )
        loss = _reduce(losses, "ce loss")
        logger.debug("megatron_bridge fb: %d samples loss=%.4f", len(data), loss)
        return {"loss": loss, "accuracy": _reduce(losses, "accuracy")}

    async def apply_optimizer_step(
        self, handle: BackendHandle, learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """optimizer.step() + scheduler.step() over the accumulated grads."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        step_out = h.optimizer.step()
        # megatron optimizer.step() -> (update_successful, grad_norm, num_zeros)
        grad_norm = None
        if isinstance(step_out, (tuple, list)) and len(step_out) >= 2:
            grad_norm = step_out[1]
        h.scheduler.step(increment=1)
        h.needs_zero = True
        lr = h.optimizer.param_groups[0]["lr"] if getattr(h.optimizer, "param_groups", None) else None
        return {
            "grad_norm": float(grad_norm) if grad_norm is not None else None,
            "learning_rate": float(lr) if lr is not None else learning_rate,
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
        """Save the LoRA adapters + head via megatron-bridge dist-checkpointing."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        try:
            from megatron.bridge.training.checkpointing import save_checkpoint as _save
            _save(state=h.state, model=h.model, optimizer=h.optimizer,
                  opt_param_scheduler=h.scheduler, num_floating_point_operations_so_far=0)
        except Exception as e:  # pragma: no cover
            raise BackendError(f"save_checkpoint failed: {e!r}",
                               backend="megatron_bridge", operation="save_checkpoint")
        return checkpoint_path

    async def load_checkpoint(
        self, handle: BackendHandle, checkpoint_path: str,
    ) -> None:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        try:
            from megatron.bridge.training.checkpointing import load_checkpoint as _load
            _load(state=h.state, model=h.model, optimizer=h.optimizer,
                  opt_param_scheduler=h.scheduler, checkpoint_path=checkpoint_path)
        except Exception as e:  # pragma: no cover
            raise BackendError(f"load_checkpoint failed: {e!r}",
                               backend="megatron_bridge", operation="load_checkpoint")

    async def delete_model(self, handle: BackendHandle) -> None:
        """Release the megatron model + optimizer for this handle."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        h.model = None
        h.optimizer = None
        h.scheduler = None
        h.state = None
        logger.info("megatron_bridge model %s deleted", h.model_id)


def _reduce(losses, key: str) -> float:
    """Reduce megatron forward_backward_func reporting dicts: sum(val)/sum(count).

    classifier loss_fn reports {key: [value_sum, count]} per microbatch."""
    if not losses:
        return float("nan")
    num = den = 0.0
    for d in losses:
        if isinstance(d, dict) and key in d:
            v = d[key]
            num += float(v[0]); den += float(v[1])
    return num / den if den else float("nan")


def _objective(handle: BackendHandle) -> str:
    return getattr(handle, "objective", Objective.SEQUENCE_CLASSIFICATION.value)


def _no_generation(operation: str) -> BackendError:
    return BackendError(
        "only valid for the language_modeling objective (no generation plane)",
        backend="megatron_bridge", operation=operation,
    )
