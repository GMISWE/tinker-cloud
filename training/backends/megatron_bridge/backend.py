"""
Megatron-Bridge backend — Evo2 (Striped Hyena) sequence classification.

Hosts the faithful BioNeMo CS2 recipe (bionemo-recipes/recipes/evo2_megatron/
examples/evo2_classifier.py) behind the tinker TrainingBackend API.

**Ray-actor architecture** (like nemo_rl / miles; NOT run under torchrun): the
megatron model + distributed context live in a per-model GPU actor
(``MegatronBridgeWorker``), so the tinker-cloud server stays a plain process.
create_model spawns the actor; forward/forward_backward/apply_optimizer_step
delegate to it. Each model gets its own actor -> megatron's global state
(parallel_state / microbatch calculator) is isolated per model (multi-model OK,
clean teardown). The actor decomposes megatron-bridge's ``pretrain()`` into
setup() + get_forward_backward_func() + optimizer.step().

Runs in the cu13 recipe venv (deploy_tinkercloud.sh --profile megatron_bridge);
ray + megatron.bridge + evo2_classifier imports are lazy. Generation plane raises.
See specs/004-bionemo-classification/P5-TINKER-BACKEND.md.
"""
import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend
from ..objectives import Objective, is_classification

logger = logging.getLogger(__name__)

# Recipe locations inside the cu13 recipe env (deploy --profile megatron_bridge).
_RECIPE_DIR = os.environ.get("EVO2_RECIPE_DIR", "/workspace/evo2_megatron")
_RECIPE_EXAMPLES = os.path.join(_RECIPE_DIR, "examples")
_TOKENIZER_PATH = os.environ.get(
    "EVO2_TOKENIZER_PATH", os.path.join(_RECIPE_DIR, "tokenizers", "nucleotide_fast_tokenizer_512")
)


@dataclass
class MegatronBridgeHandle(BackendHandle):
    """Handle for one Evo2 classifier — a reference to its GPU worker actor."""

    base_model: str = ""
    objective: str = Objective.SEQUENCE_CLASSIFICATION.value
    num_labels: int = 0
    head_config: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    worker: Any = None                # MegatronBridgeWorker Ray actor handle
    seq_length: int = 1024
    created_at: str = ""


async def _get(ref):
    """Await a Ray object ref without blocking the event loop."""
    import ray
    return await asyncio.to_thread(ray.get, ref)


class MegatronBridgeBackend(TrainingBackend):
    """Megatron-native classification backend (Ray-actor delegation, no gen plane)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import NeMo2ClassificationDataConverter
            self._converter = NeMo2ClassificationDataConverter()
        return self._converter

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
        """Spawn a GPU worker actor that builds the classifier + LoRA + optimizer."""
        if not is_classification(objective):
            raise BackendError(
                f"Megatron-Bridge backend only serves classification objectives, got {objective!r}",
                backend="megatron_bridge", operation="create_model")
        if not num_labels or num_labels < 2:
            raise BackendError(
                f"num_labels must be >= 2 for classification, got {num_labels!r}",
                backend="megatron_bridge", operation="create_model")

        hc = head_config or {}
        lc = lora_config or {}
        seq_length = int(hc.get("seq_length", max_seq_len if max_seq_len <= 8192 else 1024))
        # kwargs for evo2_1b_classifier_config (built inside the actor; Path-wrapped there)
        cfg_kwargs = dict(
            base_ckpt_dir=hc.get("base_ckpt_dir", base_model),
            train_jsonl=hc.get("train_jsonl"), val_jsonl=hc.get("val_jsonl"),
            test_jsonl=hc.get("test_jsonl"), num_classes=num_labels,
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

        try:
            import ray
            if not ray.is_initialized():
                ray.init(address=os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True)
            from .worker import MegatronBridgeWorker
        except Exception as e:
            raise BackendError(
                f"ray / megatron-bridge recipe not importable ({e!r}). Run under "
                f"deploy_tinkercloud.sh --profile megatron_bridge (cu13 recipe env).",
                backend="megatron_bridge", operation="create_model")

        worker = MegatronBridgeWorker.remote(cfg_kwargs, _RECIPE_EXAMPLES)
        await _get(worker.ready.remote())   # blocks (in a thread) until setup() done

        handle = MegatronBridgeHandle(
            model_id=model_id, backend_type="megatron_bridge", base_model=base_model,
            objective=Objective(objective).value, num_labels=num_labels,
            head_config=head_config, lora_config=lora_config, worker=worker,
            seq_length=seq_length, created_at=datetime.now().isoformat())
        logger.info(
            "[%s] megatron_bridge create_model %s: base=%s num_labels=%d seq_len=%d lora_r=%s (Ray actor)",
            request_id, model_id, base_model, num_labels, seq_length, lc.get("rank", 16))
        return handle

    async def forward(self, handle: BackendHandle, data: List[Dict], loss_fn: str) -> Dict[str, Any]:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_to_backend(data, {"seq_length": h.seq_length})
        return await _get(h.worker.forward.remote(batch))

    async def forward_backward(self, handle: BackendHandle, data: List[Dict], loss_fn: str) -> Dict[str, Any]:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        batch = self.converter.forward_backward_to_backend(data, loss_fn, {"seq_length": h.seq_length})
        return await _get(h.worker.forward_backward.remote(batch))

    async def apply_optimizer_step(self, handle: BackendHandle, learning_rate: Optional[float] = None) -> Dict[str, Any]:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        return await _get(h.worker.apply_optimizer_step.remote(learning_rate))

    # --- generation plane: N/A for classification ---

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        raise _no_generation("update_inference_weights")

    async def sample(self, handle: BackendHandle, request_id: str, prompt_tokens: List[int],
                     num_samples: int, sampling_params: Optional[Dict[str, Any]] = None,
                     prompt_logprobs: bool = False) -> Dict[str, Any]:
        raise _no_generation("sample")

    async def get_logprobs(self, handle: BackendHandle, data: List[Dict]) -> List[Any]:
        raise _no_generation("get_logprobs")

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        raise _no_generation("prepare_for_generation")

    # --- checkpoint / teardown ---

    async def save_checkpoint(self, handle: BackendHandle, checkpoint_path: str,
                              step_id: Optional[int] = None) -> str:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        try:
            return await _get(h.worker.save_checkpoint.remote(checkpoint_path))
        except Exception as e:
            raise BackendError(f"save_checkpoint failed: {e!r}",
                               backend="megatron_bridge", operation="save_checkpoint")

    async def load_checkpoint(self, handle: BackendHandle, checkpoint_path: str) -> None:
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        try:
            await _get(h.worker.load_checkpoint.remote(checkpoint_path))
        except Exception as e:
            raise BackendError(f"load_checkpoint failed: {e!r}",
                               backend="megatron_bridge", operation="load_checkpoint")

    async def delete_model(self, handle: BackendHandle) -> None:
        """Kill the model's GPU actor (frees its megatron state + GPU)."""
        h: MegatronBridgeHandle = handle  # type: ignore[assignment]
        if h.worker is not None:
            import ray
            ray.kill(h.worker)
            h.worker = None
        logger.info("megatron_bridge model %s deleted (actor killed)", h.model_id)


def _objective(handle: BackendHandle) -> str:
    return getattr(handle, "objective", Objective.SEQUENCE_CLASSIFICATION.value)


def _no_generation(operation: str) -> BackendError:
    return BackendError(
        "only valid for the language_modeling objective (no generation plane)",
        backend="megatron_bridge", operation=operation)
