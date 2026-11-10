"""
Model Service - Business Logic for Model Management

Handles:
- Model creation via TrainingBackend abstraction
- Model deletion (cleanup GPU resources)
- Model metadata retrieval

All model lifecycle operations delegate to the TrainingBackend instance.
"""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple

from ..backends.base import BackendHandle, TrainingBackend
from ..models.requests import OptimizerConfig
from ..checkpoints import CheckpointStore
from ..core import routing
from ..storage import MetadataStorage
from ..storage.sample_futures import SampleFutureStore
from ..utils.model_config import check_model_code, detect_architecture, detect_num_gpus, remote_code_trusted

logger = logging.getLogger(__name__)


class DeleteInProgress(Exception):
    """A delete of this model is already running; this one did not start."""


class ModelService:
    """Service for managing ML model lifecycle and resources."""

    def __init__(self, backend: TrainingBackend, store: CheckpointStore, sample_futures: SampleFutureStore):
        self.backend = backend
        self.store = store
        self.sample_futures = sample_futures
        self._deleting: Set[str] = set()   # model ids whose backend delete is running

    def require_optimizer(self, optimizer_config: OptimizerConfig) -> None:
        """Raises ValueError (a 400) naming the backend when it does not serve
        the family; the router runs this before the create is queued."""
        if optimizer_config.type not in self.backend.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer family {optimizer_config.type!r} is not served by {type(self.backend).__name__}; "
                f"{', '.join(sorted(self.backend.SUPPORTED_OPTIMIZERS))} are")

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        lora_config: Optional[Dict[str, Any]],
        parallelism_config: Optional[Dict[str, Any]],
        metadata_storage: MetadataStorage,
        training_clients: Dict[str, Dict[str, Any]],
        training_runs_metadata: Dict[str, Dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> Dict[str, Any]:
        """
        Create a new training model via the backend abstraction.

        `optimizer_config` passed `require_optimizer`; its record form is
        stored with the model.

        Returns:
            Dict with model_id, base_model, lora_config, status
        """
        logger.info("[%s] Creating model %s", request_id, model_id)

        # Whichever backend loads the model, its repository's code runs only for a model the operator trusts.
        await check_model_code(self.backend, base_model)

        num_gpus = detect_num_gpus()
        if parallelism_config:
            num_gpus = parallelism_config.get("num_gpus", num_gpus)

        # The backend reads everything else from the model's own config.json
        # and its backend configuration (D17). The URI is an identity; the
        # backend gets its own private area for this model, and weights arrive
        # through load_weights (D9), never at create.
        native_root = self.store.native_root(model_id)

        handle = await self.backend.create_model(
            model_id=model_id,
            request_id=request_id,
            base_model=base_model,
            num_gpus=num_gpus,
            lora_config=lora_config,
            parallelism=parallelism_config,
            native_root=native_root,
            optimizer=optimizer_config.type,
        )
        # The backend holds this model's resources (a trainer, a tenant, GPU
        # memory) from here on. Nothing else knows the handle until it is in
        # training_clients, and the reaper and delete_model look only there, so
        # a step that fails before that point, or after it, would strand it.
        # Whatever raises, undo all of it once and report the original error.
        # delete_model keeps a model's training-run record for resume, and a
        # create under the same id (create-via-load) overwrites it. Hold the
        # earlier record so a failed create puts it back and never deletes it.
        earlier_record = metadata_storage.load_training_run(model_id)
        record_written = False
        try:
            optimizer_record = optimizer_config.record()

            # Save metadata
            hf_path = handle.hf_path
            training_run_id = model_id
            metadata = {
                "training_run_id": model_id,
                "model_id": model_id,
                "base_model": base_model,
                "hf_path": hf_path,
                "lora_config": lora_config,
                "optimizer_config": optimizer_record,
                "objective": handle.objective,
                "num_labels": handle.num_labels,
                "created_at": datetime.now().isoformat(),
                "model_owner": "kgateway-user",
                "is_lora": bool(lora_config and lora_config.get("rank", 0) > 0),
                "lora_rank": lora_config.get("rank", 0) if lora_config else 0,
                "corrupted": False,
                "last_request_time": datetime.now().isoformat(),
                "last_checkpoint": None,
                "last_sampler_checkpoint": None,
            }
            record_written = True   # set first: a save that fails halfway may still have touched the file
            metadata_storage.save_training_run(model_id, metadata)
            training_runs_metadata[model_id] = metadata

            # Backend-agnostic record; engine state stays behind the handle.
            client_info = {
                "backend_handle": handle,
                "training_run_id": training_run_id,
                "hf_path": hf_path,
                "base_model": base_model,
                "lora_config": lora_config,
                "optimizer_config": optimizer_record,
                "created_at": datetime.now().isoformat(),
            }
            training_clients[model_id] = client_info
            if handle.inference_endpoints:
                routing.table.publish(model_id, routing.InferenceEndpoint(handle.inference_endpoints))

        except Exception:
            await self._undo_failed_create(handle, request_id, model_id, metadata_storage,
                                           training_clients, training_runs_metadata,
                                           earlier_record if record_written else None, record_written)
            raise

        logger.info("[%s] Model %s created successfully", request_id, model_id)
        return {
            "model_id": model_id,
            "base_model": base_model,
            "lora_config": lora_config,
            "status": "ready",
        }

    async def _undo_failed_create(
        self,
        handle: BackendHandle,
        request_id: str,
        model_id: str,
        metadata_storage: MetadataStorage,
        training_clients: Dict[str, Dict[str, Any]],
        training_runs_metadata: Dict[str, Dict[str, Any]],
        earlier_record: Optional[Dict[str, Any]],
        record_written: bool,
    ) -> None:
        """Take back what a create that is about to fail had recorded, then free what the backend allocated.

        Each step may or may not have run, so each tolerates its absence. If the
        create wrote the training-run record, the record is put back as it was,
        or removed when there was none, so list_training_runs does not list a run
        whose create failed and a record kept for resume is not lost. A create
        that failed before it wrote leaves the record alone. A step that fails
        here is logged: the create's own error is the one to report. If the
        backend cannot release the handle, no registry holds it any more, so
        that leak is accepted and only the log shows it.
        """
        training_clients.pop(model_id, None)
        training_runs_metadata.pop(model_id, None)
        routing.table.withdraw(model_id)
        if record_written:
            try:
                if earlier_record is None:
                    metadata_storage.delete_training_run(model_id)
                else:
                    metadata_storage.restore_training_run(model_id, earlier_record)
            except Exception:
                logger.exception("[%s] Could not restore the training-run record of %s", request_id, model_id)
        try:
            await self.backend.delete_model(handle)
        except Exception:
            logger.exception("[%s] Could not release %s after its create failed", request_id, model_id)

    async def delete_model(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
    ) -> Dict[str, Any]:
        """Delete training client and release GPU resources. One delete per model runs at a time:
        another that arrives while it runs raises DeleteInProgress and never reaches the backend,
        whose teardown would otherwise run twice (the model leaves `training_clients` only after it)."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")
        if model_id in self._deleting:
            raise DeleteInProgress(f"Model {model_id}: a delete is in progress")

        client_info = training_clients[model_id]
        self._deleting.add(model_id)
        try:
            await self.backend.delete_model(client_info["backend_handle"])
        finally:
            self._deleting.discard(model_id)

        # Registry first, route second: a sample resolves handle and route in
        # one synchronous stretch, so it sees "model gone", never a bare
        # "router not available"; a failed delete above leaves both intact.
        del training_clients[model_id]
        routing.table.withdraw(model_id)
        n = self.sample_futures.cancel_model(model_id)   # whatever was still in flight fails as "model deleted"
        if n:
            logger.info("Cancelled %d in-flight sample(s) of %s", n, model_id)
        # ephemeral sampler records die with the model; persistent checkpoints
        # and the native area stay for a later resume
        self.store.release_model(model_id)

        if "training_run_id" in client_info:
            metadata_storage.update_training_run(
                client_info["training_run_id"],
                {"last_request_time": datetime.now().isoformat()},
            )

        logger.info("Deleted model %s", model_id)
        return {
            "model_id": model_id,
            "message": "Training client resources freed, metadata preserved for resume",
            "resources_freed": ["backend_resources"],
        }

    def get_model_info(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get model info for tokenizer initialization."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]

        # Backend-agnostic: the LoRA shape and model name are what the client
        # asked for at create time (the engine's local checkpoint path is not
        # the client's business).
        lora_rank = int((client_info.get("lora_config") or {}).get("rank") or 0)
        is_lora = lora_rank > 0
        model_name = client_info["base_model"]
        arch = detect_architecture(model_name)

        return {
            "model_id": model_id,
            "model_data": {"arch": arch, "model_name": model_name},
            "is_lora": is_lora,
            "lora_rank": lora_rank if is_lora else None,
            "model_name": model_name,
            "optimizer_config": client_info["optimizer_config"],
        }

    def get_tokenizer_info(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get tokenizer information from HuggingFace model."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        hf_path = client_info.get("hf_path")

        if not hf_path:
            raise ValueError("HuggingFace path not available")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=remote_code_trusted(hf_path))

        return {
            "vocab_size": len(tokenizer),
            "model_max_length": tokenizer.model_max_length,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token,
                "unk_token": tokenizer.unk_token,
            },
            "hf_checkpoint": hf_path,
        }

    def list_training_runs(
        self, limit: int, offset: int, metadata_storage: MetadataStorage,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """One page of every recorded run, live or deleted, newest first by creation (so
        offsets stay stable while runs train), and the total."""
        runs = metadata_storage.list_training_runs(limit=None)
        runs.sort(key=lambda m: (m["created_at"], m["training_run_id"]), reverse=True)
        return runs[offset:offset + limit], len(runs)

    def get_training_run_metadata(
        self,
        model_id: str,
        metadata_storage: MetadataStorage,
    ) -> Dict[str, Any]:
        """Load persistent training run metadata."""
        metadata = metadata_storage.load_training_run(model_id)
        if not metadata:
            raise KeyError(f"Training run {model_id} not found")

        metadata_storage.update_training_run(
            model_id,
            {"last_request_time": datetime.now().isoformat()},
        )
        return metadata
