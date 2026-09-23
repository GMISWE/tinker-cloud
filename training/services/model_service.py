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
from typing import Dict, Any, Optional

from ..backends.base import TrainingBackend
from ..checkpoints import CheckpointStore
from ..core import routing
from ..storage import MetadataStorage
from ..storage.sample_futures import SampleFutureStore
from ..utils.model_config import detect_architecture, detect_num_gpus

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model lifecycle and resources."""

    def __init__(self, backend: TrainingBackend, store: CheckpointStore, sample_futures: SampleFutureStore):
        self.backend = backend
        self.store = store
        self.sample_futures = sample_futures

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        lora_config: Optional[Dict[str, Any]],
        debug_train_only: bool,
        checkpoint_path: Optional[str],
        parallelism_config: Optional[Dict[str, Any]],
        max_batch_size: int,
        max_seq_len: int,
        metadata_storage: MetadataStorage,
        training_clients: Dict[str, Dict[str, Any]],
        training_runs_metadata: Dict[str, Dict[str, Any]],
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        staleness_k: int = 0,
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new training model via the backend abstraction.

        Returns:
            Dict with model_id, base_model, lora_config, status
        """
        logger.info("[%s] Creating model %s", request_id, model_id)

        num_gpus = detect_num_gpus()
        if parallelism_config:
            num_gpus = parallelism_config.get("num_gpus", num_gpus)

        # The URI is an identity; the backend gets the resolved directory of a
        # completed weights checkpoint, and its own private area for this model.
        resume_from = self.store.resolve_resume(checkpoint_path) if checkpoint_path else None
        native_root = self.store.native_root(model_id)

        # Delegate to backend. The objective axis (feature 004) is forwarded so
        # classification backends can stand up a classification head; LM-only
        # backends reject non-LM objectives. See specs/004-bionemo-classification.
        handle = await self.backend.create_model(
            model_id=model_id,
            request_id=request_id,
            base_model=base_model,
            num_gpus=num_gpus,
            lora_config=lora_config,
            parallelism=parallelism_config,
            debug_train_only=debug_train_only,
            resume_from=resume_from,
            native_root=native_root,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rlve_config=rlve_config,
            wandb_config=wandb_config,
            staleness_k=staleness_k,
            objective=objective,
            num_labels=num_labels,
            head_config=head_config,
        )

        # Save metadata
        hf_path = handle.hf_path
        training_run_id = model_id
        metadata = {
            "training_run_id": model_id,
            "model_id": model_id,
            "base_model": base_model,
            "hf_path": hf_path,
            "lora_config": lora_config,
            "rlve_config": rlve_config,
            "wandb_config": wandb_config,
            "objective": objective,
            "num_labels": num_labels,
            "created_at": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "model_owner": "kgateway-user",
            "is_lora": bool(lora_config and lora_config.get("rank", 0) > 0),
            "lora_rank": lora_config.get("rank", 0) if lora_config else 0,
            "is_rlve": rlve_config is not None and rlve_config.get("enabled", False),
            "corrupted": False,
            "last_request_time": datetime.now().isoformat(),
            "last_checkpoint": None,
            "last_sampler_checkpoint": None,
        }
        metadata_storage.save_training_run(model_id, metadata)
        training_runs_metadata[model_id] = metadata

        # Backend-agnostic record; engine state stays behind the handle.
        client_info = {
            "backend_handle": handle,
            "training_run_id": training_run_id,
            "hf_path": hf_path,
            "base_model": base_model,
            "lora_config": lora_config,
            "rlve_config": rlve_config,
            "wandb_config": wandb_config,
            "created_at": datetime.now().isoformat(),
        }
        training_clients[model_id] = client_info
        if handle.inference_endpoint is not None:
            routing.table.publish(model_id, routing.InferenceEndpoint(handle.inference_endpoint))

        logger.info("[%s] Model %s created successfully", request_id, model_id)
        return {
            "model_id": model_id,
            "base_model": base_model,
            "lora_config": lora_config,
            "status": "ready",
        }

    async def delete_model(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
    ) -> Dict[str, Any]:
        """Delete training client and release GPU resources."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        await self.backend.delete_model(client_info["backend_handle"])

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

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

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
