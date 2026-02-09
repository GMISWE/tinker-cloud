"""
Model Service - Business Logic for Model Management

Handles:
- Model creation via TrainingBackend abstraction
- Model deletion (cleanup GPU resources)
- Model metadata retrieval

All model lifecycle operations delegate to the TrainingBackend instance.
"""
import asyncio
import logging
import os
import ray
from datetime import datetime
from typing import Dict, Any, Optional

from ..backends.base import TrainingBackend, BackendHandle
from ..storage import MetadataStorage
from ..utils.model_config import extract_model_name, detect_architecture

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model lifecycle and resources."""

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

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
        slime_builder: Any,  # kept for signature compat; unused — backend has its own builder
        metadata_storage: MetadataStorage,
        training_clients: Dict[str, Dict[str, Any]],
        training_runs_metadata: Dict[str, Dict[str, Any]],
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new training model via the backend abstraction.

        Returns:
            Dict with model_id, base_model, lora_config, status
        """
        logger.info("[%s] Creating model %s", request_id, model_id)

        num_gpus = int(os.environ.get("SLIME_NUM_GPUS", "4"))
        if parallelism_config:
            num_gpus = parallelism_config.get("num_gpus", num_gpus)

        # Delegate to backend
        handle = await self.backend.create_model(
            model_id=model_id,
            request_id=request_id,
            base_model=base_model,
            num_gpus=num_gpus,
            lora_config=lora_config,
            parallelism=parallelism_config,
            debug_train_only=debug_train_only,
            checkpoint_path=checkpoint_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rlve_config=rlve_config,
            wandb_config=wandb_config,
        )

        # Save metadata
        hf_path = getattr(handle, "hf_path", "")
        training_run_id = model_id
        metadata = {
            "training_run_id": model_id,
            "model_id": model_id,
            "base_model": base_model,
            "hf_path": hf_path,
            "lora_config": lora_config,
            "rlve_config": rlve_config,
            "wandb_config": wandb_config,
            "created_at": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "model_owner": "kgateway-user",
            "is_lora": lora_config is not None,
            "lora_rank": lora_config.get("rank", 0) if lora_config else 0,
            "is_rlve": rlve_config is not None and rlve_config.get("enabled", False),
            "corrupted": False,
            "last_request_time": datetime.now().isoformat(),
            "last_checkpoint": None,
            "last_sampler_checkpoint": None,
        }
        metadata_storage.save_training_run(model_id, metadata)
        training_runs_metadata[model_id] = metadata

        # Store client info — includes backend handle + legacy fields from handle
        # for backward compat with routers that read train_group/args/etc.
        client_info = {
            "backend_handle": handle,
            "training_run_id": training_run_id,
            "hf_path": hf_path,
            "base_model": base_model,
            "router_ip": getattr(handle, "router_ip", None),
            "router_port": getattr(handle, "router_port", None),
            "rlve_config": rlve_config,
            "wandb_config": wandb_config,
            "created_at": datetime.now().isoformat(),
            # Legacy Miles fields — populated from handle for router compat
            "train_group": getattr(handle, "train_group", None),
            "rollout_manager": getattr(handle, "rollout_manager", None),
            "placement_group": getattr(handle, "placement_group", None),
            "args": getattr(handle, "args", None),
        }
        training_clients[model_id] = client_info

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
        handle = client_info.get("backend_handle")

        if handle is not None:
            await self.backend.delete_model(handle)
        else:
            # Legacy fallback (should not happen after Phase 2)
            logger.warning("No backend handle for %s — using legacy cleanup", model_id)
            self._legacy_delete(client_info)

        del training_clients[model_id]

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

    @staticmethod
    def _legacy_delete(client_info: Dict[str, Any]) -> None:
        """Fallback cleanup for pre-backend client_info dicts."""
        train_group = client_info.get("train_group")
        if train_group:
            for actor in train_group._actor_handlers:
                ray.kill(actor, no_restart=True)
        rm = client_info.get("rollout_manager")
        if rm is not None:
            ray.kill(rm, no_restart=True)
        pg = client_info.get("placement_group")
        if pg is not None:
            ray.util.remove_placement_group(pg)

    def get_model_info(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get model info for tokenizer initialization."""
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        args = client_info.get("args")

        if args is not None:
            model_name = extract_model_name(args)
            arch = detect_architecture(model_name)
            is_lora = getattr(args, "lora_rank", 0) > 0
            lora_rank = args.lora_rank if is_lora else None
        else:
            # Non-Miles backend — use base_model from metadata
            model_name = client_info.get("base_model", "unknown")
            arch = detect_architecture(model_name)
            is_lora = False
            lora_rank = None

        return {
            "model_id": model_id,
            "model_data": {"arch": arch, "model_name": model_name},
            "is_lora": is_lora,
            "lora_rank": lora_rank,
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
