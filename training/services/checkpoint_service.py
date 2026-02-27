"""
Checkpoint Service - Business Logic for Model Checkpointing

Handles:
- Saving model weights to disk (delegates to backend)
- Saving weights for SGLang sampler
- Checkpoint metadata management
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..backends.base import TrainingBackend
from ..storage import MetadataStorage
from ..utils.helpers import generate_step_id

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for managing model checkpoints and weights."""

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    async def save_weights(
        self,
        model_id: str,
        request_id: str,
        path: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
    ) -> Dict[str, Any]:
        """
        Save model weights to disk.

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        handle = client_info["backend_handle"]
        training_run_id = client_info["training_run_id"]

        # Generate checkpoint name and step_id
        checkpoint_name = path or f"checkpoint_{int(time.time())}"
        step_id = generate_step_id(checkpoint_name)
        checkpoint_path = f"tinker://{training_run_id}/weights/{checkpoint_name}"

        logger.info("[%s] Saving weights for %s to %s", request_id, model_id, checkpoint_path)

        # Delegate actual save to backend
        await self.backend.save_checkpoint(handle, checkpoint_path, step_id=step_id)

        # Save checkpoint metadata
        metadata_storage.save_checkpoint(
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_data={
                "path": checkpoint_path,
                "created_at": datetime.now().isoformat(),
                "type": "manual_save",
            },
        )

        logger.info("[%s] Weights saved successfully", request_id)

        return {
            "path": checkpoint_path,
            "checkpoint_path": checkpoint_path,
            "step_id": step_id,
            "name": checkpoint_name,
            "type": "save_weights",
        }

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: Optional[str],
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
        path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Save weights for SGLang sampler.

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        handle = client_info["backend_handle"]
        training_run_id = client_info.get("training_run_id", model_id)

        # Check if this is an ephemeral save (sampling_session_seq_id provided, no path/name)
        is_ephemeral = sampling_session_seq_id is not None and path is None and name is None
        logger.info(
            "[%s] save_weights_for_sampler: sampling_session_seq_id=%s, path=%r, name=%r, is_ephemeral=%s",
            request_id, sampling_session_seq_id, path, name, is_ephemeral,
        )

        if is_ephemeral:
            # Ephemeral save — generate sampling_session_id, don't persist path.
            # Skip save_model: weights are already synced to SGLang via update_weights().
            sampling_session_id = f"{model_id}_{sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
            logger.info("[%s] Ephemeral save for sampler: %s -> %s", request_id, model_id, sampling_session_id)
            logger.info("[%s] Skipping save_model for ephemeral save (weights already in SGLang)", request_id)

            return {
                "path": None,
                "sampling_session_id": sampling_session_id,
                "type": "save_weights_for_sampler",
            }
        else:
            # Persistent save — use path/name
            logger.info("[%s] Saving weights for sampler: %s", request_id, model_id)

            checkpoint_name = path or name or f"sampler_{int(time.time())}"
            step_id = generate_step_id(checkpoint_name)
            checkpoint_path = f"/data/checkpoints/tinker/iter_{step_id:07d}"
            tinker_uri = f"tinker://{training_run_id}/weights/{checkpoint_name}"

            # Delegate actual save to backend
            await self.backend.save_checkpoint(handle, checkpoint_path, step_id=step_id)

            # Save checkpoint metadata
            metadata_storage.save_checkpoint(
                model_id=model_id,
                checkpoint_name=f"sampler_{checkpoint_name}",
                checkpoint_data={
                    "path": checkpoint_path,
                    "tinker_uri": tinker_uri,
                    "created_at": datetime.now().isoformat(),
                    "type": "sampler",
                    "step_id": step_id,
                },
            )

            logger.info("[%s] Weights saved to %s", request_id, tinker_uri)

            return {
                "path": tinker_uri,
                "sampling_session_id": None,
                "checkpoint_path": checkpoint_path,
                "step_id": step_id,
                "name": checkpoint_name,
                "type": "save_weights_for_sampler",
            }
