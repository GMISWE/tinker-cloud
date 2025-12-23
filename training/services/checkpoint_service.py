"""
Checkpoint Service - Business Logic for Model Checkpointing

Handles:
- Saving model weights to disk
- Saving weights for SGLang sampler
- Checkpoint metadata management
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..storage import MetadataStorage
from ..utils.helpers import generate_step_id

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for managing model checkpoints and weights."""

    def __init__(self):
        """Initialize CheckpointService."""
        pass

    async def save_weights(
        self,
        model_id: str,
        request_id: str,
        path: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage
    ) -> Dict[str, Any]:
        """
        Save model weights to disk.

        Args:
            model_id: Model identifier
            request_id: Request identifier for logging
            path: Optional checkpoint name/path
            training_clients: Global training clients dict
            metadata_storage: Metadata storage instance

        Returns:
            Dict with checkpoint_path

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        train_group = client_info["train_group"]
        training_run_id = client_info["training_run_id"]

        # Generate checkpoint name and step_id
        checkpoint_name = path or f"checkpoint_{int(time.time())}"
        step_id = generate_step_id(checkpoint_name)
        checkpoint_path = f"tinker://{training_run_id}/weights/{checkpoint_name}"

        logger.info(f"[{request_id}] Saving weights for {model_id} to {checkpoint_path}")

        # Save weights using async Ray API (matching original api.py pattern)
        # Call save_model.remote() on each actor directly
        object_refs = [
            actor.save_model.remote(step_id)
            for actor in train_group._actor_handlers
        ]

        # Await all save operations
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])

        # Save checkpoint metadata
        metadata_storage.save_checkpoint(
            model_id=model_id,
            checkpoint_name=checkpoint_name,
            checkpoint_data={
                "path": checkpoint_path,
                "created_at": datetime.now().isoformat(),
                "type": "manual_save"
            }
        )

        logger.info(f"[{request_id}] Weights saved successfully")

        # Return format matching original API (for backward compatibility)
        return {
            "path": checkpoint_path,  # Tinker URI format
            "checkpoint_path": checkpoint_path,  # Keep for internal use
            "step_id": step_id,
            "name": checkpoint_name,
            "type": "save_weights"
        }

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: Optional[str],
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
        path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Save weights for SGLang sampler.

        Args:
            model_id: Model identifier
            request_id: Request identifier for logging
            name: Optional checkpoint name (deprecated, use path)
            training_clients: Global training clients dict
            metadata_storage: Metadata storage instance
            path: Optional checkpoint path/name
            sampling_session_seq_id: If provided, creates ephemeral save with sampling_session_id

        Returns:
            Dict with either:
            - path, checkpoint_path, step_id, name (persistent save)
            - sampling_session_id (ephemeral save)

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        train_group = client_info["train_group"]
        training_run_id = client_info.get("training_run_id", model_id)

        # Check if this is an ephemeral save (sampling_session_seq_id provided, no path/name)
        is_ephemeral = sampling_session_seq_id is not None and path is None and name is None
        logger.info(f"[{request_id}] save_weights_for_sampler: sampling_session_seq_id={sampling_session_seq_id}, path={path!r}, name={name!r}, is_ephemeral={is_ephemeral}")

        # NOTE: Save must be called while process groups are active.
        # In the training flow: train_step() -> save_weights_for_sampler() -> update_weights()
        # Do NOT call save after update_weights has destroyed process groups.
        # The NCCL library cannot reliably recreate process groups after destruction.

        if is_ephemeral:
            # Ephemeral save - generate sampling_session_id, don't persist path
            sampling_session_id = f"{model_id}_{sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
            logger.info(f"[{request_id}] Ephemeral save for sampler: {model_id} -> {sampling_session_id}")

            # Check if offload_train is enabled
            args = client_info.get("args")
            offload_train = args.offload_train if args else False

            if offload_train:
                # With offload_train=True, process groups are destroyed after update_weights().
                # But for ephemeral saves, we don't need to actually save to disk - the weights
                # are already synced to SGLang via update_weights(). Just return the session ID.
                logger.info(f"[{request_id}] Skipping save_model for ephemeral save (offload_train=True, weights already synced to SGLang)")
            else:
                # Generate step_id for the save
                step_id = generate_step_id(f"ephemeral_{sampling_session_seq_id}")

                # Save weights using async Ray API
                object_refs = [
                    actor.save_model.remote(step_id)
                    for actor in train_group._actor_handlers
                ]

                # Await all save operations
                await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])
                logger.info(f"[{request_id}] Ephemeral weights saved to disk")

            logger.info(f"[{request_id}] Ephemeral save completed, session_id={sampling_session_id}")

            return {
                "path": None,
                "sampling_session_id": sampling_session_id,
                "type": "save_weights_for_sampler"
            }
        else:
            # Persistent save - use path/name
            logger.info(f"[{request_id}] Saving weights for sampler: {model_id}")

            # Check if offload_train is enabled
            args = client_info.get("args")
            offload_train = args.offload_train if args else False

            if offload_train:
                # With offload_train=True, process groups may be destroyed after update_weights().
                # Persistent saves require PG for distributed checkpointing.
                # Log a warning - the save may fail if PG are not active.
                logger.warning(
                    f"[{request_id}] Persistent save with offload_train=True - "
                    "this may fail if process groups are destroyed. "
                    "Consider saving before update_weights or using ephemeral saves."
                )

            # Generate checkpoint name and path
            checkpoint_name = path or name or f"sampler_{int(time.time())}"
            step_id = generate_step_id(checkpoint_name)
            checkpoint_path = f"/data/checkpoints/tinker/iter_{step_id:07d}"
            tinker_uri = f"tinker://{training_run_id}/weights/{checkpoint_name}"

            # Save weights using async Ray API (matching original api.py pattern)
            object_refs = [
                actor.save_model.remote(step_id)
                for actor in train_group._actor_handlers
            ]

            # Await all save operations
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])

            # Save checkpoint metadata
            metadata_storage.save_checkpoint(
                model_id=model_id,
                checkpoint_name=f"sampler_{checkpoint_name}",
                checkpoint_data={
                    "path": checkpoint_path,
                    "tinker_uri": tinker_uri,
                    "created_at": datetime.now().isoformat(),
                    "type": "sampler",
                    "step_id": step_id
                }
            )

            logger.info(f"[{request_id}] Weights saved to {tinker_uri}")

            return {
                "path": tinker_uri,
                "sampling_session_id": None,
                "checkpoint_path": checkpoint_path,
                "step_id": step_id,
                "name": checkpoint_name,
                "type": "save_weights_for_sampler"
            }
