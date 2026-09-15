"""
Checkpoint Service - saves, loads, listing and deletion.

The service orchestrates: the store (training/checkpoints) decides where a
checkpoint lives and records what happened to it; the backend writes and
reads directories it is handed. Every save is pending from begin_save until
the backend returns, then completed or failed.
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from ..backends.base import TrainingBackend
from ..checkpoints import CheckpointKind, CheckpointRef, CheckpointStore
from ..storage import MetadataStorage

logger = logging.getLogger(__name__)


class CheckpointService:
    """Service for managing model checkpoints and weights."""

    def __init__(self, backend: TrainingBackend, store: CheckpointStore):
        self.backend = backend
        self.store = store

    @staticmethod
    def _client(training_clients: Dict[str, Dict[str, Any]], model_id: str) -> Dict[str, Any]:
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")
        return training_clients[model_id]

    async def _save(self, handle, ticket) -> None:
        try:
            await self.backend.save_checkpoint(
                handle, ticket.root, step=ticket.step, persist=ticket.step is not None,
            )
        except Exception as e:
            self.store.fail(ticket.ref, str(e))
            raise
        self.store.complete(ticket.ref)

    async def save_weights(
        self,
        model_id: str,
        request_id: str,
        path: Optional[str],
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Save a training checkpoint (weights + whatever optimizer state the
        backend can write) under the name the client chose."""
        client_info = self._client(training_clients, model_id)
        handle = client_info["backend_handle"]
        name = path or f"checkpoint_{int(time.time())}"
        ticket = self.store.begin_save(
            model_id, CheckpointKind.WEIGHTS, name,
            weight_version=handle.weight_version,
        )
        logger.info("[%s] Saving weights for %s to %s (step %s)", request_id, model_id, ticket.ref.uri, ticket.step)
        await self._save(handle, ticket)
        logger.info("[%s] Weights saved: %s", request_id, ticket.ref.uri)
        return {"path": ticket.ref.uri, "step": ticket.step, "name": name, "type": "save_weights"}

    async def load_weights(
        self,
        model_id: str,
        request_id: str,
        ref: CheckpointRef,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage,
        optimizer: bool = False,
    ) -> Dict[str, Any]:
        """Load a completed training checkpoint into a live model: weights, plus
        optimizer state when asked (the backend refuses rather than partially
        resumes)."""
        handle = self._client(training_clients, model_id)["backend_handle"]
        root = self.store.require(ref, kind=CheckpointKind.WEIGHTS)
        logger.info("[%s] Loading %s for %s from %s", request_id,
                    "weights + optimizer state" if optimizer else "weights", model_id, ref.uri)
        await self.backend.load_checkpoint(handle, root, optimizer=optimizer)
        metadata_storage.update_training_run(
            model_id, {"loaded_from": ref.uri, "last_request_time": datetime.now().isoformat()},
        )
        return {"type": "load_weights", "path": ref.uri, "model_id": model_id}

    def delete_checkpoint(self, ref: CheckpointRef) -> bool:
        """Remove a checkpoint's record and bytes. False if unknown."""
        return self.store.delete(ref)

    def list_checkpoints(self, model_id: str) -> list:
        """Tinker-shaped checkpoint records for a training run."""
        out = []
        for rec in self.store.list(model_id):
            kind = CheckpointKind(rec["kind"])
            out.append({
                "checkpoint_id": f"{kind.value}/{rec['name']}",
                "checkpoint_type": "sampler" if kind is CheckpointKind.SAMPLER_WEIGHTS else "training",
                "time": rec.get("created_at"),
                "tinker_path": rec["uri"],
                "size_bytes": rec.get("size_bytes"),
                "status": rec.get("status"),
            })
        return out

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: Optional[str],
        training_clients: Dict[str, Dict[str, Any]],
        path: Optional[str] = None,
        sampling_session_seq_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Save sampler weights. With a sampling_session_seq_id and no name the
        save is ephemeral: the weights already reached the inference engine, so
        the backend writes nothing and the record only names the sampler; the
        client gets path=None and a sampling_session_id, as the SDK expects."""
        client_info = self._client(training_clients, model_id)
        handle = client_info["backend_handle"]
        ephemeral = sampling_session_seq_id is not None and path is None and name is None
        weight_version = handle.weight_version

        if ephemeral:
            name = f"{model_id}_{sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
            ticket = self.store.begin_save(
                model_id, CheckpointKind.SAMPLER_WEIGHTS, name, persist=False, weight_version=weight_version,
            )
            logger.info("[%s] Ephemeral sampler save for %s: %s", request_id, model_id, ticket.ref.uri)
            await self._save(handle, ticket)
            return {
                "path": None, "sampling_session_id": name, "uri": ticket.ref.uri,
                "type": "save_weights_for_sampler",
            }

        name = path or name or f"sampler_{int(time.time())}"
        ticket = self.store.begin_save(
            model_id, CheckpointKind.SAMPLER_WEIGHTS, name, weight_version=weight_version,
        )
        logger.info("[%s] Saving sampler weights for %s to %s (step %s)", request_id, model_id, ticket.ref.uri, ticket.step)
        await self._save(handle, ticket)
        logger.info("[%s] Sampler weights saved: %s", request_id, ticket.ref.uri)
        return {
            "path": ticket.ref.uri, "sampling_session_id": None, "uri": ticket.ref.uri,
            "step": ticket.step, "name": name, "type": "save_weights_for_sampler",
        }
