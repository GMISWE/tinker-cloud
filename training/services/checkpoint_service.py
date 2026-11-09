"""
Checkpoint Service - saves, loads, listing and deletion.

The service orchestrates: the store (training/checkpoints) decides where a
checkpoint lives and records what happened to it; the backend writes and
reads directories it is handed. Every save is pending from begin_save until
the backend returns, then completed or failed.
"""
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from ..backends.base import TrainingBackend
from ..checkpoints import CheckpointKind, CheckpointRef, CheckpointSaveOptions, CheckpointStatus, CheckpointStore, is_expired
from ..models.requests import OptimizerConfig
from ..models.responses import Checkpoint, CheckpointType
from ..storage import MetadataStorage

logger = logging.getLogger(__name__)

# store kind -> the SDK's CheckpointType
CHECKPOINT_TYPES: Dict[CheckpointKind, CheckpointType] = {
    CheckpointKind.WEIGHTS: "training",
    CheckpointKind.SAMPLER_WEIGHTS: "sampler",
    CheckpointKind.EXTERNAL_WEIGHTS: "external",
}


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
        options: CheckpointSaveOptions,
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Save a training checkpoint (weights + whatever optimizer state the
        backend can write) under the name the client chose."""
        client_info = self._client(training_clients, model_id)
        handle = client_info["backend_handle"]
        name = path or f"checkpoint_{int(time.time())}"
        ticket = self.store.begin_save(
            model_id, CheckpointKind.WEIGHTS, name,
            options=options, weight_version=handle.weight_version,
            optimizer_config=client_info["optimizer_config"],
        )
        logger.info("[%s] Saving weights for %s to %s (step %s)", request_id, model_id, ticket.ref.uri, ticket.step)
        await self._save(handle, ticket)
        logger.info("[%s] Weights saved: %s", request_id, ticket.ref.uri)
        return {"path": ticket.ref.uri, "step": ticket.step, "name": name, "type": "save_weights"}

    async def save_weights_external(
        self,
        model_id: str,
        request_id: str,
        name: str,
        options: CheckpointSaveOptions,
        training_clients: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Export the weights in HF format under an external_weights checkpoint;
        the files are then served one by one through signed URLs."""
        client_info = self._client(training_clients, model_id)
        handle = client_info["backend_handle"]
        ticket = self.store.begin_save(
            model_id, CheckpointKind.EXTERNAL_WEIGHTS, name,
            options=options, weight_version=handle.weight_version,
            optimizer_config=client_info["optimizer_config"],
        )
        logger.info("[%s] Exporting external weights for %s to %s (step %s)", request_id, model_id, ticket.ref.uri, ticket.step)
        try:
            await self.backend.export_external_weights(handle, ticket.root, ticket.step)
        except Exception as e:
            self.store.fail(ticket.ref, str(e))
            raise
        self.store.complete(ticket.ref)
        size_bytes = sum(os.path.getsize(p) for p in self.store.files(ticket.ref).values())
        logger.info("[%s] External weights exported: %s (%d bytes)", request_id, ticket.ref.uri, size_bytes)
        return {"path": ticket.ref.uri, "size_bytes": size_bytes, "type": "save_weights_external"}

    def require_load_family(self, client_info: Dict[str, Any], ref: CheckpointRef, optimizer: bool,
                            optimizer_config: Optional[OptimizerConfig]) -> None:
        """Raises ValueError (a 400) naming both families when a load would
        mix them: a declared `optimizer_config` that
        is not the model's, or optimizer state from a checkpoint of another
        family. A weights-only load crosses families freely."""
        family = client_info["optimizer_config"]["type"]
        if optimizer_config is not None and optimizer_config.type not in self.backend.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"optimizer family {optimizer_config.type!r} is not served by {type(self.backend).__name__}; "
                f"{', '.join(sorted(self.backend.SUPPORTED_OPTIMIZERS))} are")
        if optimizer_config is not None and optimizer_config.type != family:
            raise ValueError(f"load_weights declares optimizer {optimizer_config.type} but the model's optimizer is {family}")
        if optimizer:
            saved = self.store.optimizer_config(ref)["type"]
            if saved != family:
                raise ValueError(f"checkpoint {ref.uri} holds {saved} optimizer state but the model's optimizer is {family}; "
                                 "load it with optimizer=false")

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

    def _checkpoint(self, rec: Dict[str, Any]) -> Checkpoint:
        """A store record (from store.list) as the SDK's Checkpoint, its size walked now."""
        kind = CheckpointKind(rec["kind"])
        return Checkpoint(
            checkpoint_id=f"{kind.value}/{rec['name']}",
            checkpoint_type=CHECKPOINT_TYPES[kind],
            # created_at is naive UTC (MetadataStorage.save_checkpoint); the SDK CLI compares it with aware dates
            time=datetime.fromisoformat(rec["created_at"]).replace(tzinfo=timezone.utc).isoformat(),
            tinker_path=rec["uri"],
            size_bytes=self.store.size_bytes(CheckpointRef.parse(rec["uri"])),
            status=rec["status"],
            expires_at=rec.get("expires_at"),  # absent on records older than ttl support
            user_metadata=rec.get("user_metadata"),  # absent on records older than user_metadata support
        )

    def list_checkpoints(self, model_id: str) -> List[Checkpoint]:
        """A training run's checkpoints, newest first."""
        return [self._checkpoint(rec) for rec in self.store.list(model_id, with_size=False)]

    def list_all_checkpoints(self, limit: int, offset: int) -> Tuple[List[Checkpoint], int]:
        """One page of every training run's checkpoints, newest first, and the
        total. Sizes are walked for the page only, not for every checkpoint."""
        records = [rec for model_id in self.store.model_ids() for rec in self.store.list(model_id, with_size=False)]
        records.sort(key=lambda r: (r["created_at"], r["uri"]), reverse=True)
        return [self._checkpoint(rec) for rec in records[offset:offset + limit]], len(records)

    def latest_checkpoints(self, model_id: str) -> Dict[str, Optional[Checkpoint]]:
        """A run's newest loadable checkpoint of each kind a TrainingRun names:
        completed and not expired, so the path it reports still resolves."""
        latest: Dict[str, Optional[Checkpoint]] = {"last_checkpoint": None, "last_sampler_checkpoint": None}
        field_of = {CheckpointKind.WEIGHTS: "last_checkpoint", CheckpointKind.SAMPLER_WEIGHTS: "last_sampler_checkpoint"}
        for rec in self.store.list(model_id, with_size=False):   # newest first
            field = field_of.get(CheckpointKind(rec["kind"]))   # None: a run names no external checkpoint
            if field is None or latest[field] is not None:
                continue
            if rec["status"] != CheckpointStatus.COMPLETED.value or is_expired(rec):
                continue
            latest[field] = self._checkpoint(rec)
        return latest

    async def save_weights_for_sampler(
        self,
        model_id: str,
        request_id: str,
        name: Optional[str],
        options: CheckpointSaveOptions,
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
                options=options,
                optimizer_config=client_info["optimizer_config"],
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
            options=options,
            optimizer_config=client_info["optimizer_config"],
        )
        logger.info("[%s] Saving sampler weights for %s to %s (step %s)", request_id, model_id, ticket.ref.uri, ticket.step)
        await self._save(handle, ticket)
        logger.info("[%s] Sampler weights saved: %s", request_id, ticket.ref.uri)
        return {
            "path": ticket.ref.uri, "sampling_session_id": None, "uri": ticket.ref.uri,
            "step": ticket.step, "name": name, "type": "save_weights_for_sampler",
        }
