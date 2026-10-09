"""Where a checkpoint lives, and what the server knows about it.

A ``tinker://<model_id>/<kind>/<name>`` URI is an identity, never a location.
The store is the only module that turns that identity into a directory, mints
the per-model save counter, and records what happened to each save:

    <checkpoint_base>/<model_id>/weights/<name>/           kind=weights
    <checkpoint_base>/<model_id>/sampler_weights/<name>/   kind=sampler_weights
    <checkpoint_base>/<model_id>/native/                   the backend's private
                                                           per-model area
                                                           (Megatron --save, ...)

Backends receive resolved directories (``root``, ``native_root``) and own what
goes inside them; they never see a URI. Records hold identity, status, the
counter value and the served weight version -- not a path -- so a record and
the bytes can never disagree about where the bytes are.

Records live under MetadataStorage as ``<kind>--<name>.json`` per model. A
save is ``pending`` between ``begin_save`` and ``complete``/``fail``; a read
through ``require`` refuses anything but ``completed``. Pending rows left
behind by a crash are marked failed at boot (``sweep_pending``).
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..storage.metadata import MetadataStorage

logger = logging.getLogger(__name__)

SCHEME = "tinker://"
NATIVE_DIRNAME = "native"


class CheckpointKind(str, Enum):
    WEIGHTS = "weights"
    SAMPLER_WEIGHTS = "sampler_weights"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# --- errors: routers map these to status codes in one place (api.py) ---------

class CheckpointError(Exception):
    status_code = 500


class InvalidCheckpointPath(CheckpointError, ValueError):
    status_code = 400


class CheckpointNotFound(CheckpointError):
    status_code = 404


class CheckpointPending(CheckpointError):
    status_code = 425


class CheckpointFailed(CheckpointError):
    status_code = 500


class CheckpointKindMismatch(CheckpointError):
    status_code = 400


# --- identity ----------------------------------------------------------------

def _segment(value: str, what: str) -> str:
    if not value or "/" in value or value in (".", "..") or "\x00" in value:
        raise InvalidCheckpointPath(f"invalid {what} {value!r}: one non-empty path segment")
    return value


@dataclass(frozen=True)
class CheckpointRef:
    """Identity of a checkpoint: which model, which kind, which name."""

    model_id: str
    kind: CheckpointKind
    name: str

    @classmethod
    def parse(cls, uri: str) -> "CheckpointRef":
        """``tinker://<model_id>/<weights|sampler_weights>/<name>`` -> ref.

        Anything else -- a filesystem path, a bare ``tinker://<model>``, an
        unknown kind, extra segments -- is InvalidCheckpointPath."""
        if not isinstance(uri, str) or not uri.startswith(SCHEME):
            raise InvalidCheckpointPath(
                f"checkpoint path must be {SCHEME}<model_id>/<weights|sampler_weights>/<name>, got {uri!r}")
        parts = uri[len(SCHEME):].split("/")
        if len(parts) != 3:
            raise InvalidCheckpointPath(
                f"checkpoint path must be {SCHEME}<model_id>/<weights|sampler_weights>/<name>, got {uri!r}")
        model_id, kind, name = parts
        try:
            kind_v = CheckpointKind(kind)
        except ValueError:
            raise InvalidCheckpointPath(
                f"checkpoint kind must be weights or sampler_weights, got {kind!r} in {uri!r}") from None
        return cls(_segment(model_id, "model_id"), kind_v, _segment(name, "checkpoint name"))

    @classmethod
    def make(cls, model_id: str, kind: CheckpointKind, name: str) -> "CheckpointRef":
        return cls(_segment(model_id, "model_id"), CheckpointKind(kind), _segment(name, "checkpoint name"))

    @property
    def uri(self) -> str:
        return f"{SCHEME}{self.model_id}/{self.kind.value}/{self.name}"

    @property
    def record_key(self) -> str:
        return f"{self.kind.value}--{self.name}"

    def __str__(self) -> str:
        return self.uri


@dataclass(frozen=True)
class SaveTicket:
    """What a save in flight needs: its identity, the directory to fill and
    the counter value it was assigned (None for an ephemeral save)."""

    ref: CheckpointRef
    root: Path
    step: Optional[int]


# --- store -------------------------------------------------------------------

class CheckpointStore:
    def __init__(self, base: Path, metadata: MetadataStorage):
        self.base = Path(base)
        self.metadata = metadata

    # layout ------------------------------------------------------------------

    def model_dir(self, model_id: str) -> Path:
        return self.base / _segment(model_id, "model_id")

    def root(self, ref: CheckpointRef) -> Path:
        return self.model_dir(ref.model_id) / ref.kind.value / ref.name

    def native_root(self, model_id: str) -> Path:
        """The backend's private per-model directory; created on demand."""
        p = self.model_dir(model_id) / NATIVE_DIRNAME
        p.mkdir(parents=True, exist_ok=True)
        return p

    # records -----------------------------------------------------------------

    def get(self, ref: CheckpointRef) -> Optional[Dict[str, Any]]:
        return self.metadata.load_checkpoint(ref.model_id, ref.record_key)

    def _write(self, ref: CheckpointRef, record: Dict[str, Any]) -> None:
        self.metadata.save_checkpoint(ref.model_id, ref.record_key, record)

    def records(self, model_id: str) -> List[Dict[str, Any]]:
        out = []
        for rec in self.metadata.list_checkpoints(model_id, limit=100000):
            if "kind" in rec and "name" in rec:  # this store's records only
                out.append(rec)
        return out

    def next_step(self, model_id: str) -> int:
        """Monotonic per model: one past the largest counter ever assigned,
        failed and deleted saves included, so a backend's counter-named
        native artifact is never overwritten by a later save."""
        steps = [int(r["step"]) for r in self.records(model_id) if r.get("step") is not None]
        return max(steps, default=0) + 1

    # lifecycle ---------------------------------------------------------------

    def begin_save(
        self,
        model_id: str,
        kind: CheckpointKind,
        name: str,
        *,
        persist: bool = True,
        weight_version: Optional[int] = None,
    ) -> SaveTicket:
        """Record a pending save and hand back where to write. A name already
        recorded for this model and kind is overwritten by the new save (the
        API lets a client save twice under one name); its old counter value is
        not reused."""
        ref = CheckpointRef.make(model_id, kind, name)
        root = self.root(ref)
        step = self.next_step(model_id) if persist else None
        if persist:
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        self._write(ref, {
            "model_id": ref.model_id, "kind": ref.kind.value, "name": ref.name,
            "status": CheckpointStatus.PENDING.value, "step": step, "ephemeral": not persist,
            "weight_version": weight_version, "error": None, "completed_at": None,
        })
        return SaveTicket(ref=ref, root=root, step=step)

    def complete(self, ref: CheckpointRef) -> None:
        self._set_status(ref, CheckpointStatus.COMPLETED, None)

    def fail(self, ref: CheckpointRef, error: str) -> None:
        self._set_status(ref, CheckpointStatus.FAILED, error)

    def _set_status(self, ref: CheckpointRef, status: CheckpointStatus, error: Optional[str]) -> None:
        rec = self.get(ref)
        if rec is None:
            raise CheckpointNotFound(f"no record for {ref.uri}")
        rec["status"] = status.value
        rec["error"] = error
        rec["completed_at"] = datetime.utcnow().isoformat()
        self._write(ref, rec)

    def sweep_pending(self) -> int:
        """Boot: a save that was in flight when the process died never finished."""
        n = 0
        for model_dir in sorted(self.metadata.checkpoints_dir.glob("*")):
            if not model_dir.is_dir():
                continue
            for rec in self.records(model_dir.name):
                if rec.get("status") == CheckpointStatus.PENDING.value:
                    ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
                    self.fail(ref, "server restarted while the save was in flight")
                    n += 1
        return n

    # reads -------------------------------------------------------------------

    def require(self, ref: CheckpointRef, kind: Optional[CheckpointKind] = None) -> Path:
        """The root of a completed checkpoint, or the error a client should see."""
        if kind is not None and ref.kind != kind:
            raise CheckpointKindMismatch(f"{ref.uri} is a {ref.kind.value} checkpoint; {kind.value} required")
        rec = self.get(ref)
        if rec is None:
            raise CheckpointNotFound(f"Checkpoint not found: {ref.uri}")
        status = rec.get("status")
        if status == CheckpointStatus.PENDING.value:
            raise CheckpointPending(f"Checkpoint is still being created: {ref.uri}")
        if status == CheckpointStatus.FAILED.value:
            raise CheckpointFailed(f"Checkpoint creation failed: {ref.uri}: {rec.get('error')}")
        return self.root(ref)

    def resolve_resume(self, uri: str) -> Path:
        """The root a training resume (create_model / load_weights) may load."""
        return self.require(CheckpointRef.parse(uri), kind=CheckpointKind.WEIGHTS)

    def list(self, model_id: str, with_size: bool = True) -> List[Dict[str, Any]]:
        """Persistent records of a model, newest first, each with its uri and
        (optionally) the bytes under its root."""
        out = []
        for rec in self.records(model_id):
            if rec.get("ephemeral"):
                continue
            ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
            item = dict(rec, uri=ref.uri)
            if with_size:
                root = self.root(ref)
                item["size_bytes"] = (
                    sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)
                    if root.is_dir() else None
                )
            out.append(item)
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    def model_ids(self) -> List[str]:
        """Every model with a checkpoint record, live or not."""
        return sorted(d.name for d in self.metadata.checkpoints_dir.glob("*") if d.is_dir())

    def usage(self) -> Tuple[int, int]:
        """(count, bytes) over the completed persistent checkpoints of every
        model. A completed record whose root is gone counts with 0 bytes."""
        count = 0
        size_bytes = 0
        for model_id in self.model_ids():
            for item in self.list(model_id, with_size=True):
                if item["status"] != CheckpointStatus.COMPLETED.value:
                    continue
                count += 1
                size_bytes += item["size_bytes"] or 0
        return count, size_bytes

    # deletion ----------------------------------------------------------------

    def delete(self, ref: CheckpointRef) -> bool:
        """Remove the record and the bytes under the root. A backend's private
        artifacts under native/ are not touched (the store does not know them)."""
        if self.get(ref) is None:
            return False
        root = self.root(ref)
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
        self.metadata.delete_checkpoint(ref.model_id, ref.record_key)
        logger.info("Deleted checkpoint %s (%s)", ref.uri, root)
        return True

    def release_model(self, model_id: str) -> int:
        """The model is gone from the server: drop its ephemeral records.
        Persistent checkpoints and native/ stay -- a client resumes from them
        after its training client is deleted."""
        n = 0
        for rec in self.records(model_id):
            if rec.get("ephemeral"):
                ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
                self.metadata.delete_checkpoint(ref.model_id, ref.record_key)
                n += 1
        return n
