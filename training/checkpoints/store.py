"""Where a checkpoint lives, and what the server knows about it.

A ``tinker://<model_id>/<kind>/<name>`` URI is an identity, never a location.
The store is the only module that turns that identity into a directory, mints
the per-model save counter, and records what happened to each save:

    <checkpoint_base>/<model_id>/weights/<name>/           kind=weights
    <checkpoint_base>/<model_id>/sampler_weights/<name>/   kind=sampler_weights
    <checkpoint_base>/<model_id>/external_weights/<name>/  kind=external_weights
                                                           (HF-format export,
                                                           served file by file)
    <checkpoint_base>/<model_id>/native/                   the backend's private
                                                           per-model area
                                                           (Megatron --save, ...)

Backends receive resolved directories (``root``, ``native_root``) and own what
goes inside them; they never see a URI. Records hold identity, status, the
counter value and the served weight version -- not a path -- so a record and
the bytes can never disagree about where the bytes are.

Records live under MetadataStorage as ``<kind>--<name>.json`` per model. A
save is ``pending`` between ``begin_save`` and ``complete``/``fail``; a read
through ``require`` refuses anything but ``completed``, and a completed record
past its ``expires_at`` (the client's ttl_seconds, or a later ``set_ttl``) reads
as not found. Expired records and bytes are swept in the background. Pending
rows left behind by a crash are marked failed at boot (``sweep_pending``).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..storage.metadata import MetadataStorage

logger = logging.getLogger(__name__)

SCHEME = "tinker://"
NATIVE_DIRNAME = "native"
EXPIRED_SWEEP_INTERVAL_S = 60.0
MAX_RECORD_NAME_BYTES = 255


class CheckpointKind(str, Enum):
    WEIGHTS = "weights"
    SAMPLER_WEIGHTS = "sampler_weights"
    EXTERNAL_WEIGHTS = "external_weights"


KINDS_HELP = "|".join(k.value for k in CheckpointKind)


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointExists(Exception):
    pass


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


class CheckpointDeleteFailed(CheckpointError):
    """The files under a checkpoint could not be removed; the record stays so the delete can be retried."""
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
        """``tinker://<model_id>/<kind>/<name>`` -> ref (kind: KINDS_HELP).

        Anything else -- a filesystem path, a bare ``tinker://<model>``, an
        unknown kind, extra segments -- is InvalidCheckpointPath."""
        if not isinstance(uri, str) or not uri.startswith(SCHEME):
            raise InvalidCheckpointPath(
                f"checkpoint path must be {SCHEME}<model_id>/<{KINDS_HELP}>/<name>, got {uri!r}")
        parts = uri[len(SCHEME):].split("/")
        if len(parts) != 3:
            raise InvalidCheckpointPath(
                f"checkpoint path must be {SCHEME}<model_id>/<{KINDS_HELP}>/<name>, got {uri!r}")
        model_id, kind, name = parts
        try:
            kind_v = CheckpointKind(kind)
        except ValueError:
            raise InvalidCheckpointPath(
                f"checkpoint kind must be one of {KINDS_HELP}, got {kind!r} in {uri!r}") from None
        return cls(_segment(model_id, "model_id"), kind_v, _segment(name, "checkpoint name"))

    @classmethod
    def make(cls, model_id: str, kind: CheckpointKind, name: str) -> "CheckpointRef":
        ref = cls(_segment(model_id, "model_id"), CheckpointKind(kind), _segment(name, "checkpoint name"))
        if len(name.encode("utf-8")) > MAX_RECORD_NAME_BYTES:
            raise InvalidCheckpointPath(
                f"checkpoint name {name!r} exceeds {MAX_RECORD_NAME_BYTES} UTF-8 bytes")
        if len(ref.record_key.encode("utf-8")) > MAX_RECORD_NAME_BYTES:
            raise InvalidCheckpointPath(
                f"checkpoint name {name!r} would make record key {ref.record_key!r} exceed "
                f"{MAX_RECORD_NAME_BYTES} UTF-8 bytes")
        return ref

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


@dataclass(frozen=True)
class CheckpointSaveOptions:
    """Options carried from a save request to the store.

    `ttl_seconds=None` means no expiry. `overwrite=False` refuses pending or
    unexpired completed records. `user_metadata` replaces the old value.
    """

    ttl_seconds: Optional[int]
    overwrite: bool
    user_metadata: Optional[Dict[str, str]]


def is_expired(record: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    """Records written before ttl support carry no expires_at: never expire.
    expires_at is an aware UTC ISO string; `now` must be aware too."""
    expires_at = record.get("expires_at")
    if expires_at is None:
        return False
    return (now or datetime.now(timezone.utc)) >= datetime.fromisoformat(expires_at)


def _blocks_save(record: Dict[str, Any]) -> bool:
    """Whether overwrite=False refuses a record.

    pending: yes; completed: until expiry; failed: no.
    """
    status = record["status"]
    return status == CheckpointStatus.PENDING.value or (
        status == CheckpointStatus.COMPLETED.value and not is_expired(record)
    )


def _expires_at(ttl_seconds: Optional[int]) -> Optional[str]:
    """``ttl_seconds`` from now as an aware UTC ISO string; None never expires."""
    if ttl_seconds is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


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
        options: CheckpointSaveOptions,
        persist: bool = True,
        weight_version: Optional[int] = None,
        optimizer_config: Dict[str, Any],
    ) -> SaveTicket:
        """Record a pending save and hand back where to write. An existing
        live name is refused unless overwrite is true; its old counter value
        is not reused when replaced. ``optimizer_config`` is the model's
        optimizer family record."""
        ref = CheckpointRef.make(model_id, kind, name)
        existing = self.get(ref)
        if not options.overwrite and existing is not None and _blocks_save(existing):
            raise CheckpointExists(f"Checkpoint already exists: {ref.uri}")
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
            "expires_at": _expires_at(options.ttl_seconds), "user_metadata": options.user_metadata,
            "optimizer_config": optimizer_config,
        })
        return SaveTicket(ref=ref, root=root, step=step)

    def optimizer_config(self, ref: CheckpointRef) -> Dict[str, Any]:
        """The optimizer family recorded with a checkpoint, as weights_info
        echoes it. Every record this store writes carries the field; one
        without it is from before the field existed, when AdamW was the only
        family the server had. Raises KeyError for an unknown ref."""
        rec = self.get(ref)
        if rec is None:
            raise KeyError(f"no checkpoint record for {ref.uri}")
        if "optimizer_config" not in rec:
            return {"type": "adamw"}
        return rec["optimizer_config"]

    def complete(self, ref: CheckpointRef) -> None:
        self._set_status(ref, CheckpointStatus.COMPLETED, None)

    def fail(self, ref: CheckpointRef, error: str) -> None:
        self._set_status(ref, CheckpointStatus.FAILED, error)

    def set_ttl(self, ref: CheckpointRef, ttl_seconds: Optional[int]) -> bool:
        """Expire a persistent checkpoint ``ttl_seconds`` from now, or never
        (None). False when there is nothing to set it on: no record, an
        ephemeral one, or one already past its expires_at (it reads as gone)."""
        rec = self.get(ref)
        if rec is None or rec["ephemeral"] or is_expired(rec):
            return False
        rec["expires_at"] = _expires_at(ttl_seconds)
        self._write(ref, rec)
        return True

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
        for model_id in self.model_ids():
            for rec in self.records(model_id):
                if rec["status"] == CheckpointStatus.PENDING.value:
                    ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
                    self.fail(ref, "server restarted while the save was in flight")
                    n += 1
        return n

    def sweep_expired(self) -> int:
        """Remove expired terminal checkpoints and their bytes."""
        n = 0
        now = datetime.now(timezone.utc)
        for model_id in self.model_ids():
            for rec in self.records(model_id):
                if rec["status"] != CheckpointStatus.PENDING.value and is_expired(rec, now):
                    ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
                    try:
                        if self.delete(ref):
                            n += 1
                    except CheckpointDeleteFailed as e:
                        # delete() has already logged the full error. A stuck checkpoint is retried every sweep, so
                        # this is one line each time and not a traceback.
                        logger.warning("Could not delete expired checkpoint %s (%s); the next sweep retries it", ref.uri, e)
        return n

    async def sweep_forever(self) -> None:
        """sweep_expired now, then every EXPIRED_SWEEP_INTERVAL_S."""
        while True:
            n = self.sweep_expired()
            if n:
                logger.info("Deleted %d expired checkpoint(s)", n)
            await asyncio.sleep(EXPIRED_SWEEP_INTERVAL_S)

    # reads -------------------------------------------------------------------

    def require(self, ref: CheckpointRef, kind: Optional[CheckpointKind] = None) -> Path:
        """The root of a completed checkpoint, or the error a client should see."""
        if kind is not None and ref.kind != kind:
            raise CheckpointKindMismatch(f"{ref.uri} is a {ref.kind.value} checkpoint; {kind.value} required")
        rec = self.get(ref)
        if rec is None:
            raise CheckpointNotFound(f"Checkpoint not found: {ref.uri}")
        status = rec["status"]
        if status == CheckpointStatus.PENDING.value:
            raise CheckpointPending(f"Checkpoint is still being created: {ref.uri}")
        if status == CheckpointStatus.FAILED.value:
            if rec.get("delete_failed"):
                raise CheckpointFailed(f"Checkpoint is partly deleted and cannot be used: {ref.uri}: {rec.get('error')}")
            raise CheckpointFailed(f"Checkpoint creation failed: {ref.uri}: {rec.get('error')}")
        if is_expired(rec):
            raise CheckpointNotFound(f"Checkpoint expired at {rec['expires_at']}: {ref.uri}")
        return self.root(ref)

    def files(self, ref: CheckpointRef, kind: Optional[CheckpointKind] = None) -> Dict[str, Path]:
        """Regular files of a completed checkpoint, keyed by path relative to
        its root (what a per-file download serves)."""
        root = self.require(ref, kind=kind)
        return {
            str(p.relative_to(root)): p
            for p in sorted(root.rglob("*")) if p.is_file()
        }

    def resolve_resume(self, uri: str) -> Path:
        """The root a training resume (create_model / load_weights) may load."""
        return self.require(CheckpointRef.parse(uri), kind=CheckpointKind.WEIGHTS)

    def list(self, model_id: str, with_size: bool = True) -> List[Dict[str, Any]]:
        """Persistent records of a model, newest first, each with its uri and
        (optionally) the bytes under its root."""
        out = []
        for rec in self.records(model_id):
            if rec["ephemeral"]:
                continue
            ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
            item = dict(rec, uri=ref.uri)
            if with_size:
                item["size_bytes"] = self.size_bytes(ref)
            out.append(item)
        out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return out

    def size_bytes(self, ref: CheckpointRef) -> Optional[int]:
        """Bytes under the checkpoint's root; None when the root is gone."""
        root = self.root(ref)
        if not root.is_dir():
            return None
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)

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
        artifacts under native/ are not touched (the store does not know them).

        A checkpoint still being saved is refused (CheckpointPending): the
        backend may be writing under the root, and `complete()` would turn a
        half-deleted record back into a completed one. The record goes only
        after the bytes.

        Any storage error on the way (the files, or the volume that holds the
        records) is raised as CheckpointDeleteFailed, so the client is told why
        and sweep_expired, which has no other handler, survives it. The message
        and the stored error carry the reason without the server's path; the
        full error is in the log. If the bytes could not be removed the record
        stays, marked failed so nothing loads a half-deleted checkpoint. The
        client can retry the delete, and so can the expiry sweep when the
        checkpoint has expired."""
        rec = self.get(ref)
        if rec is None:
            return False
        if rec["status"] == CheckpointStatus.PENDING.value:
            raise CheckpointPending(f"Checkpoint is still being created, so it cannot be deleted yet: {ref.uri}")
        root = self.root(ref)
        try:
            if root.is_dir():
                try:
                    shutil.rmtree(root)
                except OSError as e:
                    self._mark_delete_failed(ref, rec, e)
                    raise
            self.metadata.delete_checkpoint(ref.model_id, ref.record_key)
        except OSError as e:
            logger.error("Could not delete checkpoint %s (%s): %s", ref.uri, root, e)
            raise CheckpointDeleteFailed(f"Could not delete checkpoint {ref.uri}: {e.strerror or type(e).__name__}") from e
        logger.info("Deleted checkpoint %s (%s)", ref.uri, root)
        return True

    def _mark_delete_failed(self, ref: CheckpointRef, rec: Dict[str, Any], e: OSError) -> None:
        """Best effort: the volume that holds the records may be the thing that is failing, and then the caller's
        error is the one to report. `completed_at` stays: it is when the save finished."""
        rec.update(status=CheckpointStatus.FAILED.value, error=f"delete did not finish: {e.strerror or type(e).__name__}",
                   delete_failed=True)
        try:
            self._write(ref, rec)
        except OSError:
            logger.exception("Could not record the failed delete of %s", ref.uri)

    def release_model(self, model_id: str) -> int:
        """The model is gone from the server: drop its ephemeral records.
        Persistent checkpoints and native/ stay -- a client resumes from them
        after its training client is deleted."""
        n = 0
        for rec in self.records(model_id):
            if rec["ephemeral"]:
                ref = CheckpointRef.make(rec["model_id"], rec["kind"], rec["name"])
                self.metadata.delete_checkpoint(ref.model_id, ref.record_key)
                n += 1
        return n
