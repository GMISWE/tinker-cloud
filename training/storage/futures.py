"""
Futures storage: the record of every async operation the server has accepted.

One SQLite connection for the process (WAL, guarded by a lock) instead of a
fresh connection per call; the request payload is never persisted — only its
sha256 and byte size, which is all seq_id idempotency needs. A bounded
in-memory cache keeps the futures being polled cheap to read without
retaining every result of a long run.

Everything here is synchronous and cheap (point lookups on a warm
connection); the router's long-poll awaits the operation's asyncio task
directly, so nothing polls this table.
"""
import hashlib
import json
import logging
import sqlite3
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_COLUMNS = ("request_id", "model_id", "operation", "status", "result",
            "created_at", "updated_at", "seq_id", "payload_hash", "payload_bytes",
            "result_proto")


def _serialize_result(result: Any) -> Optional[str]:
    if result is None:
        return None
    return json.dumps(_as_plain(result))


def _as_plain(result: Any) -> Any:
    """Pydantic model -> dict; a plain dict / list unchanged."""
    if isinstance(result, BaseModel):
        return result.model_dump()
    return result

class DuplicateSeqId(ValueError):
    """A (model_id, seq_id) pair was reused with a different request."""


class FuturesStorage:
    """Thread-safe futures store: SQLite of record, bounded memory cache in front."""

    CACHE_SIZE = 512
    TRAINING_OPERATIONS = ("forward", "forward_backward", "optim_step")

    def __init__(self, db_path: Path, cache_size: int = CACHE_SIZE):
        self.db_path = db_path
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_database()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ schema
    def _init_database(self) -> None:
        cur = self._conn
        columns = {row[1] for row in cur.execute("PRAGMA table_info(futures)").fetchall()}
        if columns and set(_COLUMNS) != columns:
            # Older layout. Rows never survive a server start (startup purges
            # all futures), so rebuild rather than migrate.
            logger.info("Rebuilding futures table at %s (legacy schema)", self.db_path)
            cur.execute("DROP TABLE futures")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS futures (
                request_id TEXT PRIMARY KEY,
                model_id TEXT,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                seq_id INTEGER,
                payload_hash TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                result_proto BLOB
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_status ON futures(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_created ON futures(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_futures_model ON futures(model_id)")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_futures_model_seq
            ON futures(model_id, seq_id) WHERE seq_id IS NOT NULL
        """)
        logger.info("Initialized futures database at %s", self.db_path)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def payload_digest(payload: Dict[str, Any]) -> "tuple[str, int]":
        """(sha256, byte size) of the canonical JSON form of a request payload."""
        raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest(), len(raw)

    @classmethod
    def payload_hash(cls, payload: Dict[str, Any]) -> str:
        return cls.payload_digest(payload)[0]

    @staticmethod
    def _row_to_future(row: tuple) -> Dict[str, Any]:
        fut = dict(zip(_COLUMNS, row))
        fut["result"] = json.loads(fut["result"]) if fut["result"] else None
        return fut

    def _cache_put(self, fut: Dict[str, Any]) -> None:
        # caller holds the lock
        rid = fut["request_id"]
        self._cache[rid] = fut
        self._cache.move_to_end(rid)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _load(self, request_id: str) -> Optional[Dict[str, Any]]:
        # caller holds the lock
        row = self._conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM futures WHERE request_id = ?", (request_id,)
        ).fetchone()
        return self._row_to_future(row) if row else None

    def find_by_seq_id(self, model_id: str, seq_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT request_id, operation, payload_hash FROM futures WHERE model_id = ? AND seq_id = ?",
                (model_id, seq_id),
            ).fetchone()
        if row is None:
            return None
        return {"request_id": row[0], "operation": row[1], "payload_hash": row[2]}

    # --------------------------------------------------------------- lifecycle
    def save_future(
        self,
        request_id: str,
        operation: str,
        payload: Dict[str, Any],
        model_id: Optional[str] = None,
        seq_id: Optional[int] = None,
    ) -> str:
        """
        Register a pending future.

        With a `seq_id`, (model_id, seq_id) is unique: a retry carrying the same
        operation and payload returns the existing request_id; a different
        operation or payload under a reused seq_id raises DuplicateSeqId.

        Returns the request_id that owns this (model_id, seq_id).
        """
        phash, pbytes = self.payload_digest(payload)
        now = datetime.utcnow().isoformat()
        fut = {
            "request_id": request_id, "model_id": model_id, "operation": operation,
            "status": "pending", "result": None, "created_at": now, "updated_at": now,
            "seq_id": seq_id, "payload_hash": phash, "payload_bytes": pbytes,
            "result_proto": None,
        }
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO futures ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                    tuple(fut[c] for c in _COLUMNS),
                )
            except sqlite3.IntegrityError:
                row = None
                if seq_id is not None and model_id is not None:
                    row = self._conn.execute(
                        "SELECT request_id, operation, payload_hash FROM futures WHERE model_id = ? AND seq_id = ?",
                        (model_id, seq_id),
                    ).fetchone()
                if row and row[1] == operation and row[2] == phash:
                    logger.info("[%s] retry of seq_id %s for %s -> %s", request_id, seq_id, model_id, row[0])
                    return row[0]
                if row:
                    raise DuplicateSeqId(
                        f"Training request sequence number {seq_id} was reused for {model_id} "
                        f"with a different {'operation' if row[1] != operation else 'payload'}"
                    ) from None
                raise
            self._cache_put(fut)
        return request_id

    def update_status(self, request_id: str, status: str, result: Optional[Any] = None,
                      result_proto: Optional[bytes] = None) -> bool:
        """Set a future's status (and result; and its proto view, when the
        operation has one). False if the future is unknown."""
        plain = _as_plain(result) if result is not None else None
        now = datetime.utcnow().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE futures SET status = ?, result = ?, result_proto = ?, updated_at = ? WHERE request_id = ?",
                (status, json.dumps(plain) if plain is not None else None, result_proto, now, request_id),
            )
            if cur.rowcount == 0:
                logger.warning("Future %s not found for update", request_id)
                return False
            fut = self._cache.get(request_id) or self._load(request_id)
            if fut is not None:
                fut.update(status=status, updated_at=now, result_proto=result_proto)
                if plain is not None:
                    fut["result"] = plain
                self._cache_put(fut)
        return True

    def set_result_proto(self, request_id: str, result_proto: bytes) -> None:
        """Attach the proto view of a completed result built after completion."""
        with self._lock:
            self._conn.execute("UPDATE futures SET result_proto = ? WHERE request_id = ?",
                               (result_proto, request_id))
            fut = self._cache.get(request_id)
            if fut is not None:
                fut["result_proto"] = result_proto

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            fut = self._cache.get(request_id)
            if fut is None:
                fut = self._load(request_id)
                if fut is not None:
                    self._cache_put(fut)
            return dict(fut) if fut is not None else None

    def list_futures(self, model_id: Optional[str] = None, status: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        query = f"SELECT {', '.join(_COLUMNS)} FROM futures WHERE 1=1"
        params: List[Any] = []
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_future(r) for r in rows]

    def has_training_requests(self, model_id: str) -> bool:
        """True once the model has seen any forward / forward_backward / optim_step."""
        placeholders = ",".join("?" * len(self.TRAINING_OPERATIONS))
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM futures WHERE model_id = ? AND operation IN ({placeholders}) LIMIT 1",
                (model_id, *self.TRAINING_OPERATIONS),
            ).fetchone()
        return row is not None

    def cleanup_old_futures(self, max_age_hours: int = 24) -> int:
        """Delete futures created more than `max_age_hours` ago; returns the count."""
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        with self._lock:
            for rid in [r for r, f in self._cache.items() if f["created_at"] < cutoff]:
                del self._cache[rid]
            removed = self._conn.execute("DELETE FROM futures WHERE created_at < ?", (cutoff,)).rowcount
        if removed:
            logger.info("Cleaned up %d futures older than %dh", removed, max_age_hours)
        return removed

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            stats = dict(self._conn.execute("SELECT status, COUNT(*) FROM futures GROUP BY status").fetchall())
            stats["total"] = self._conn.execute("SELECT COUNT(*) FROM futures").fetchone()[0]
            stats["in_memory"] = len(self._cache)
        return stats
