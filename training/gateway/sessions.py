"""Sessions, samplers and the model registry: one SQLite file under METADATA_DIR shared by every gateway worker.

The workers are processes, so the file is the source of truth and each worker caches only
samplers, which never change once written. A session row is claimed for expiry inside one
transaction, so N workers running the reaper free each model once.
"""
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dbfile import opened_alone

logger = logging.getLogger(__name__)

SESSION_COLUMNS = "session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat, finished_at, finish_reason, finish_detail"
SAMPLER_COLUMNS = "sampler_id, session_id, model_id, base_model, model_path, pinned_version, client_counter"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Sampler:
    sampler_id: str
    session_id: str
    model_id: Optional[str]        # None: a sampler with no model behind it (never minted here; a pre-split row)
    base_model: Optional[str]
    model_path: Optional[str]
    pinned_version: Optional[int]  # the weight version the sampler was saved at; None: served live
    client_counter: int


@dataclass
class Session:
    session_id: str
    tags: List[str]
    user_metadata: Dict[str, Any]
    sdk_version: str
    created_at: str
    last_heartbeat: str
    finished_at: Optional[str]
    finish_reason: Optional[str]
    finish_detail: Optional[str]
    model_ids: List[str]     # in model_seq_id order
    sampler_ids: List[str]


class SessionStore:
    """The gateway's durable state. `path=None` is an in-memory store for one process (tests)."""

    def __init__(self, path: Optional[Path]) -> None:
        self._conn = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False,
                                     isolation_level=None, timeout=10.0)
        self._conn.execute("PRAGMA busy_timeout=10000")
        self.reads = 0    # statements that read; tests/unit/gateway/test_hot_path.py pins the warm path at zero
        self.writes = 0
        if path is None:
            self._init()
        else:
            opened_alone(path, self._open_file)

    def _open_file(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        c = self._conn
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, sdk_version TEXT, tags TEXT, user_metadata TEXT,
            created_at TEXT NOT NULL, last_heartbeat TEXT NOT NULL,
            finished_at TEXT, finish_reason TEXT, finish_detail TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS samplers (
            sampler_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, model_id TEXT, base_model TEXT, model_path TEXT,
            created_at TEXT NOT NULL, client_counter INTEGER NOT NULL DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS session_models (
            session_id TEXT NOT NULL, model_id TEXT NOT NULL, model_seq_id INTEGER, base_model TEXT, model_path TEXT,
            created_at TEXT NOT NULL, PRIMARY KEY (session_id, model_id))""")
        # columns added after the first release; a pre-existing file lacks them
        self._add_missing_columns("sessions", {"finished_at": "TEXT", "finish_reason": "TEXT", "finish_detail": "TEXT"})
        self._add_missing_columns("samplers", {"client_counter": "INTEGER NOT NULL DEFAULT 0", "pinned_version": "INTEGER"})
        self._add_missing_columns("session_models", {"engine": "TEXT"})
        c.execute("CREATE INDEX IF NOT EXISTS idx_samplers_session ON samplers(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_models_seq ON session_models(session_id, model_seq_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_models_model ON session_models(model_id)")

    def _add_missing_columns(self, table: str, columns: Dict[str, str]) -> None:
        present = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in columns.items():
            if name not in present:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self._conn.close()

    def _read(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        self.reads += 1
        return self._conn.execute(sql, params).fetchall()

    def _write(self, sql: str, params: Tuple = ()) -> int:
        self.writes += 1
        return self._conn.execute(sql, params).rowcount

    # ---------------------------------------------------------------- sessions

    def create_session(self, tags: List[str], user_metadata: Dict[str, Any], sdk_version: str) -> str:
        session_id = str(uuid.uuid4())
        now = _now()
        self._write("INSERT INTO sessions (session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, sdk_version, json.dumps(tags), json.dumps(user_metadata), now, now))
        return session_id

    def exists(self, session_id: str) -> bool:
        return bool(self._read("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)))

    def get_session(self, session_id: str) -> Optional[Session]:
        rows = self._read(f"SELECT {SESSION_COLUMNS} FROM sessions WHERE session_id = ?", (session_id,))
        if not rows:
            return None
        r = rows[0]
        models = self._read("SELECT model_id FROM session_models WHERE session_id = ? ORDER BY model_seq_id ASC", (session_id,))
        samplers = self._read("SELECT sampler_id FROM samplers WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        return Session(session_id=r[0], sdk_version=r[1], tags=json.loads(r[2]), user_metadata=json.loads(r[3]),
                       created_at=r[4], last_heartbeat=r[5],
                       finished_at=r[6], finish_reason=r[7], finish_detail=r[8],
                       model_ids=[m[0] for m in models], sampler_ids=[s[0] for s in samplers])

    def list_sessions(self, limit: int, offset: int) -> List[str]:
        return [r[0] for r in self._read("SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))]

    def touch(self, session_id: str) -> Optional[bool]:
        """Record a heartbeat. True: recorded; False: the session has finished; None: unknown session."""
        if self._write("UPDATE sessions SET last_heartbeat = ? WHERE session_id = ? AND finished_at IS NULL", (_now(), session_id)):
            return True
        return False if self.exists(session_id) else None

    def finish(self, session_id: str, reason: str, detail: Optional[str]) -> Optional[Session]:
        """First-wins: a later finish leaves the recorded reason untouched. None: unknown session."""
        self._write("UPDATE sessions SET finished_at = ?, finish_reason = ?, finish_detail = ? WHERE session_id = ? AND finished_at IS NULL",
                    (_now(), reason, detail, session_id))
        return self.get_session(session_id)

    def claim_expired(self, timeout_s: float) -> List[Tuple[str, List[str]]]:
        """Drop every session silent for longer than `timeout_s`, with its models and samplers, in one
        transaction, and return (session_id, model_ids) pairs: the caller frees the models. A row one
        worker claimed is gone before another worker's tick can see it."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_s)).isoformat()
        claimed: List[Tuple[str, List[str]]] = []
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for (session_id,) in self._read("SELECT session_id FROM sessions WHERE last_heartbeat < ?", (cutoff,)):
                models = [m[0] for m in self._read("SELECT model_id FROM session_models WHERE session_id = ?", (session_id,))]
                self._write("DELETE FROM samplers WHERE session_id = ?", (session_id,))
                self._write("DELETE FROM session_models WHERE session_id = ?", (session_id,))
                self._write("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                claimed.append((session_id, models))
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return claimed

    # ------------------------------------------------------------------ models

    def add_model(self, session_id: str, model_id: str, model_seq_id: int, base_model: str, engine: str) -> None:
        """Record a model at submit; a create that fails later leaves the row until its delete."""
        self._write("INSERT OR REPLACE INTO session_models (session_id, model_id, model_seq_id, base_model, created_at, engine) VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, model_id, model_seq_id, base_model, _now(), engine))

    def remove_model(self, model_id: str) -> Optional[str]:
        """Drop the model's row; the session that owned it, or None."""
        rows = self._read("SELECT session_id FROM session_models WHERE model_id = ?", (model_id,))
        if not rows:
            return None
        self._write("DELETE FROM session_models WHERE model_id = ?", (model_id,))
        return rows[0][0]

    def session_of_model(self, model_id: str) -> Optional[str]:
        rows = self._read("SELECT session_id FROM session_models WHERE model_id = ?", (model_id,))
        return rows[0][0] if rows else None

    def model_ids(self) -> List[str]:
        return [r[0] for r in self._read("SELECT model_id FROM session_models")]

    # ---------------------------------------------------------------- samplers

    def add_sampler(self, sampler_id: str, session_id: str, model_id: Optional[str], base_model: Optional[str],
                    model_path: Optional[str], pinned_version: Optional[int]) -> Sampler:
        self._write("INSERT OR REPLACE INTO samplers (sampler_id, session_id, model_id, base_model, model_path, pinned_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (sampler_id, session_id, model_id, base_model, model_path, pinned_version, _now()))
        return Sampler(sampler_id, session_id, model_id, base_model, model_path, pinned_version, 0)

    def load_sampler(self, sampler_id: str) -> Optional[Sampler]:
        rows = self._read(f"SELECT {SAMPLER_COLUMNS} FROM samplers WHERE sampler_id = ?", (sampler_id,))
        return Sampler(*rows[0]) if rows else None

    def join(self, sampler_id: str) -> Optional[int]:
        """The next client id of a sampler (the creator is 0); None when the sampler is unknown."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if not self._write("UPDATE samplers SET client_counter = client_counter + 1 WHERE sampler_id = ?", (sampler_id,)):
                self._conn.execute("COMMIT")
                return None
            counter = self._read("SELECT client_counter FROM samplers WHERE sampler_id = ?", (sampler_id,))[0][0]
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return int(counter)


class SamplerCache:
    """Per-worker cache of samplers: a hit costs no I/O, a miss reads the store once. Samplers never
    change, so an entry is right until its session is reaped, after which the engine answers 404
    for the model."""

    def __init__(self, store: SessionStore, max_entries: int = 65536) -> None:
        self._store = store
        self._max = max_entries
        self._cache: Dict[str, Sampler] = {}

    def lookup(self, sampler_id: str) -> Optional[Sampler]:
        sampler = self._cache.get(sampler_id)
        if sampler is None:
            sampler = self._store.load_sampler(sampler_id)
            if sampler is not None:
                self.put(sampler)
        return sampler

    def put(self, sampler: Sampler) -> None:
        if len(self._cache) >= self._max:
            self._cache.clear()
        self._cache[sampler.sampler_id] = sampler
