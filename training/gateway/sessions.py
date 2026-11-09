"""Sessions, samplers and the model registry: one SQLite file under METADATA_DIR shared by every gateway worker.

The workers are processes, so the file is the source of truth and each worker caches only
samplers, which never change once written, and the stamps of the reap claims it took. A session
row is claimed for expiry inside one transaction, so N workers running the reaper free each model
once; a claim its worker did not settle is taken again once it is CLAIM_LEASE_S old.
"""
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dbfile import opened_alone

logger = logging.getLogger(__name__)

SESSION_COLUMNS = ("session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat, finished_at, finish_reason, "
                   "finish_detail, display_name")
SAMPLER_COLUMNS = "sampler_id, session_id, model_id, base_model, model_path, pinned_version, client_counter"
# A model claimed for deletion is its worker's until the delete settles the claim (acknowledged, or
# released for a retry). A worker killed mid-free settles nothing, so a claim this old is taken again;
# so is one whose delete is still running, and the engine then runs that teardown a second time.
CLAIM_LEASE_S = 600.0


_process: Tuple[int, str] = (0, "")


def _worker() -> str:
    """This process as a holder of pending create requests: its pid and a nonce drawn once per
    process, so a later process that is given the same pid is a different worker."""
    global _process
    if _process[0] != os.getpid():   # first call, or the first after a fork
        _process = (os.getpid(), uuid.uuid4().hex[:8])
    return f"{_process[0]}:{_process[1]}"


def _worker_alive(worker: str) -> bool:
    """Whether the process that recorded `worker` still runs. Every gateway worker runs on one host,
    so its pid is checked here; a reused pid reads as alive, which keeps a row, never drops one."""
    pid = int(worker.split(":", 1)[0])
    if pid == os.getpid():
        return worker == _worker()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:   # a process of another user holds the pid: it runs
        return True
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Sampler:
    sampler_id: str
    session_id: str
    model_id: Optional[str]        # None: a base-model sampler, or a pre-split row (base_model None too)
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
    display_name: Optional[str]   # None: the client gave none
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
        self._claims: Dict[str, str] = {}   # model_id -> claimed_at of this store's claim; retry_reap releases only that
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
        c.execute("""CREATE TABLE IF NOT EXISTS model_reaps (
            model_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, claimed_at TEXT)""")
        # a model's create requests that have no answer yet, per worker that forwarded them
        c.execute("""CREATE TABLE IF NOT EXISTS model_requests (
            model_id TEXT NOT NULL, worker TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (model_id, worker))""")
        # columns added after the first release; a pre-existing file lacks them
        self._add_missing_columns("sessions", {"finished_at": "TEXT", "finish_reason": "TEXT", "finish_detail": "TEXT",
                                               "display_name": "TEXT"})
        self._add_missing_columns("samplers", {"client_counter": "INTEGER NOT NULL DEFAULT 0", "pinned_version": "INTEGER"})
        # confirmed: the engine accepted a request that creates the model (a row from before the column
        # is an accepted model). A file written by an earlier release also has an unused `pending` column.
        self._add_missing_columns("session_models", {"engine": "TEXT", "confirmed": "INTEGER NOT NULL DEFAULT 1"})
        c.execute("CREATE INDEX IF NOT EXISTS idx_samplers_session ON samplers(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_heartbeat ON sessions(last_heartbeat)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_models_seq ON session_models(session_id, model_seq_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_models_model ON session_models(model_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_model_reaps_claimed ON model_reaps(claimed_at)")

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

    def create_session(self, tags: List[str], user_metadata: Dict[str, Any], sdk_version: str,
                       display_name: Optional[str]) -> str:
        session_id = str(uuid.uuid4())
        now = _now()
        self._write("INSERT INTO sessions (session_id, sdk_version, tags, user_metadata, created_at, last_heartbeat, display_name) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, sdk_version, json.dumps(tags), json.dumps(user_metadata), now, now, display_name))
        return session_id

    def exists(self, session_id: str) -> bool:
        return bool(self._read("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)))

    def get_session(self, session_id: str) -> Optional[Session]:
        rows = self._read(f"SELECT {SESSION_COLUMNS} FROM sessions WHERE session_id = ?", (session_id,))
        if not rows:
            return None
        return self._session(rows[0])

    def _session(self, r: Tuple) -> Session:
        models = self._read("SELECT model_id FROM session_models WHERE session_id = ? ORDER BY model_seq_id ASC", (r[0],))
        samplers = self._read("SELECT sampler_id FROM samplers WHERE session_id = ? ORDER BY created_at ASC", (r[0],))
        return Session(session_id=r[0], sdk_version=r[1], tags=json.loads(r[2]), user_metadata=json.loads(r[3]),
                       created_at=r[4], last_heartbeat=r[5],
                       finished_at=r[6], finish_reason=r[7], finish_detail=r[8], display_name=r[9],
                       model_ids=[m[0] for m in models], sampler_ids=[s[0] for s in samplers])

    def sessions_newest_first(self) -> List[Session]:
        """Every live session, newest first (ties: session_id descending): the search order."""
        return [self._session(r) for r in
                self._read(f"SELECT {SESSION_COLUMNS} FROM sessions ORDER BY created_at DESC, session_id DESC")]

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

    def claim_expired(self, timeout_s: float, lease_s: float = CLAIM_LEASE_S) -> List[Tuple[str, List[str]]]:
        """Drop sessions silent for longer than `timeout_s` and claim their models for deletion.
        An active model claim stays hidden from the other workers until a failed delete releases it,
        or until it is `lease_s` old (its worker died or its delete is stuck): the model is claimed again."""
        now = datetime.now(timezone.utc)
        stamp = now.isoformat()
        cutoff = (now - timedelta(seconds=timeout_s)).isoformat()
        abandoned = (now - timedelta(seconds=lease_s)).isoformat()
        claimed: Dict[str, List[str]] = {}
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for (session_id,) in self._read("SELECT session_id FROM sessions WHERE last_heartbeat < ?", (cutoff,)):
                models = [m[0] for m in self._read("SELECT model_id FROM session_models WHERE session_id = ?", (session_id,))]
                for model_id in models:
                    self._write("INSERT OR IGNORE INTO model_reaps (model_id, session_id) VALUES (?, ?)",
                                (model_id, session_id))
                self._write("DELETE FROM samplers WHERE session_id = ?", (session_id,))
                self._write("DELETE FROM session_models WHERE session_id = ?", (session_id,))
                self._write("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            rows = self._read("SELECT session_id, model_id, claimed_at FROM model_reaps "
                              "WHERE claimed_at IS NULL OR claimed_at < ? ORDER BY session_id, model_id", (abandoned,))
            for session_id, model_id, claimed_at in rows:
                if claimed_at is not None:
                    logger.warning("Model %s was claimed for deletion at %s and never settled (its worker died "
                                   "or its delete is stuck); claiming it again", model_id, claimed_at)
                self._write("UPDATE model_reaps SET claimed_at = ? WHERE model_id = ?", (stamp, model_id))
                claimed.setdefault(session_id, []).append(model_id)
            self._conn.execute("COMMIT")
        except BaseException:   # the transaction's own rollback; the error goes on
            self._conn.execute("ROLLBACK")
            raise
        for model_ids in claimed.values():
            self._claims.update(dict.fromkeys(model_ids, stamp))
        return list(claimed.items())

    def retry_reap(self, model_id: str) -> None:
        """Make a failed model delete claimable again. Only this store's own claim is released: once
        the lease ran out another worker may have claimed the model, and that claim stands."""
        claimed_at = self._claims.pop(model_id, None)
        if claimed_at is not None:
            self._write("UPDATE model_reaps SET claimed_at = NULL WHERE model_id = ? AND claimed_at = ?",
                        (model_id, claimed_at))

    # ------------------------------------------------------------------ models

    def add_model(self, session_id: str, model_id: str, model_seq_id: int, base_model: Optional[str], engine: str) -> None:
        """Record a model at submit, and one more request pending for it from this worker: a retried
        create-via-load, or another load racing it, finds the row the first one wrote and leaves it as
        it is. settle_model ends each request. A create the engine fails later is dropped when the
        engine reports it (a report that itself fails leaves the row to its delete or the reaper); a
        delete of a model the engine never had gets the engine's 404 and drops the row."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._write("INSERT INTO session_models (session_id, model_id, model_seq_id, base_model, created_at, engine, confirmed) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0) ON CONFLICT (session_id, model_id) DO NOTHING",
                        (session_id, model_id, model_seq_id, base_model, _now(), engine))
            self._write("INSERT INTO model_requests (model_id, worker, n) VALUES (?, ?, 1) "
                        "ON CONFLICT (model_id, worker) DO UPDATE SET n = n + 1", (model_id, _worker()))
            self._conn.execute("COMMIT")
        except BaseException:   # the transaction's own rollback; the error goes on
            self._conn.execute("ROLLBACK")
            raise

    def settle_model(self, model_id: str, accepted: Optional[bool]) -> None:
        """A request add_model counted has its answer. True: the engine accepted it, and the row stays.
        False: the engine refused it, and the row goes unless the engine accepted another request for the
        model or a running worker still waits on one; two loads of one create-via-load id race, and the
        refused one may answer first or last. A request whose worker died unanswered holds nothing.
        None: no answer (the engine unreachable), and the row stays for its delete or the reaper. A row
        that is gone stays gone: the engine reported the create failed."""
        worker = _worker()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._write("UPDATE model_requests SET n = n - 1 WHERE model_id = ? AND worker = ?", (model_id, worker))
            self._write("DELETE FROM model_requests WHERE model_id = ? AND worker = ? AND n <= 0", (model_id, worker))
            if accepted is True:
                self._write("UPDATE session_models SET confirmed = 1 WHERE model_id = ?", (model_id,))
            elif accepted is False:
                holders = [w for (w,) in self._read("SELECT worker FROM model_requests WHERE model_id = ?", (model_id,))]
                for gone in [w for w in holders if not _worker_alive(w)]:
                    self._write("DELETE FROM model_requests WHERE model_id = ? AND worker = ?", (model_id, gone))
                    holders.remove(gone)
                if not holders:
                    self._write("DELETE FROM session_models WHERE model_id = ? AND confirmed = 0", (model_id,))
            self._conn.execute("COMMIT")
        except BaseException:   # the transaction's own rollback; the error goes on
            self._conn.execute("ROLLBACK")
            raise

    def remove_model(self, model_id: str) -> Optional[str]:
        """Drop the model and pending reap rows; the session that owned it, or None."""
        rows = self._read("SELECT session_id FROM session_models WHERE model_id = ?", (model_id,))
        self._write("DELETE FROM model_reaps WHERE model_id = ?", (model_id,))
        self._write("DELETE FROM model_requests WHERE model_id = ?", (model_id,))
        self._claims.pop(model_id, None)
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
        except BaseException:   # the transaction's own rollback; the error goes on
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
        if sampler is None:   # the miss: one store read, cached when found (the warm path reads nothing)
            sampler = self._store.load_sampler(sampler_id)
            if sampler is not None:
                self.put(sampler)
        return sampler

    def put(self, sampler: Sampler) -> None:
        if len(self._cache) >= self._max:
            self._cache.clear()
        self._cache[sampler.sampler_id] = sampler
