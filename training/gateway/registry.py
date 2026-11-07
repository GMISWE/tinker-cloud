"""The engines the gateway knows: who registered, where its socket is, when it last heartbeat.

The rows live in SQLite so every gateway worker sees a registration that landed on another
worker; each worker keeps a copy and re-reads the table on a short period (`refresh`), so the
sampling hot path never reads it; a request that finds no engine, or an unreachable one, refreshes
on the spot. A row exists between register and unregister and is UP the whole time.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .dbfile import opened_alone

logger = logging.getLogger(__name__)


class RegistryUnavailable(Exception):
    """The shared file was busy or locked past the timeout (another worker holds it); the rows are as before."""


class RegistryConflict(Exception):
    """A registration the gateway cannot hold alongside the current one."""


@dataclass(frozen=True)
class EngineRow:
    name: str
    backend: str
    gpus: Optional[List[int]]   # None: the engine did not state them (no engines table yet)
    socket: str
    pid: int
    base_model_sampling: bool   # the engine serves bare base_model samplers (create_sampling_session)
    registered_at: float
    last_heartbeat: float

    def as_dict(self, now: float) -> dict:
        return {
            "name": self.name, "backend": self.backend, "gpus": self.gpus, "socket": self.socket,
            "pid": self.pid, "base_model_sampling": self.base_model_sampling, "state": "UP",
            "heartbeat_age_s": round(now - self.last_heartbeat, 3),
        }


_COLUMNS = "name, backend, gpus, socket, pid, base_model_sampling, registered_at, last_heartbeat"


def _row(t: tuple) -> EngineRow:
    name, backend, gpus, socket, pid, base_model_sampling, registered_at, last_heartbeat = t
    return EngineRow(name, backend, None if gpus is None else json.loads(gpus), socket, int(pid),
                     bool(base_model_sampling), float(registered_at), float(last_heartbeat))


class Registry:
    """`path=None` keeps the rows in memory (one process)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path   # None: in memory, one process
        self._conn = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False,
                                     isolation_level=None, timeout=10.0)
        self._conn.execute("PRAGMA busy_timeout=10000")
        if path is not None:
            opened_alone(path, lambda: self._conn.execute("PRAGMA journal_mode=WAL"))
        self._conn.execute("""CREATE TABLE IF NOT EXISTS engines (
            name TEXT PRIMARY KEY, backend TEXT NOT NULL, gpus TEXT, socket TEXT NOT NULL, pid INTEGER NOT NULL,
            registered_at REAL NOT NULL, last_heartbeat REAL NOT NULL,
            base_model_sampling INTEGER NOT NULL DEFAULT 0)""")
        # a file written before engines declared base_model_sampling: its rows are re-registered by
        # their engines, so the column's default only stands until then
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(engines)").fetchall()}
        if "base_model_sampling" not in columns:
            self._conn.execute("ALTER TABLE engines ADD COLUMN base_model_sampling INTEGER NOT NULL DEFAULT 0")
        self._rows: Dict[str, EngineRow] = {}
        self._on_drop: List[Callable[[EngineRow], None]] = []
        self.refresh()

    def close(self) -> None:
        self._conn.close()

    def on_drop(self, fn: Callable[[EngineRow], None]) -> None:
        """Called with a row that stops being current (unregistered or replaced by a new pid), in every worker."""
        self._on_drop.append(fn)

    def register(self, name: str, backend: str, gpus: Optional[List[int]], socket: str, pid: int,
                 base_model_sampling: bool) -> EngineRow:
        now = time.time()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            others = [r[0] for r in self._conn.execute("SELECT name FROM engines WHERE name != ?", (name,)).fetchall()]
            if others:
                raise RegistryConflict(f"engine {others[0]!r} is registered; this gateway forwards to one engine")
            old = self._conn.execute("SELECT pid FROM engines WHERE name = ?", (name,)).fetchone()
            self._conn.execute("INSERT OR REPLACE INTO engines (name, backend, gpus, socket, pid, base_model_sampling, registered_at, last_heartbeat) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (name, backend, None if gpus is None else json.dumps(list(gpus)), socket, pid,
                                int(base_model_sampling), now, now))
            self._conn.execute("COMMIT")
        except BaseException:   # the transaction's own rollback; the error goes on
            self._conn.execute("ROLLBACK")
            raise
        if old is not None:
            logger.warning("Engine %s registered again (pid %d replaces %d)", name, pid, old[0])
        logger.info("Engine %s registered: backend=%s socket=%s pid=%d gpus=%s", name, backend, socket, pid, gpus)
        self.refresh()
        return self._rows[name]

    def heartbeat(self, name: str) -> bool:
        now = time.time()
        if not self._conn.execute("UPDATE engines SET last_heartbeat = ? WHERE name = ?", (now, name)).rowcount:
            return False
        row = self._rows.get(name)
        if row is not None:
            self._rows[name] = EngineRow(row.name, row.backend, row.gpus, row.socket, row.pid, row.base_model_sampling,
                                         row.registered_at, now)
        return True

    def unregister(self, name: str) -> Optional[EngineRow]:
        """The row as the file held it, or None when no such engine is registered (on any worker)."""
        found = self._conn.execute(f"SELECT {_COLUMNS} FROM engines WHERE name = ?", (name,)).fetchall()
        if not found or not self._conn.execute("DELETE FROM engines WHERE name = ?", (name,)).rowcount:
            return None
        logger.info("Engine %s unregistered", name)
        self.refresh()
        return _row(found[0])

    def refresh(self) -> None:
        """Re-read the table; a cached row that is gone or has a new pid is dropped (its callbacks run)."""
        try:
            found = self._conn.execute(f"SELECT {_COLUMNS} FROM engines").fetchall()
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            raise RegistryUnavailable(str(exc)) from exc   # another worker holds the file past the timeout
        fresh = {t[0]: _row(t) for t in found}
        for name, old in self._rows.items():
            new = fresh.get(name)
            if new is None or new.pid != old.pid or new.registered_at != old.registered_at:
                for fn in self._on_drop:
                    fn(old)
        self._rows = fresh

    def get(self, name: str) -> Optional[EngineRow]:
        return self._rows.get(name)

    def rows(self) -> List[EngineRow]:
        return list(self._rows.values())

    def only(self) -> Optional[EngineRow]:
        """The engine a request without an id goes to; None while no engine is registered."""
        rows = list(self._rows.values())
        return rows[0] if rows else None
