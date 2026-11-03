"""The engines the gateway knows: who registered, where its socket is, when it last heartbeat.

The rows live in SQLite so every gateway worker sees a registration that landed on another
worker; each worker keeps a copy and re-reads the table on a short period (`refresh`), so no
request reads it. A row exists between register and unregister and is UP the whole time.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegistryConflict(Exception):
    """A registration the gateway cannot hold alongside the current one."""


@dataclass(frozen=True)
class EngineRow:
    name: str
    backend: str
    gpus: Optional[List[int]]   # None: the engine did not state them (no engines table yet)
    socket: str
    pid: int
    registered_at: float
    last_heartbeat: float

    def as_dict(self, now: float) -> dict:
        return {
            "name": self.name, "backend": self.backend, "gpus": self.gpus, "socket": self.socket,
            "pid": self.pid, "state": "UP", "heartbeat_age_s": round(now - self.last_heartbeat, 3),
        }


class Registry:
    """`path=None` keeps the rows in memory (one process)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._conn = sqlite3.connect(str(path) if path is not None else ":memory:", check_same_thread=False,
                                     isolation_level=None, timeout=10.0)
        if path is not None:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS engines (
            name TEXT PRIMARY KEY, backend TEXT NOT NULL, gpus TEXT, socket TEXT NOT NULL, pid INTEGER NOT NULL,
            registered_at REAL NOT NULL, last_heartbeat REAL NOT NULL)""")
        self._rows: Dict[str, EngineRow] = {}
        self._on_drop: List[Callable[[EngineRow], None]] = []
        self.refresh()

    def close(self) -> None:
        self._conn.close()

    def on_drop(self, fn: Callable[[EngineRow], None]) -> None:
        """Called with a row that stops being current (unregistered or replaced by a new pid), in every worker."""
        self._on_drop.append(fn)

    def register(self, name: str, backend: str, gpus: Optional[List[int]], socket: str, pid: int) -> EngineRow:
        now = time.time()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            others = [r[0] for r in self._conn.execute("SELECT name FROM engines WHERE name != ?", (name,)).fetchall()]
            if others:
                raise RegistryConflict(f"engine {others[0]!r} is registered; this gateway forwards to one engine")
            old = self._conn.execute("SELECT pid FROM engines WHERE name = ?", (name,)).fetchone()
            self._conn.execute("INSERT OR REPLACE INTO engines (name, backend, gpus, socket, pid, registered_at, last_heartbeat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (name, backend, None if gpus is None else json.dumps(list(gpus)), socket, pid, now, now))
            self._conn.execute("COMMIT")
        except BaseException:
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
            self._rows[name] = EngineRow(row.name, row.backend, row.gpus, row.socket, row.pid, row.registered_at, now)
        return True

    def unregister(self, name: str) -> Optional[EngineRow]:
        if not self._conn.execute("DELETE FROM engines WHERE name = ?", (name,)).rowcount:
            return None
        logger.info("Engine %s unregistered", name)
        row = self._rows.get(name)
        self.refresh()
        return row

    def refresh(self) -> None:
        """Re-read the table; a cached row that is gone or has a new pid is dropped (its callbacks run)."""
        fresh: Dict[str, EngineRow] = {}
        for name, backend, gpus, socket, pid, registered_at, last_heartbeat in self._conn.execute(
                "SELECT name, backend, gpus, socket, pid, registered_at, last_heartbeat FROM engines").fetchall():
            fresh[name] = EngineRow(name, backend, None if gpus is None else json.loads(gpus), socket, int(pid),
                                    float(registered_at), float(last_heartbeat))
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
