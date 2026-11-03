"""The engines the gateway knows: who registered, where its socket is, when it last heartbeat."""
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegistryConflict(Exception):
    """A registration the gateway cannot hold alongside the current one."""


@dataclass
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
    """One row per registered engine. A row exists between register and unregister; it is UP the whole time."""

    def __init__(self) -> None:
        self._rows: Dict[str, EngineRow] = {}
        self._on_drop: List[Callable[[EngineRow], None]] = []

    def on_drop(self, fn: Callable[[EngineRow], None]) -> None:
        """Called with a row that stops being current (unregistered or replaced by a new pid)."""
        self._on_drop.append(fn)

    def register(self, name: str, backend: str, gpus: Optional[List[int]], socket: str, pid: int) -> EngineRow:
        others = [r.name for r in self._rows.values() if r.name != name]
        if others:
            raise RegistryConflict(f"engine {others[0]!r} is registered; this gateway forwards to one engine")
        now = time.monotonic()
        old = self._rows.get(name)
        if old is not None:
            logger.warning("Engine %s registered again (pid %d replaces %d)", name, pid, old.pid)
            self._drop(old)
        row = EngineRow(name, backend, None if gpus is None else list(gpus), socket, pid, now, now)
        self._rows[name] = row
        logger.info("Engine %s registered: backend=%s socket=%s pid=%d gpus=%s", name, backend, socket, pid, gpus)
        return row

    def heartbeat(self, name: str) -> bool:
        row = self._rows.get(name)
        if row is None:
            return False
        row.last_heartbeat = time.monotonic()
        return True

    def unregister(self, name: str) -> Optional[EngineRow]:
        row = self._rows.pop(name, None)
        if row is not None:
            logger.info("Engine %s unregistered", name)
            self._drop(row)
        return row

    def get(self, name: str) -> Optional[EngineRow]:
        return self._rows.get(name)

    def rows(self) -> List[EngineRow]:
        return list(self._rows.values())

    def only(self) -> Optional[EngineRow]:
        """The engine a request without an id goes to; None while no engine is registered."""
        rows = list(self._rows.values())
        return rows[0] if rows else None

    def _drop(self, row: EngineRow) -> None:
        for fn in self._on_drop:
            fn(row)
