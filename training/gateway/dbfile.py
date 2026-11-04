"""Opening a SQLite file that N gateway workers open at once."""
import fcntl
from pathlib import Path
from typing import Callable


def opened_alone(path: Path, fn: Callable[[], None]) -> None:
    """Run `fn` holding an exclusive lock on `<path>.lock`. The journal-mode switch and a column check
    that races another worker's ALTER both fail when they collide, and SQLite's busy handler does not
    cover the switch, so the open is serialised here instead."""
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fn()
