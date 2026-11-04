"""Opening a SQLite file that N gateway workers open at once."""
import fcntl
from pathlib import Path
from typing import Callable


def opened_alone(path: Path, fn: Callable[[], None]) -> None:
    """Run `fn` holding an exclusive lock on `<path>.lock`. Opening one new file from several
    connections at once was observed to fail at the journal-mode switch ("database is locked",
    4 of 40 eight-thread opens under CPU load; 0 of 40 with this lock) and to add a migrated
    column twice; the busy timeout did not retry the switch in that experiment."""
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fn()
