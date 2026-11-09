"""A create request the gateway forwarded belongs to the worker process that forwarded it: a refusal
drops the model's row only when no running worker still waits on a request for it. Each worker here
is a real process on the shared session file."""
import os
import subprocess
import sys
import time
from pathlib import Path

from training.gateway.sessions import SessionStore

REPO = Path(__file__).resolve().parents[3]
MODEL = "s:train:0"

# a second gateway worker: records a create request for MODEL, says so, then follows stdin
_WORKER = """
import sys
from training.gateway.sessions import SessionStore
store = SessionStore(sys.argv[1])
store.add_model(sys.argv[2], sys.argv[3], 0, "fake/tiny", "fake")
print("added", flush=True)
for line in sys.stdin:
    if line.strip() == "accept":
        store.settle_model(sys.argv[3], True)
        print("settled", flush=True)
    elif line.strip() == "exit":
        break
"""


def _worker(path: Path, session_id: str) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", _WORKER, str(path), session_id, MODEL], cwd=str(REPO),
                            env={**os.environ, "PYTHONPATH": str(REPO)}, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "added"
    return proc


def _send(proc: subprocess.Popen, line: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def test_a_request_of_a_dead_worker_does_not_keep_a_refused_model(tmp_path):
    """#142 kept the pending count in the file: a worker that died before its answer left it at 1, and
    the SDK's resend, refused, never dropped the row. The dead worker's request holds nothing now."""
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    sid = store.create_session([], {}, "t", None)
    dead = _worker(path, sid)
    _send(dead, "exit")
    assert dead.wait(timeout=30) == 0
    assert store.session_of_model(MODEL) == sid   # its row, from a request nobody answered
    store.add_model(sid, MODEL, 0, "fake/tiny", "fake")   # the resend
    store.settle_model(MODEL, False)                     # refused
    assert store.session_of_model(MODEL) is None
    store.close()


def test_a_request_of_a_dead_worker_nobody_reaped_does_not_keep_a_refused_model(tmp_path):
    """A worker that died is a zombie until its parent reaps it, and a zombie still has its pid: a
    liveness check by pid alone read it as running."""
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    sid = store.create_session([], {}, "t", None)
    dead = _worker(path, sid)
    try:
        _send(dead, "exit")
        deadline = time.time() + 10
        while time.time() < deadline:
            stat = Path(f"/proc/{dead.pid}/stat").read_text()
            if stat[stat.rindex(")") + 2] == "Z":
                break
            time.sleep(0.02)
        assert stat[stat.rindex(")") + 2] == "Z", stat   # exited, not reaped
        store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
        store.settle_model(MODEL, False)
        assert store.session_of_model(MODEL) is None
    finally:
        dead.wait(timeout=30)
        store.close()


def test_a_live_worker_s_request_keeps_the_row_through_another_worker_s_refusal(tmp_path):
    """Two loads of one id: the refused one answers first, on this worker; the other, on a live
    worker, is still pending, then accepted. The row stays throughout and ends an accepted model."""
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    sid = store.create_session([], {}, "t", None)
    live = _worker(path, sid)
    try:
        store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
        store.settle_model(MODEL, False)
        assert store.session_of_model(MODEL) == sid   # the live worker still waits on its request
        _send(live, "accept")
        assert live.stdout is not None and live.stdout.readline().strip() == "settled"
        store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
        store.settle_model(MODEL, False)              # a later refusal of an accepted model
        assert store.session_of_model(MODEL) == sid
    finally:
        _send(live, "exit")
        live.wait(timeout=30)
        store.close()
