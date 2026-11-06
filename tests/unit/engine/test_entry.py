"""`python -m training.engine`: a unix socket and nothing else; a busy socket path is refused; --name is required."""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]


def _env(tmp: Path, port: int) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp / "meta"),
            "TINKERCLOUD_CHECKPOINT_BASE": str(tmp / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
            "TRAINING_PORT": str(port), "RAY_DISABLE_AUTO_INIT": "1"}


def _start(tmp: Path, sock: Path, port: int, log: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake", "--socket", str(sock)],
                            cwd=str(REPO), env=_env(tmp, port), stdout=open(tmp / log, "w"), stderr=subprocess.STDOUT)


def _health(sock: Path) -> httpx.Response:
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), timeout=2) as c:
        return c.get("http://engine/health")


def _wait_healthy(proc: subprocess.Popen, sock: Path, tmp: Path, log: str) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        assert proc.poll() is None, (tmp / log).read_text()[-3000:]
        if sock.exists():
            try:
                if _health(sock).status_code == 200:
                    return
            except httpx.TransportError:
                pass
        time.sleep(0.1)
    raise AssertionError("engine not healthy: " + (tmp / log).read_text()[-3000:])


def _listening_tcp_inodes(pid: int) -> set:
    inodes = set()
    for fd in os.listdir(f"/proc/{pid}/fd"):
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    listening = set()
    for table in ("tcp", "tcp6"):
        with open(f"/proc/{pid}/net/{table}") as f:
            next(f)
            for line in f:
                cols = line.split()
                if cols[3] == "0A" and cols[9] in inodes:   # 0A = LISTEN
                    listening.add(cols[1])
    return listening


def test_engine_serves_its_socket_and_opens_no_tcp_listener(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sock = tmp_path / "t.sock"
    proc = _start(tmp_path, sock, port, "engine.log")
    try:
        _wait_healthy(proc, sock, tmp_path, "engine.log")
        assert _health(sock).json()["status"] == "healthy"
        assert _listening_tcp_inodes(proc.pid) == set()
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0   # TRAINING_PORT is not an engine's port

        second = _start(tmp_path, sock, port, "second.log")
        assert second.wait(timeout=60) == 1
        assert str(sock) in (tmp_path / "second.log").read_text()
        assert _health(sock).status_code == 200   # the first engine is untouched

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0, (tmp_path / "engine.log").read_text()[-3000:]
    finally:
        if proc.poll() is None:
            proc.kill()


def test_sigterm_at_the_first_healthy_answer_still_unregisters(tmp_path):
    """The window between the socket answering and the engine's own signal handlers taking over is closed."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    gw_sock = tmp_path / "gw.sock"
    gateway = subprocess.Popen([sys.executable, "-m", "training.gateway", "--host", "127.0.0.1", "--port", str(port),
                                "--engines-socket", str(gw_sock)], cwd=str(REPO), env=_env(tmp_path, port),
                               stdout=open(tmp_path / "gateway.log", "w"), stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 60
        while not gw_sock.exists() and time.time() < deadline:
            time.sleep(0.1)
        engines = httpx.Client(transport=httpx.HTTPTransport(uds=str(gw_sock)), base_url="http://gateway", timeout=5,
                               headers={"X-API-Key": "k"})
        for trial in range(3):
            sock = tmp_path / f"e{trial}.sock"
            proc = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake",
                                     "--socket", str(sock), "--gateway-socket", str(gw_sock)],
                                    cwd=str(REPO), env=_env(tmp_path, port), stdout=open(tmp_path / f"e{trial}.log", "w"),
                                    stderr=subprocess.STDOUT)
            try:
                _wait_healthy(proc, sock, tmp_path, f"e{trial}.log")
                proc.send_signal(signal.SIGTERM)   # the moment the socket first answers
                assert proc.wait(timeout=30) == 0, (tmp_path / f"e{trial}.log").read_text()[-2000:]
                rows = engines.get("/engines").json()["engines"]
                assert rows == [], (trial, rows)
            finally:
                if proc.poll() is None:
                    proc.kill()
        engines.close()
    finally:
        gateway.terminate()
        gateway.wait(timeout=30)


def test_name_is_required_and_must_fit_inside_an_id():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-m", "training.engine", "--backend", "fake"], cwd=str(REPO),
                       capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 2 and "--name" in r.stderr
    for bad in ("", "Bad-Name", "a b", "x.y"):
        r = subprocess.run([sys.executable, "-m", "training.engine", "--name", bad, "--backend", "fake"], cwd=str(REPO),
                           capture_output=True, text=True, timeout=60, env=env)
        assert r.returncode == 2 and "lowercase letters, digits and underscores" in r.stderr, (bad, r.returncode, r.stderr[-300:])
    r = subprocess.run([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake", "--gpus", "GPU-1a2b,0"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 2 and "comma-separated integers" in r.stderr, r.stderr[-300:]
