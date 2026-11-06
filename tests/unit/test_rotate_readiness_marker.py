"""scripts/pod_rotate_server.sh calls a rotated server up when its log carries a readiness line;
`python -m training` must print that line, or every rotation of a healthy server exits 1."""
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "pod_rotate_server.sh"


def _readiness_markers() -> list:
    """The strings the script greps the new server's log for, other than its bind-failure check."""
    markers = [m for m in re.findall(r"grep -q '([^']+)' \$SERVER_LOG", SCRIPT.read_text()) if "error" not in m]
    assert markers, f"no readiness grep found in {SCRIPT}"
    return markers


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_server_prints_the_line_the_rotate_script_waits_for(tmp_path):
    port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp_path / "meta"),
           "TINKERCLOUD_CHECKPOINT_BASE": str(tmp_path / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
           "RAY_DISABLE_AUTO_INIT": "1", "NUM_GPUS": "1"}
    log = tmp_path / "server.log"
    proc = subprocess.Popen([sys.executable, "-m", "training", "--backend", "fake", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=str(REPO), env=env, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            assert proc.poll() is None, log.read_text()[-3000:]
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "not healthy: " + log.read_text()[-3000:]
            time.sleep(0.3)
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
    text = log.read_text()
    for marker in _readiness_markers():
        assert marker in text, f"{SCRIPT.name} waits for {marker!r}; the server's log has no such line"
