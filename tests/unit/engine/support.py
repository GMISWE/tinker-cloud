"""One fake-backend engine process for the engine tests: started on its own socket with the backend's
trace and gate under a temp dir, a source model created and saved, stopped with SIGTERM."""
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import httpx

REPO = Path(__file__).resolve().parents[3]
KEY = {"X-API-Key": "k"}
SOURCE = "model_t_0123456789abcdef"


def env(tmp: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp / "meta"),
            "TINKERCLOUD_CHECKPOINT_BASE": str(tmp / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
            "RAY_DISABLE_AUTO_INIT": "1", "NUM_GPUS": "1",   # the fake backend takes any count; a runner without GPUs detects none
            "FAKE_BACKEND_TRACE": str(tmp / "trace.jsonl"), "FAKE_BACKEND_GATE": str(tmp / "gate")}


def traced(tmp: Path, op: str) -> list:
    """The model ids of the backend calls `op` in the trace, in order."""
    trace = [json.loads(line) for line in (tmp / "trace.jsonl").read_text().splitlines()]
    return [t["model_id"] for t in trace if t["op"] == op]


def done(c: httpx.Client, submitted: httpx.Response) -> dict:
    assert submitted.status_code == 200, submitted.text
    rid = submitted.json()["request_id"]
    r = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid})
    assert r.status_code == 200, r.text
    return r.json()


def client(tmp: Path) -> httpx.Client:
    return httpx.Client(transport=httpx.HTTPTransport(uds=str(tmp / "e.sock")), base_url="http://engine", timeout=30, headers=KEY)


@contextlib.contextmanager
def engine(tmp: Path, gateway_socket: Optional[Path] = None) -> Iterator[Tuple[httpx.Client, str]]:
    """A healthy engine with SOURCE created and saved; yields a client and SOURCE's checkpoint path.
    The backend's gate dir `tmp/gate` exists."""
    (tmp / "gate").mkdir(exist_ok=True)
    args = ["--gateway-socket", str(gateway_socket)] if gateway_socket is not None else []
    proc = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake",
                             "--socket", str(tmp / "e.sock"), *args],
                            cwd=str(REPO), env=env(tmp), stdout=open(tmp / "engine.log", "w"), stderr=subprocess.STDOUT)
    try:
        with client(tmp) as c:
            deadline = time.time() + 60
            while time.time() < deadline:
                assert proc.poll() is None, (tmp / "engine.log").read_text()[-3000:]
                try:
                    if c.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            create = {"session_id": "s", "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}
            done(c, c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": SOURCE}))
            yield c, done(c, c.post("/api/v1/save_weights", json={"model_id": SOURCE, "seq_id": 1, "path": "w"}))["path"]
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
