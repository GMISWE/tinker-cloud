"""Create-via-load on the engine: the SDK resends a request whose answer it lost, so the same load again is
answered with the first one's future, while that create is held in the queue and after it has run, and the
backend creates the model once; another load naming the model is refused."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
KEY = {"X-API-Key": "k"}
SOURCE = "model_t_0123456789abcdef"


def _env(tmp: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp / "meta"),
            "TINKERCLOUD_CHECKPOINT_BASE": str(tmp / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
            "RAY_DISABLE_AUTO_INIT": "1", "NUM_GPUS": "1",   # the fake backend takes any count; a runner without GPUs detects none
            "FAKE_BACKEND_TRACE": str(tmp / "trace.jsonl"), "FAKE_BACKEND_GATE": str(tmp / "gate")}


def _creates(tmp: Path) -> list:
    trace = [json.loads(line) for line in (tmp / "trace.jsonl").read_text().splitlines()]
    return [t["model_id"] for t in trace if t["op"] == "create_model"]


def _done(c: httpx.Client, submitted: httpx.Response) -> dict:
    assert submitted.status_code == 200, submitted.text
    rid = submitted.json()["request_id"]
    r = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_retried_create_via_load_is_the_first_requests_future(tmp_path):
    sock = tmp_path / "e.sock"
    hold = tmp_path / "gate" / "s:train:1.hold"
    hold.parent.mkdir()
    proc = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake", "--socket", str(sock)],
                            cwd=str(REPO), env=_env(tmp_path), stdout=open(tmp_path / "engine.log", "w"), stderr=subprocess.STDOUT)
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://engine", timeout=30, headers=KEY) as c:
            deadline = time.time() + 60
            while time.time() < deadline:
                assert proc.poll() is None, (tmp_path / "engine.log").read_text()[-3000:]
                try:
                    if c.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            create = {"session_id": "s", "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}
            _done(c, c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": SOURCE}))
            path = _done(c, c.post("/api/v1/save_weights", json={"model_id": SOURCE, "seq_id": 1, "path": "w"}))["path"]
            load = {"session_id": "s", "model_seq_id": 1, "path": path, "optimizer": False}
            hold.touch()
            first = c.post("/api/v1/load_weights", json=load)
            assert first.status_code == 200 and first.json()["model_id"] == "s:train:1", first.text
            retry = c.post("/api/v1/load_weights", json=load)   # the first create is held: queued, not run
            assert retry.status_code == 200 and retry.json() == first.json(), retry.text
            other = c.post("/api/v1/load_weights", json={**load, "optimizer": True})
            assert other.status_code == 400 and "taken by an earlier load_weights" in other.json()["error"], other.text
            assert _creates(tmp_path) == [SOURCE]
            hold.unlink()
            _done(c, first)
            retry = c.post("/api/v1/load_weights", json=load)   # the create has run
            assert retry.status_code == 200 and retry.json() == first.json(), retry.text
            other = c.post("/api/v1/load_weights", json={**load, "optimizer": True})
            assert other.status_code == 400 and "already exists" in other.json()["error"], other.text
        assert _creates(tmp_path) == [SOURCE, "s:train:1"]
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
