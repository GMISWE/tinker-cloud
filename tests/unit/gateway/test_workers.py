"""`python -m training.gateway --workers 3`: one registration serves every worker; a sampler minted through one
connection is routed on every other; the engine's exit is 503 on every worker within a second."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
KEY = "k"


def _env(tmp: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp / "meta"),
            "TINKERCLOUD_CHECKPOINT_BASE": str(tmp / "ckpt"), "TINKER_API_KEY": KEY, "SESSION_TIMEOUT_S": "-1",
            "RAY_DISABLE_AUTO_INIT": "1", "NUM_GPUS": "1"}   # the fake backend takes any count; a runner without GPUs detects none


def _fresh(port: int, method: str, path: str, **kw) -> httpx.Response:
    """A new connection per call: the kernel hands each accept to whichever worker is waiting."""
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", headers={"X-API-Key": KEY}, timeout=30) as c:
        return c.request(method, path, **kw)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(proc: subprocess.Popen, ok, log: Path, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        assert proc.poll() is None, log.read_text()[-3000:]
        try:
            if ok():
                return
        except (httpx.TransportError, OSError):
            pass
        time.sleep(0.1)
    raise AssertionError("not ready: " + log.read_text()[-3000:])


def _future(port: int, request_id: str) -> dict:
    r = _fresh(port, "POST", f"/api/v1/retrieve_future/{request_id}", json={"request_id": request_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_three_workers_share_the_registry_the_sessions_and_the_samplers(tmp_path):
    port = _free_port()
    gw_sock, e_sock = tmp_path / "gw.sock", tmp_path / "e.sock"
    gateway = subprocess.Popen([sys.executable, "-m", "training.gateway", "--host", "127.0.0.1", "--port", str(port),
                                "--engines-socket", str(gw_sock), "--workers", "3"],
                               cwd=str(REPO), env=_env(tmp_path), stdout=open(tmp_path / "gateway.log", "w"), stderr=subprocess.STDOUT)
    engine = None
    try:
        _wait(gateway, lambda: _fresh(port, "GET", "/health").status_code == 503, tmp_path / "gateway.log")
        engine = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "fake", "--backend", "fake",
                                   "--socket", str(e_sock), "--gateway-socket", str(gw_sock), "--heartbeat-interval", "1"],
                                  cwd=str(REPO), env=_env(tmp_path), stdout=open(tmp_path / "engine.log", "w"), stderr=subprocess.STDOUT)
        _wait(engine, lambda: _fresh(port, "GET", "/health").status_code == 200, tmp_path / "engine.log")
        time.sleep(1.0)   # every worker's registry refresh has run
        statuses = [_fresh(port, "GET", "/health").status_code for _ in range(60)]
        assert statuses == [200] * 60, statuses
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(gw_sock)), base_url="http://gateway", timeout=5,
                          headers={"X-API-Key": KEY}) as eng:
            for _ in range(6):
                rows = eng.get("/engines").json()["engines"]
                assert [r["name"] for r in rows] == ["fake"], rows
        # a session, a model, an ephemeral sampler and a created sampler, each through a fresh connection
        sid = _fresh(port, "POST", "/api/v1/create_session", json={"tags": [], "user_metadata": {}, "sdk_version": "t"}).json()["session_id"]
        assert all(_fresh(port, "POST", "/api/v1/session_heartbeat", json={"session_id": sid}).status_code == 200 for _ in range(6))
        r = _fresh(port, "POST", "/api/v1/create_model", json={"session_id": sid, "model_seq_id": 0, "base_model": "fake/tiny",
                                                               "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}).json()
        _future(port, r["request_id"])
        model_id = r["model_id"]
        r = _fresh(port, "POST", "/api/v1/save_weights_for_sampler", json={"model_id": model_id, "seq_id": 1, "sampling_session_seq_id": 0}).json()
        ephemeral = _future(port, r["request_id"])["sampling_session_id"]
        r = _fresh(port, "POST", "/api/v1/save_weights_for_sampler", json={"model_id": model_id, "seq_id": 2, "path": "named"}).json()
        path = _future(port, r["request_id"])["path"]
        created = _fresh(port, "POST", "/api/v1/create_sampling_session", json={"session_id": sid, "sampling_session_seq_id": 1, "model_path": path}).json()["sampling_session_id"]
        prompt = {"prompt": {"tokens": [1, 2, 3]}, "num_samples": 1, "sampling_params": {"max_tokens": 2}}
        for sampler in (ephemeral, created):
            for seq in range(12):
                r = _fresh(port, "POST", "/api/v1/asample", json={**prompt, "sampling_session_id": sampler, "seq_id": seq})
                assert r.status_code == 200, (sampler, r.text)
                assert r.json()["model_id"] == model_id
        assert sorted(_fresh(port, "GET", f"/api/v1/sessions/{sid}").json()["sampler_ids"]) == sorted([ephemeral, created])
        assert [_fresh(port, "POST", "/api/v1/join_sampling_session", json={"sampling_session_id": created}).json()["client_counter"] for _ in range(3)] == [1, 2, 3]
        # a create that fails inside the engine's task leaves no row behind, on whichever worker answers
        cls = tmp_path / "fake-cls"
        cls.mkdir()
        (cls / "config.json").write_text('{"architectures": ["FakeForSequenceClassification"], "num_labels": 3, "max_position_embeddings": 2048}')
        r = _fresh(port, "POST", "/api/v1/create_model", json={"session_id": sid, "model_seq_id": 9, "base_model": str(cls), "lora_config": {"rank": 2}}).json()
        failed = _fresh(port, "POST", f"/api/v1/retrieve_future/{r['request_id']}", json={"request_id": r["request_id"]})
        assert failed.status_code == 400, failed.text
        assert all(r["model_id"] not in _fresh(port, "GET", f"/api/v1/sessions/{sid}").json()["training_run_ids"] for _ in range(6))
        # the engine leaves: every worker answers 503 within a second
        engine.send_signal(signal.SIGTERM)
        assert engine.wait(timeout=30) == 0
        t0 = time.monotonic()
        statuses = [_fresh(port, "POST", "/api/v1/asample", json={**prompt, "sampling_session_id": created, "seq_id": 99}).status_code for _ in range(60)]
        assert statuses == [503] * 60 and time.monotonic() - t0 < 1.0 + 60 * 0.05, statuses
    finally:
        if engine is not None and engine.poll() is None:
            engine.kill()
        gateway.send_signal(signal.SIGTERM)
        try:
            gateway.wait(timeout=30)
        except subprocess.TimeoutExpired:
            gateway.kill()
