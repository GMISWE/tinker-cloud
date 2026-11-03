"""The engine has no sessions and trusts the gateway: session routes are 404 on its socket, a sampling request
naming a sampler needs X-Model-Id (400 without it), a create needs the id the gateway minted, and
resolve_sampler_target answers the gateway's question about a model path."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
KEY = {"X-API-Key": "k"}


def _env(tmp: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp / "meta"),
            "TINKERCLOUD_CHECKPOINT_BASE": str(tmp / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
            "RAY_DISABLE_AUTO_INIT": "1"}


def test_engine_routes_without_sessions(tmp_path):
    sock = tmp_path / "e.sock"
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
            for path in ("/api/v1/create_session", "/api/v1/session_heartbeat", "/api/v1/create_sampling_session", "/api/v1/join_sampling_session"):
                assert c.post(path, json={"session_id": "s"}).status_code == 404, path
            assert c.get("/api/v1/sessions").status_code == 404
            prompt = {"prompt": {"tokens": [1, 2]}, "num_samples": 1, "sampling_params": {"max_tokens": 1}}
            r = c.post("/api/v1/asample", json={**prompt, "sampling_session_id": "s_1"})
            assert r.status_code == 400 and "X-Model-Id" in r.json()["error"], r.text
            r = c.post("/api/v1/asample", json={**prompt, "sampling_session_id": "s_1"}, headers={"X-Model-Id": "model_t_0000000000000000"})
            assert r.status_code == 404 and "no longer exists" in r.json()["error"], r.text
            create = {"session_id": "s", "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}
            r = c.post("/api/v1/create_model", json=create)
            assert r.status_code == 400 and "X-Model-Id" in r.json()["error"], r.text
            r = c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": "model_other_0123456789abcdef"})
            assert r.status_code == 400 and "does not name this engine" in r.json()["error"], r.text
            r = c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": "model_t_0123456789abcdef"})
            assert r.status_code == 200 and r.json()["model_id"] == "model_t_0123456789abcdef", r.text
            rid = r.json()["request_id"]
            assert c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid}).status_code == 200
            r = c.post("/api/v1/asample", json={**prompt, "sampling_session_id": "s_1"}, headers={"X-Model-Id": "model_t_0123456789abcdef", "X-Pinned-Version": "x"})
            assert r.status_code == 400 and "X-Pinned-Version" in r.json()["error"], r.text
            r = c.post("/api/v1/asample", json={**prompt, "sampling_session_id": "s_1"}, headers={"X-Model-Id": "model_t_0123456789abcdef"})
            assert r.status_code == 200, r.text
            # create_sampling_client naming a sampler reads the same headers
            r = c.post("/api/v1/create_sampling_client", json={"sampling_session_id": "s_1"})
            assert r.status_code == 400 and "X-Model-Id" in r.json()["error"], r.text
            r = c.post("/api/v1/create_sampling_client", json={"sampling_session_id": "s_1"}, headers={"X-Model-Id": "model_t_0123456789abcdef", "X-Pinned-Version": "0"})
            assert r.status_code == 200, r.text
            rid = r.json()["request_id"]
            created = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid})
            assert created.status_code == 200 and created.json()["model_path"] == "tinker://model_t_0123456789abcdef", created.text
            r = c.post("/api/v1/resolve_sampler_target", json={"base_model": "fake/tiny"})
            assert r.status_code == 400 and "base-model sampling" in r.json()["error"]
            r = c.post("/api/v1/resolve_sampler_target", json={"model_path": "tinker://model_gone/weights/x"})
            assert r.status_code == 404
            r = c.post("/api/v1/save_weights_for_sampler", json={"model_id": "model_t_0123456789abcdef", "seq_id": 1, "path": "p"})
            rid = r.json()["request_id"]
            saved = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid}).json()
            r = c.post("/api/v1/resolve_sampler_target", json={"model_path": saved["path"]})
            assert r.status_code == 200 and r.json() == {"model_id": "model_t_0123456789abcdef", "pinned_version": 0, "base_model": "fake/tiny"}, r.text
            # an engine with no gateway still saves an ephemeral sampler; nothing routes to it
            r = c.post("/api/v1/save_weights_for_sampler", json={"model_id": "model_t_0123456789abcdef", "seq_id": 2, "sampling_session_seq_id": 0})
            rid = r.json()["request_id"]
            assert "sampling_session_id" in c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid}).json()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
