"""One delete per model at a time on the engine: a second delete of a model whose backend delete is
still running is answered 409 and never reaches the backend, whose teardown would run twice."""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
KEY = {"X-API-Key": "k"}
MODEL = "model_t_0123456789abcdef"


def _deletes(tmp: Path) -> list:
    trace = [json.loads(line) for line in (tmp / "trace.jsonl").read_text().splitlines()]
    return [t["model_id"] for t in trace if t["op"] == "delete_model"]


def test_a_second_delete_while_the_first_runs_is_409_and_the_backend_deletes_once(tmp_path):
    sock = tmp_path / "e.sock"
    hold = tmp_path / "gate" / f"{MODEL}.delete.hold"
    hold.parent.mkdir()
    env = {**os.environ, "PYTHONPATH": str(REPO), "METADATA_DIR": str(tmp_path / "meta"),
           "TINKERCLOUD_CHECKPOINT_BASE": str(tmp_path / "ckpt"), "TINKER_API_KEY": "k", "SESSION_TIMEOUT_S": "-1",
           "RAY_DISABLE_AUTO_INIT": "1", "NUM_GPUS": "1",
           "FAKE_BACKEND_TRACE": str(tmp_path / "trace.jsonl"), "FAKE_BACKEND_GATE": str(hold.parent)}
    proc = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake", "--socket", str(sock)],
                            cwd=str(REPO), env=env, stdout=open(tmp_path / "engine.log", "w"), stderr=subprocess.STDOUT)
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
            create = {"session_id": "s", "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2},
                      "optimizer_config": {"type": "adamw"}}
            rid = c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": MODEL}).json()["request_id"]
            assert c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid}).status_code == 200
            hold.touch()
            answers: "queue.Queue[httpx.Response]" = queue.Queue()

            def delete() -> None:
                with httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://engine", timeout=30,
                                  headers=KEY) as own:
                    answers.put(own.post("/api/v1/delete_model", json={"model_id": MODEL}))

            threads = [threading.Thread(target=delete) for _ in range(2)]
            for t in threads:
                t.start()
            # one delete is held inside the backend; the other must not wait for it
            refused = answers.get(timeout=10)
            assert refused.status_code == 409 and "delete is in progress" in refused.json()["error"], refused.text
            hold.unlink()
            done = answers.get(timeout=30)
            assert done.status_code == 200, done.text
            for t in threads:
                t.join(timeout=30)
            after = c.post("/api/v1/delete_model", json={"model_id": MODEL})
            assert after.status_code == 404, after.text
        assert _deletes(tmp_path) == [MODEL]
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
