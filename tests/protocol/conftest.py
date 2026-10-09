"""Protocol suite: a real `tinker` SDK against a live server on the fake backend.

No GPU, no Ray. One server per test session; each test creates its own models.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
API_KEY = "tml-protocol-test-key"
SIGNING_KEY = "protocol-test-url-signing-key"  # tests forge expired download tokens with it


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, tmp: Path):
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.metadata_dir = tmp / "metadata"
        self.checkpoint_base = tmp / "checkpoints"
        self.trace_path = tmp / "trace.jsonl"
        self.log_path = tmp / "server.log"
        for d in (self.metadata_dir, self.checkpoint_base):
            d.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            "METADATA_DIR": str(self.metadata_dir),
            "TINKERCLOUD_CHECKPOINT_BASE": str(self.checkpoint_base),
            "TINKERCLOUD_BACKEND": "fake",
            "TINKER_API_KEY": API_KEY,
            "FAKE_BACKEND_TRACE": str(self.trace_path),
            "TINKERCLOUD_URL_SIGNING_KEY": SIGNING_KEY,
            "TRAINING_HOST": "127.0.0.1",
            "TRAINING_PORT": str(self.port),
            "SESSION_TIMEOUT_S": "-1",
            "RAY_DISABLE_AUTO_INIT": "1",
        }
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "training.api", "--backend", "fake",
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=str(REPO_ROOT), env=env, stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_healthy(self, timeout_s: float = 60.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; log:\n{self.log_path.read_text()[-4000:]}")
            try:
                if requests.get(self.base_url + "/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"server not healthy in {timeout_s}s; log:\n{self.log_path.read_text()[-4000:]}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()

    def trace(self):
        if not self.trace_path.exists():
            return []
        return [json.loads(line) for line in self.trace_path.read_text().splitlines() if line.strip()]

    def post_raw(self, path: str, data: bytes, headers: dict, key: str | None = API_KEY, timeout: float = 30):
        """POST a prepared body (proto / zstd) with explicit headers."""
        h = {"X-API-Key": key, **headers} if key else dict(headers)
        return requests.post(self.base_url + path, data=data, headers=h, timeout=timeout)

    def post(self, path: str, body: dict, key: str | None = API_KEY, **kw):
        headers = {"X-API-Key": key} if key else {}
        return requests.post(self.base_url + path, json=body, headers=headers, timeout=kw.pop("timeout", 30))


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    srv = Server(tmp_path_factory.mktemp("protocol"))
    try:
        srv.wait_healthy()
        yield srv
    finally:
        srv.stop()


@pytest.fixture(scope="session")
def service_client(server):
    os.environ["TINKER_BASE_URL"] = server.base_url
    os.environ["TINKER_API_KEY"] = API_KEY
    import tinker
    return tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY)


def make_datum(tokens):
    import tinker
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=[1.0] * (len(tokens) - 1), dtype="float32", shape=[len(tokens) - 1]),
            "target_tokens": tinker.TensorData(data=tokens[1:], dtype="int64", shape=[len(tokens) - 1]),
        },
    )
