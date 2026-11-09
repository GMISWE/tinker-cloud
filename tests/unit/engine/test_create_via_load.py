"""Create-via-load on the engine: the SDK resends a request whose answer it lost, so the same load again is
answered with the first one's future, while that create is held in the queue and after it has run, and the
backend creates the model once; another load naming the model is refused."""
import contextlib
import http.server
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

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


class _Gateway(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """The gateway's engines face as the engine sees it: every call answered 200, each `model_failed`
    (DELETE /engines/<name>/models/<id>) recorded, and the reply to the first one held while `hold` is set."""
    daemon_threads = True

    def __init__(self, path: str) -> None:
        self.reports: list = []
        self.hold = threading.Event()
        self.released = threading.Event()
        super().__init__(path, _GatewayHandler)


class _GatewayHandler(http.server.BaseHTTPRequestHandler):
    server: _Gateway

    def _answer(self) -> None:
        length = int(self.headers.get("content-length") or 0)   # a bodiless DELETE carries no content-length
        self.rfile.read(length)
        if self.command == "DELETE" and "/models/" in self.path:
            self.server.reports.append(self.path.rsplit("/", 1)[1])
            if len(self.server.reports) == 1 and self.server.hold.is_set():
                self.server.released.wait(timeout=30)
        body = b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_DELETE = _answer

    def address_string(self) -> str:   # a unix socket has no peer address
        return "engine"

    def log_message(self, format, *args) -> None:   # noqa: A002 (the base class's name)
        pass


@contextlib.contextmanager
def _engine(tmp: Path, gateway_socket: Optional[Path] = None):
    """A fake-backend engine on its own socket, healthy; yields a client and the source checkpoint path."""
    sock = tmp / "e.sock"
    (tmp / "gate").mkdir(exist_ok=True)
    args = ["--gateway-socket", str(gateway_socket)] if gateway_socket is not None else []
    proc = subprocess.Popen([sys.executable, "-m", "training.engine", "--name", "t", "--backend", "fake", "--socket", str(sock), *args],
                            cwd=str(REPO), env=_env(tmp), stdout=open(tmp / "engine.log", "w"), stderr=subprocess.STDOUT)
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://engine", timeout=30, headers=KEY) as c:
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
            _done(c, c.post("/api/v1/create_model", json=create, headers={"X-Model-Id": SOURCE}))
            yield c, _done(c, c.post("/api/v1/save_weights", json={"model_id": SOURCE, "seq_id": 1, "path": "w"}))["path"]
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_resend_with_the_failed_create_report_in_flight_reports_too(tmp_path):
    """The failed create is marked on its future before it is reported. A resend that arrives while the
    first report's answer is still on its way (the gateway already dropped the row, the resend recorded
    it again, the future is still pending) reports too, so the row goes again."""
    gw_sock = tmp_path / "gw.sock"
    gateway = _Gateway(str(gw_sock))
    threading.Thread(target=gateway.serve_forever, daemon=True).start()
    try:
        with _engine(tmp_path, gw_sock) as (c, path):
            load = {"session_id": "s", "model_seq_id": 1, "path": path, "optimizer": False}
            (tmp_path / "gate" / "s:train:1.fail").touch()
            gateway.hold.set()
            first = c.post("/api/v1/load_weights", json=load)
            assert first.status_code == 200, first.text
            deadline = time.time() + 30
            while not gateway.reports and time.time() < deadline:
                time.sleep(0.02)
            assert gateway.reports == ["s:train:1"]   # reported; its answer held
            retry = c.post("/api/v1/load_weights", json=load)
            assert retry.status_code == 200 and retry.json() == first.json(), retry.text
            assert gateway.reports == ["s:train:1", "s:train:1"]
            gateway.released.set()
            rid = first.json()["request_id"]
            failed = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid})
            assert failed.status_code == 400 and "failed by" in failed.text, failed.text   # the create's own error
    finally:
        gateway.released.set()
        gateway.shutdown()
        gateway.server_close()


def test_a_resend_after_a_cleanup_of_futures_is_still_the_first_request(tmp_path):
    """cleanup_futures at any age used to delete the pending create-via-load future, and the resend
    then filed a second create of the model."""
    with _engine(tmp_path) as (c, path):
        load = {"session_id": "s", "model_seq_id": 1, "path": path, "optimizer": False}
        hold = tmp_path / "gate" / "s:train:1.hold"
        hold.touch()
        first = c.post("/api/v1/load_weights", json=load)
        assert first.status_code == 200, first.text
        assert c.post("/api/v1/cleanup_futures", json={"max_age_hours": 0}).status_code == 200
        retry = c.post("/api/v1/load_weights", json=load)
        assert retry.status_code == 200 and retry.json() == first.json(), retry.text
        hold.unlink()
        _done(c, first)
    assert _creates(tmp_path) == [SOURCE, "s:train:1"]


def test_a_resend_after_delete_of_the_model_is_refused_naming_the_deletion(tmp_path):
    """The first request's future is completed, but its model is gone: answering with that future
    told the SDK the model exists, and its next call failed on a model that does not."""
    with _engine(tmp_path) as (c, path):
        load = {"session_id": "s", "model_seq_id": 1, "path": path, "optimizer": False}
        _done(c, c.post("/api/v1/load_weights", json=load))
        assert c.post("/api/v1/delete_model", json={"model_id": "s:train:1"}).status_code == 200
        retry = c.post("/api/v1/load_weights", json=load)
        assert retry.status_code == 400 and "was deleted" in retry.json()["error"], retry.text
    assert _creates(tmp_path) == [SOURCE, "s:train:1"]


def test_a_resend_after_delete_of_a_model_whose_load_failed_is_refused_naming_the_deletion(tmp_path):
    """The create worked and the load into it failed, so the future is failed but no create failed;
    the model is then deleted. The resend is refused like any resend after a delete, not reported as
    a failed create and answered with the failed future."""
    with _engine(tmp_path) as (c, path):
        load = {"session_id": "s", "model_seq_id": 1, "path": path, "optimizer": False}
        (tmp_path / "gate" / "s:train:1.load.fail").touch()
        first = c.post("/api/v1/load_weights", json=load)
        assert first.status_code == 200, first.text
        rid = first.json()["request_id"]
        failed = c.post(f"/api/v1/retrieve_future/{rid}", json={"request_id": rid})
        assert failed.status_code == 400 and "load into s:train:1 failed" in failed.text, failed.text
        assert c.post("/api/v1/delete_model", json={"model_id": "s:train:1"}).status_code == 200
        retry = c.post("/api/v1/load_weights", json=load)
        assert retry.status_code == 400 and "was deleted" in retry.json()["error"], retry.text
    assert _creates(tmp_path) == [SOURCE, "s:train:1"]
