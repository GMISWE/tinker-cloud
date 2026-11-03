"""Fixtures for the gateway tests: a recording engine on a unix socket and a gateway driven in-process."""
import asyncio
import contextlib
import os
from typing import Any, Dict, List

import httpx
import orjson
import uvicorn

from training.config import TrainingConfig
from training.gateway.app import build_app, engines_face
from training.gateway.registry import Registry
from training.gateway.sockets import bind_unix_socket

API_KEY = "gw-test-key"
VARIANTS = [("asgi", "h11"), ("fastapi", "httpx")]


def config(server: str, client: str, **gateway: Any) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.auth.api_key = API_KEY
    cfg.gateway.server = server
    cfg.gateway.client = client
    cfg.gateway.connect_timeout_s = 1.0
    for k, v in gateway.items():
        setattr(cfg.gateway, k, v)
    return cfg


class RecordingEngine:
    """An ASGI app standing in for an engine. `/api/v1/echo` returns the body; `/api/v1/status/<n>` answers n
    with a JSON error body; `/api/v1/stream/<mb>` streams that many MB; anything else answers 200 JSON."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body", False):
                break
        headers = {k.decode(): v.decode("latin-1") for k, v in scope["headers"]}
        path = scope["path"]
        self.requests.append({"method": scope["method"], "path": path, "query": scope["query_string"].decode(),
                              "headers": headers, "body": body})
        if path.startswith("/api/v1/status/"):
            status = int(path.rsplit("/", 1)[1])
            doc = orjson.dumps({"error": f"engine says {status}", "status_code": status, "path": path})
            await _respond(send, status, [(b"content-type", b"application/json")], doc)
        elif path == "/api/v1/echo":
            ct = headers.get("content-type", "application/octet-stream").encode()
            await _respond(send, 200, [(b"content-type", ct), (b"x-engine", b"yes")], body)
        elif path.startswith("/api/v1/stream/"):
            mb = int(path.rsplit("/", 1)[1])
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/octet-stream")]})
            chunk = bytes(range(256)) * 4096   # 1 MB
            for _ in range(mb):
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b""})
        else:
            await _respond(send, 200, [(b"content-type", b"application/json")], orjson.dumps({"ok": True, "path": path}))


async def _respond(send, status: int, headers, body: bytes) -> None:
    headers = list(headers) + [(b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


@contextlib.asynccontextmanager
async def serve_uds(app, path: str):
    """uvicorn on a unix socket, in this event loop, until the block ends."""
    sock = bind_unix_socket(path)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off", timeout_keep_alive=300))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started and not task.done():
        await asyncio.sleep(0.01)
    if task.done():
        await task
    try:
        yield server
    finally:
        server.should_exit = True
        await task


class Harness:
    """A gateway (one variant) with its registry, an engine on a socket, and HTTP clients for both faces."""

    def __init__(self, tmp: str, server: str, client: str, **gateway: Any) -> None:
        self.tmp = tmp
        self.registry = Registry()
        self.cfg = config(server, client, **gateway)
        self.app = build_app(self.cfg, self.registry)
        self.sdk = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gateway", timeout=30)
        self.engines = httpx.AsyncClient(transport=httpx.ASGITransport(app=engines_face(self.registry)), base_url="http://gateway")
        self.engine = RecordingEngine()
        self.engine_socket = os.path.join(tmp, "fake.sock")

    async def register(self, name: str = "fake", socket: str = "", pid: int = 4242) -> httpx.Response:
        return await self.engines.post("/engines/register", json={
            "name": name, "backend": "fake", "gpus": [], "socket": socket or self.engine_socket, "pid": pid})

    async def post(self, path: str, body: bytes = b"", key: str = API_KEY, **headers: str) -> httpx.Response:
        h = {"X-API-Key": key} if key else {}
        h.update(headers)
        return await self.sdk.post(path, content=body, headers=h)

    async def close(self) -> None:
        await self.sdk.aclose()
        await self.engines.aclose()
        await self.app._forwarder.aclose()


def run(coro) -> None:
    asyncio.run(coro)
