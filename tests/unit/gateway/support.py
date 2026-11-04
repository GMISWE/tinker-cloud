"""Fixtures for the gateway tests: a recording engine on a unix socket and a gateway driven in-process."""
import asyncio
import contextlib
import os
from typing import Any, Dict, List, Optional

import httpx
import orjson
import uvicorn

from training.config import TrainingConfig
from training.gateway.app import build_app
from training.gateway.registry import Registry
from training.gateway.sessions import SessionStore
from training.gateway.sockets import bind_unix_socket

API_KEY = "gw-test-key"
VARIANTS = ["aiohttp", "httpx"]   # engine-side clients: the one the gateway runs on, and the baseline
HEX = "0123456789abcdef"
MODEL = f"model_fake_{HEX}"


def config(client: str = "aiohttp", **gateway: Any) -> TrainingConfig:
    cfg = TrainingConfig()
    cfg.auth.api_key = API_KEY
    cfg.gateway.client = client
    cfg.gateway.connect_timeout_s = 1.0
    cfg.session_timeout_s = -1
    for k, v in gateway.items():
        setattr(cfg.gateway, k, v)
    return cfg


class RecordingEngine:
    """An ASGI app standing in for an engine; each request is recorded with its headers folded to a dict and as
    the raw list (`raw_headers`, a duplicate name shows there). `/api/v1/echo` returns the body; `/api/v1/status/<n>` answers n
    with a JSON error body; `/api/v1/stream/<mb>` streams that many MB; `/health` reports `model_ids`;
    `resolve_sampler_target` answers `target`; `create_model` echoes the id header; `delete_model` of an
    id in `gone` is 404; anything else answers 200 JSON. `on_request` runs before each answer."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.model_ids: List[str] = []
        self.target: Dict[str, Any] = {"model_id": MODEL, "pinned_version": 3, "base_model": "fake/tiny"}
        self.gone: set = set()
        self.answers: Dict[str, int] = {}   # path -> status to answer with an error body, whatever the route
        self.on_request: Optional[Any] = None

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        body = b""
        while True:
            m = await receive()
            body += m.get("body", b"")
            if not m.get("more_body", False):
                break
        raw = [(k.decode().lower(), v.decode("latin-1")) for k, v in scope["headers"]]
        headers = dict(raw)
        path = scope["path"]
        self.requests.append({"method": scope["method"], "path": path, "query": scope["query_string"].decode(),
                              "headers": headers, "raw_headers": raw, "body": body})
        if self.on_request is not None:
            self.on_request(self.requests[-1])
        if path.startswith("/api/v1/status/") or path in self.answers:
            status = self.answers[path] if path in self.answers else int(path.rsplit("/", 1)[1])
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
        elif path in ("/health", "/api/v1/health"):
            await _json(send, 200, {"status": "healthy", "version": "t", "ray_initialized": False,
                                    "active_training_clients": len(self.model_ids), "model_ids": list(self.model_ids), "futures_count": 0})
        elif path == "/api/v1/resolve_sampler_target":
            await _json(send, 200, self.target)
        elif path == "/api/v1/create_model":
            # the stand-in echoes the gateway's header back as the model id; a create sent without one echoes null
            await _json(send, 200, {"request_id": f"req_fake_{HEX}", "model_id": headers.get("x-model-id")})
        elif path == "/api/v1/delete_model":
            try:
                model_id = orjson.loads(body)["model_id"]
            except (orjson.JSONDecodeError, KeyError, TypeError):   # a proto or empty body in a route sweep
                model_id = None
            if model_id in self.gone:
                await _json(send, 404, {"error": f"'{model_id}'", "status_code": 404, "path": path})
            else:
                await _json(send, 200, {"model_id": model_id, "status": "deleted"})
        else:
            await _json(send, 200, {"ok": True, "path": path})


async def _respond(send, status: int, headers, body: bytes) -> None:
    headers = list(headers) + [(b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _json(send, status: int, doc: dict) -> None:
    await _respond(send, status, [(b"content-type", b"application/json")], orjson.dumps(doc))


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
    """A gateway (one client variant) over an in-memory registry and store, an engine on a socket, and HTTP clients for both faces."""

    def __init__(self, tmp: str, client: str = "aiohttp", store: Optional[SessionStore] = None, **gateway: Any) -> None:
        self.tmp = tmp
        self.registry = Registry()
        self.store = store if store is not None else SessionStore(None)
        self.cfg = config(client, **gateway)
        self.app = build_app(self.cfg, self.registry, self.store)
        self.gateway = self.app.state.gateway
        self.sdk = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gateway", timeout=30)
        self.engines = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app.state.engines), base_url="http://gateway")
        self.engine = RecordingEngine()
        self.engine_socket = os.path.join(tmp, "fake.sock")

    async def register(self, name: str = "fake", socket: str = "", pid: int = 4242) -> httpx.Response:
        return await self.engines.post("/engines/register", json={
            "name": name, "backend": "fake", "gpus": [], "socket": socket or self.engine_socket, "pid": pid})

    async def post(self, path: str, body: bytes = b"", key: str = API_KEY, **headers: str) -> httpx.Response:
        h = {"X-API-Key": key} if key else {}
        h.update(headers)
        return await self.sdk.post(path, content=body, headers=h)

    async def post_json(self, path: str, doc: dict, key: str = API_KEY) -> httpx.Response:
        return await self.post(path, orjson.dumps(doc), key=key, **{"content-type": "application/json"})

    async def get(self, path: str, key: str = API_KEY) -> httpx.Response:
        return await self.sdk.get(path, headers={"X-API-Key": key} if key else {})

    async def create_session(self) -> str:
        r = await self.post_json("/api/v1/create_session", {"tags": [], "user_metadata": {}, "sdk_version": "t"})
        assert r.status_code == 200, r.text
        return r.json()["session_id"]

    async def close(self) -> None:
        await self.sdk.aclose()
        await self.engines.aclose()
        await self.gateway.stop()


def run(coro) -> None:
    asyncio.run(coro)
