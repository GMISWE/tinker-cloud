"""The gateway's ASGI app: the SDK's face on the TCP listener, the engines' face on the unix socket.

The SDK face checks the API key, finds the engine a request names and forwards the
request with its body untouched; the engines face takes register, heartbeat and
unregister. Both are plain ASGI; a FastAPI variant of the SDK face exists so the
forwarder's own cost can be measured against it.
"""
import asyncio
import logging
import re
import time
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional, Union

import orjson

from ..config import TrainingConfig
from . import routing
from .forward import Body, BodyStream, EngineClient, EngineUnreachable, Headers, make_client
from .registry import EngineRow, Registry, RegistryConflict

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({
    "/health", "/api/v1/health", "/api/v1/healthz",
    "/api/v1/client/config", "/api/v1/client/dynamic_config", "/api/v1/telemetry",
})
PUBLIC_PREFIXES = ("/api/v1/external_weights/",)   # signed URLs carry their own token
# each hop frames its own messages; date and server are uvicorn's own on the way back. Host stays:
# the engine builds the download URLs it hands out from it, and the SDK must be able to reach them.
REQUEST_DROP = frozenset({b"connection", b"keep-alive", b"content-length", b"transfer-encoding"})
RESPONSE_DROP = frozenset({b"connection", b"keep-alive", b"transfer-encoding", b"date", b"server"})
_ENGINE_PATH = re.compile(r"^/engines/([a-z0-9_]+)(/heartbeat)?$")

ReadBody = Callable[[], Awaitable[bytes]]
StreamBody = Callable[[], AsyncIterator[bytes]]


class ClientDisconnected(Exception):
    """The SDK went away before its request body was complete; nothing is forwarded or answered."""


class Reply:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: Headers, body: Union[bytes, BodyStream]) -> None:
        self.status = status
        self.headers = headers
        self.body = body


def _json_reply(status: int, doc: dict) -> Reply:
    body = orjson.dumps(doc)
    return Reply(status, [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())], body)


def _error(status: int, message: str, path: str) -> Reply:
    """The same shape as the engine's own error handler, so the SDK sees one error format."""
    return _json_reply(status, {"error": message, "status_code": status, "path": path})


def _header(headers: Headers, name: bytes) -> bytes:
    for k, v in headers:
        if k.lower() == name:
            return v
    return b""


class Forwarder:
    """Everything between the SDK's request and the engine's response, independent of the server adapter."""

    def __init__(self, config: TrainingConfig, registry: Registry) -> None:
        self._keys = {k.strip() for k in config.auth.api_key.split(",") if k.strip()} if config.auth.enabled else None
        self._key_header = config.auth.api_key_header.lower().encode()
        self._registry = registry
        self._gw = config.gateway
        self._clients: Dict[str, EngineClient] = {}
        registry.on_drop(self._drop_client)

    def _authorized(self, path: str, headers: Headers) -> Optional[str]:
        """None when the request may pass; else the 401 message."""
        if self._keys is None or path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return None
        key = _header(headers, self._key_header).decode("latin-1")
        if not key:
            return f"Missing API key (provide via {self._key_header.decode()} header)"
        if key not in self._keys:
            return "Invalid API key"
        return None

    def _client_for(self, row: EngineRow) -> EngineClient:
        client = self._clients.get(row.name)
        if client is None:
            client = make_client(self._gw.client, row.socket, self._gw.pool_size, self._gw.connect_timeout_s, self._gw.keepalive_s)
            self._clients[row.name] = client
        return client

    def _drop_client(self, row: EngineRow) -> None:
        client = self._clients.pop(row.name, None)
        if client is not None:
            asyncio.get_running_loop().create_task(client.aclose())

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def forward(self, method: str, path: str, query: bytes, headers: Headers,
                      read_body: ReadBody, stream_body: StreamBody) -> Reply:
        denied = self._authorized(path, headers)
        if denied is not None:
            return _error(401, denied, path)
        if not _header(headers, b"host"):
            return _error(400, "a Host header is required: the engine builds the URLs it returns from it", path)
        row = self._registry.only()
        if row is None:
            return _error(503, "no engine registered", path)
        content_type = _header(headers, b"content-type")
        content_encoding = _header(headers, b"content-encoding")
        body: Body
        if routing.reads_body(path):
            body = await read_body()
            named = routing.named_engine(path, query, content_type, content_encoding, body)
        else:
            named = routing.named_engine(path, query, content_type, content_encoding, None)
            body = b"" if method in ("GET", "HEAD") else stream_body()
        if named is not None and named[1] != row.name:
            return _error(404, f"no engine named {named[1]!r} is registered for id {named[0]!r}", path)
        target = f"{path}?{query.decode()}" if query else path
        fwd = [(k, v) for k, v in headers if k.lower() not in REQUEST_DROP]
        try:
            resp = await self._client_for(row).request(method, target, fwd, body)
        except EngineUnreachable as exc:
            logger.warning("Engine %s unreachable for %s %s: %s", row.name, method, path, exc)
            return _error(503, f"engine {row.name} unreachable: {exc}", path)
        return Reply(resp.status, [(k, v) for k, v in resp.headers if k.lower() not in RESPONSE_DROP], resp.body)


# ------------------------------------------------------------ ASGI faces

async def _read_all(receive) -> bytes:
    chunks: List[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ClientDisconnected()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_reply(send, reply: Reply) -> None:
    await send({"type": "http.response.start", "status": reply.status, "headers": reply.headers})
    if isinstance(reply.body, bytes):
        await send({"type": "http.response.body", "body": reply.body})
        return
    try:
        async for chunk in reply.body:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})
    finally:
        await reply.body.aclose()


def asgi_sdk_face(forwarder: Forwarder):
    async def app(scope, receive, send) -> None:
        async def read_body() -> bytes:
            return await _read_all(receive)

        async def stream_body() -> AsyncIterator[bytes]:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise ClientDisconnected()   # the engine sees a cut-off chunked body, never a shorter complete one
                chunk = message.get("body", b"")
                if chunk:
                    yield chunk
                if not message.get("more_body", False):
                    return

        try:
            reply = await forwarder.forward(scope["method"], scope["path"], scope["query_string"], scope["headers"],
                                            read_body, stream_body)
        except ClientDisconnected:
            return
        await _send_reply(send, reply)
    return app


def fastapi_sdk_face(forwarder: Forwarder):
    """The same forwarder behind FastAPI's request and response objects: the measurement baseline."""
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse
    from starlette.requests import ClientDisconnect

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def forward(path: str, request: Request):
        try:
            reply = await forwarder.forward(request.method, request.url.path, request.url.query.encode(),
                                            request.headers.raw, request.body, request.stream)
        except ClientDisconnect:
            return Response(status_code=499)   # nobody is left to read it
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in reply.headers}
        if isinstance(reply.body, bytes):
            return Response(content=reply.body, status_code=reply.status, headers=headers)
        return StreamingResponse(reply.body, status_code=reply.status, headers=headers)

    return app


def engines_face(registry: Registry):
    async def app(scope, receive, send) -> None:
        method, path = scope["method"], scope["path"]
        try:
            body = await _read_all(receive)
        except ClientDisconnected:
            return
        if method == "GET" and path == "/engines":
            now = time.monotonic()
            reply = _json_reply(200, {"engines": [r.as_dict(now) for r in registry.rows()]})
        elif method == "POST" and path == "/engines/register":
            reply = _register(registry, body, path)
        else:
            m = _ENGINE_PATH.match(path)
            if m is None:
                reply = _error(404, f"no such route: {method} {path}", path)
            elif method == "POST" and m.group(2):
                reply = _json_reply(200, {}) if registry.heartbeat(m.group(1)) else _error(404, f"engine {m.group(1)!r} is not registered", path)
            elif method == "DELETE" and not m.group(2):
                reply = Reply(204, [], b"") if registry.unregister(m.group(1)) else _error(404, f"engine {m.group(1)!r} is not registered", path)
            else:
                reply = _error(404, f"no such route: {method} {path}", path)
        await _send_reply(send, reply)
    return app


def _register(registry: Registry, body: bytes, path: str) -> Reply:
    try:
        row = orjson.loads(body)
        gpus = row["gpus"]
        if gpus is not None and not (isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)):
            raise TypeError("gpus must be a list of ints or null")
        registry.register(name=row["name"], backend=row["backend"], gpus=gpus,
                          socket=row["socket"], pid=int(row["pid"]))
    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _error(400, f"bad registration: {type(exc).__name__}: {exc}", path)
    except RegistryConflict as exc:
        return _error(409, str(exc), path)
    return _json_reply(200, {"state": "UP"})


def is_unix_listener(scope) -> bool:
    """The ASGI scope's `server` for a unix-socket listener: None (ASGI spec, uvicorn ≤ 0.38) or
    `(path, None)` with the socket's absolute path (uvicorn ≥ 0.53, `protocols/utils.py get_local_addr`;
    the gateway binds absolute paths). A TCP listener carries a host name, with or without a port."""
    server = scope["server"]
    if server is None:
        return True
    host, port = server
    return port is None and isinstance(host, str) and host.startswith("/")


class GatewayApp:
    """One ASGI app for both listeners, told apart by the listener's address in the scope."""

    def __init__(self, forwarder: Forwarder, sdk_face, engines) -> None:
        self._forwarder = forwarder
        self._sdk = sdk_face
        self._engines = engines

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
        elif scope["type"] != "http":
            raise RuntimeError(f"unsupported ASGI scope type {scope['type']!r}")
        elif is_unix_listener(scope):
            await self._engines(scope, receive, send)
        else:
            await self._sdk(scope, receive, send)

    async def _lifespan(self, receive, send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self._forwarder.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return


def build_app(config: TrainingConfig, registry: Optional[Registry] = None) -> GatewayApp:
    registry = registry if registry is not None else Registry()
    forwarder = Forwarder(config, registry)
    face = asgi_sdk_face(forwarder) if config.gateway.server == "asgi" else fastapi_sdk_face(forwarder)
    return GatewayApp(forwarder, face, engines_face(registry))
