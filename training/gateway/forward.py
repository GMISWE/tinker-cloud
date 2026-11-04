"""Engine-side HTTP clients: one connection pool per engine socket, and the forward step itself.

`AiohttpClient` is the one the gateway runs on: aiohttp's connector keeps a deque of idle
connections per host and pops one per request, and its parser is C. `HttpxClient` stays as
the measurement baseline: httpcore's pool scans every queued request against every pooled
connection on each pool event (`_async/connection_pool.py` `_assign_requests_to_connections`,
encode/httpcore#1035), 29 req/s against the engine's 804 at 256 in flight.

Both read the engine's whole response before the status goes to the SDK: an engine that
dies mid-response is then a 503, never a 200 with a cut-off body.

A failed exchange is reported, never retried: the gateway cannot know whether the
engine read a request before the connection died, and a replayed `optim_step` would
run twice. The SDK retries a 503 itself.
"""
import asyncio
import logging
from typing import AsyncIterator, Dict, List, Protocol, Tuple, Union

import aiohttp
import httpx
import yarl

from ..config import GatewayConfig
from ..utils.helpers import MODEL_ID_HEADER, PINNED_VERSION_HEADER
from .registry import EngineRow, Registry

logger = logging.getLogger(__name__)

Headers = List[Tuple[bytes, bytes]]
Body = Union[bytes, AsyncIterator[bytes]]

# each hop frames its own messages; date and server are uvicorn's own on the way back. Host stays:
# the engine builds the download URLs it hands out from it, and the SDK must be able to reach them.
# hop-by-hop framing, and the two headers only the gateway sets: an SDK's copy is dropped, never forwarded
REQUEST_DROP = frozenset({b"connection", b"keep-alive", b"content-length", b"transfer-encoding",
                          MODEL_ID_HEADER.lower().encode(), PINNED_VERSION_HEADER.lower().encode()})
RESPONSE_DROP = frozenset({b"connection", b"keep-alive", b"transfer-encoding", b"date", b"server"})
# aiohttp would add these on its own; the engine sees only what the SDK sent
AUTO_HEADERS = ("User-Agent", "Accept", "Accept-Encoding", "Content-Type")


class EngineUnreachable(Exception):
    """The socket refused, vanished, or closed before a whole response arrived."""


class EngineResponse:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: Headers, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body


class EngineClient(Protocol):
    async def request(self, method: str, target: str, headers: Headers, body: Body) -> EngineResponse: ...
    async def aclose(self) -> None: ...


class AiohttpClient:
    def __init__(self, uds: str, pool: int, connect_timeout_s: float, keepalive_s: float) -> None:
        self._session = aiohttp.ClientSession(
            connector=aiohttp.UnixConnector(path=uds, limit=pool, keepalive_timeout=keepalive_s),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout_s),   # a retrieve_future holds for minutes
            auto_decompress=False, skip_auto_headers=AUTO_HEADERS,
        )

    async def request(self, method: str, target: str, headers: Headers, body: Body) -> EngineResponse:
        hdrs = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in headers]
        data = body if isinstance(body, bytes) else _chunks(body)
        try:
            url = yarl.URL("http://engine" + target, encoded=True)   # the SDK's percent-encoding stays as sent
            async with self._session.request(method, url, headers=hdrs, data=data) as r:
                content = await r.read()
                return EngineResponse(r.status, list(r.raw_headers), content)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise EngineUnreachable(f"{type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        await self._session.close()


async def _chunks(body: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async for chunk in body:   # a body that fails mid-way propagates: the engine sees a cut-off chunked body
        if chunk:
            yield chunk


class HttpxClient:
    def __init__(self, uds: str, pool: int, connect_timeout_s: float, keepalive_s: float) -> None:
        limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool, keepalive_expiry=keepalive_s)
        self._client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=uds, limits=limits),
            base_url="http://engine", timeout=httpx.Timeout(None, connect=connect_timeout_s),
        )

    async def request(self, method: str, target: str, headers: Headers, body: Body) -> EngineResponse:
        req = self._client.build_request(method, target, headers=headers, content=body)
        try:
            r = await self._client.send(req)
        except httpx.TransportError as exc:
            raise EngineUnreachable(f"{type(exc).__name__}: {exc}") from exc
        return EngineResponse(r.status_code, list(r.headers.raw), r.content)

    async def aclose(self) -> None:
        await self._client.aclose()


def make_client(kind: str, uds: str, pool: int, connect_timeout_s: float, keepalive_s: float) -> EngineClient:
    if kind == "aiohttp":
        return AiohttpClient(uds, pool, connect_timeout_s, keepalive_s)
    if kind == "httpx":
        return HttpxClient(uds, pool, connect_timeout_s, keepalive_s)
    raise ValueError(f"unknown engine client {kind!r}")


class Pool:
    """One client per engine, opened on first use and closed when the engine's row is dropped."""

    def __init__(self, gateway: GatewayConfig, registry: Registry) -> None:
        self._gw = gateway
        self._clients: Dict[str, EngineClient] = {}
        registry.on_drop(self._drop)

    def client_for(self, row: EngineRow) -> EngineClient:
        client = self._clients.get(row.name)
        if client is None:
            client = make_client(self._gw.client, row.socket, self._gw.pool_size, self._gw.connect_timeout_s, self._gw.keepalive_s)
            self._clients[row.name] = client
        return client

    def _drop(self, row: EngineRow) -> None:
        client = self._clients.pop(row.name, None)
        if client is not None:
            asyncio.get_running_loop().create_task(client.aclose())

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


async def forward(client: EngineClient, method: str, path: str, query: bytes, headers: Headers,
                  add_headers: Headers, body: Body) -> EngineResponse:
    """Move the request to the engine with its body untouched and the hop-by-hop headers replaced."""
    target = f"{path}?{query.decode()}" if query else path
    fwd = [(k, v) for k, v in headers if k.lower() not in REQUEST_DROP] + list(add_headers)
    resp = await client.request(method, target, fwd, body)
    resp.headers = [(k, v) for k, v in resp.headers if k.lower() not in RESPONSE_DROP]
    return resp
