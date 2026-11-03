"""Engine-side HTTP clients: one connection pool per engine socket.

Two implementations, chosen by configuration, so the forwarder's own cost can be
measured against each other: `HttpxClient` (httpx over its pool) and `H11Client` (h11
on raw asyncio unix-socket streams, nothing between the socket and the ASGI send).
Both read the engine's whole response before the status goes to the SDK: an engine that
dies mid-response is then a 503, never a 200 with a cut-off body. Measured at 256 in
flight against an engine serving 804 req/s directly:
h11 737 req/s, httpx 29. httpcore's `_async/connection_pool.py`
`_assign_requests_to_connections` scans every queued request against every pooled
connection on each pool event (encode/httpcore#1035); the h11 pool pops one idle
stream per request.

A failed exchange is reported, never retried: the gateway cannot know whether the
engine read a request before the connection died, and a replayed `optim_step` would
run twice. The SDK retries a 503 itself.
"""
import asyncio
import logging
import time
from collections import deque
from typing import AsyncGenerator, AsyncIterator, Deque, List, Protocol, Tuple, Union

import h11
import httpx

logger = logging.getLogger(__name__)

Headers = List[Tuple[bytes, bytes]]
Body = Union[bytes, AsyncIterator[bytes]]
BodyStream = AsyncGenerator[bytes, None]   # the caller iterates it to the end or calls aclose()
_READ = 65536


class EngineUnreachable(Exception):
    """The socket refused, vanished, or closed before a response arrived."""


class EngineResponse:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: Headers, body: BodyStream) -> None:
        self.status = status
        self.headers = headers
        self.body = body


class EngineClient(Protocol):
    async def request(self, method: str, target: str, headers: Headers, body: Body) -> EngineResponse: ...
    async def aclose(self) -> None: ...


# ------------------------------------------------------------------ httpx

class HttpxClient:
    """Responses are buffered: a connection held open while a response streams out keeps
    httpcore's pool scanning it for every queued request."""

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
        return EngineResponse(r.status_code, list(r.headers.raw), _once(r.content))

    async def aclose(self) -> None:
        await self._client.aclose()


async def _once(body: bytes) -> BodyStream:
    yield body


# -------------------------------------------------------------------- h11

class _Conn:
    __slots__ = ("reader", "writer", "idle_since")

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.idle_since = 0.0


class H11Client:
    """A deque of idle connections and a semaphore of `pool` slots; one h11 state machine per request."""

    def __init__(self, uds: str, pool: int, connect_timeout_s: float, keepalive_s: float) -> None:
        self._uds = uds
        self._connect_timeout = connect_timeout_s
        self._keepalive = keepalive_s
        self._idle: Deque[_Conn] = deque()
        self._slots = asyncio.Semaphore(pool)

    async def request(self, method: str, target: str, headers: Headers, body: Body) -> EngineResponse:
        await self._slots.acquire()   # released by _exchange on success, here on any failure
        try:
            conn = await self._acquire()
            try:
                return await self._exchange(conn, method, target, headers, body)
            except BaseException:
                conn.writer.close()   # whatever interrupted the exchange, the stream is not reusable
                raise
        except BaseException:
            self._slots.release()
            raise

    async def aclose(self) -> None:
        while self._idle:
            self._idle.popleft().writer.close()

    async def _acquire(self) -> _Conn:
        now = time.monotonic()
        while self._idle:
            conn = self._idle.popleft()
            if now - conn.idle_since < self._keepalive and not conn.writer.is_closing() and not conn.reader.at_eof():
                return conn
            conn.writer.close()
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(self._uds), self._connect_timeout)
        except (OSError, asyncio.TimeoutError) as exc:
            raise EngineUnreachable(f"connect {self._uds}: {exc}") from exc
        return _Conn(reader, writer)

    async def _exchange(self, conn: _Conn, method: str, target: str, headers: Headers, body: Body) -> EngineResponse:
        h = h11.Connection(our_role=h11.CLIENT)
        hdrs = list(headers)   # Host included: the forwarder refuses a request without one before it gets here
        if isinstance(body, bytes):
            hdrs.append((b"content-length", str(len(body)).encode()))
        else:
            hdrs.append((b"transfer-encoding", b"chunked"))
        w = conn.writer
        body_chunks: List[bytes] = []
        try:
            w.write(h.send(h11.Request(method=method, target=target, headers=hdrs)))
            if isinstance(body, bytes):
                if body:
                    w.write(h.send(h11.Data(data=body)))
            else:
                async for chunk in body:   # a body that fails mid-way propagates: the engine sees a cut-off chunked body
                    if chunk:
                        w.write(h.send(h11.Data(data=chunk)))
                        await w.drain()
            w.write(h.send(h11.EndOfMessage()))
            await w.drain()
            response = None
            while True:
                ev = h.next_event()
                if ev is h11.NEED_DATA:
                    data = await conn.reader.read(_READ)
                    if not data:
                        where = "before responding" if response is None else "mid-response"
                        raise EngineUnreachable(f"engine closed the connection {where}")
                    h.receive_data(data)
                elif isinstance(ev, h11.InformationalResponse):
                    continue
                elif isinstance(ev, h11.Response):
                    response = ev
                elif isinstance(ev, h11.Data):
                    body_chunks.append(bytes(ev.data))
                elif isinstance(ev, h11.EndOfMessage):
                    break
                else:
                    raise EngineUnreachable(f"unexpected {type(ev).__name__} from the engine")
        except (OSError, h11.ProtocolError) as exc:
            raise EngineUnreachable(f"{type(exc).__name__}: {exc}") from exc
        if response is None:
            raise EngineUnreachable("engine ended the message without a response")
        if h.our_state is h11.DONE and h.their_state is h11.DONE:
            conn.idle_since = time.monotonic()
            self._idle.append(conn)
        else:
            conn.writer.close()
        self._slots.release()
        return EngineResponse(response.status_code, [(k, v) for k, v in response.headers], _once(b"".join(body_chunks)))

def make_client(kind: str, uds: str, pool: int, connect_timeout_s: float, keepalive_s: float) -> EngineClient:
    if kind == "h11":
        return H11Client(uds, pool, connect_timeout_s, keepalive_s)
    if kind == "httpx":
        return HttpxClient(uds, pool, connect_timeout_s, keepalive_s)
    raise ValueError(f"unknown engine client {kind!r}")
