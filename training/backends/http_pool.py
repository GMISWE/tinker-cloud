"""
One persistent HTTP client per inference-engine URL (D13).

A backend that samples over HTTP owns one pool, resolves the model's endpoint
from core.routing per request, takes the client for that URL here, and closes
the pool in close(). Clients are keyed by URL, created on first use, and hold
their connections across requests (keep-alive). The engine codecs
(backends/<backend>/*_client.py) build requests on top of these clients; the
pool knows no engine.
"""
from typing import Dict, Optional

import httpx


class HttpClientPool:
    def __init__(
        self,
        *,
        max_connections: Optional[int] = None,
        max_keepalive_connections: int = 64,
        keepalive_expiry: float = 5.0,
        timeout: Optional[httpx.Timeout] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """
        Args:
            max_connections: concurrent connections per URL; None = unbounded
                (back-pressure then comes from the router / engine queue)
            max_keepalive_connections: idle connections kept open per URL
            keepalive_expiry: seconds an idle connection is reused; keep it
                below the server's keep-alive timeout or a reused connection
                the server already closed fails the next request
            timeout: httpx timeout for every request on these clients; None =
                httpx default (5 s each phase). Codecs may override per call.
            transport: test seam (httpx.MockTransport); None = real network
        """
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=keepalive_expiry,
        )
        self._timeout = timeout
        self._transport = transport
        self._clients: Dict[str, httpx.AsyncClient] = {}

    def for_url(self, base_url: str) -> httpx.AsyncClient:
        key = base_url.rstrip("/")
        client = self._clients.get(key)
        if client is None:
            kwargs = {"limits": self._limits, "transport": self._transport}
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            client = self._clients[key] = httpx.AsyncClient(**kwargs)
        return client

    def __len__(self) -> int:
        return len(self._clients)

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for c in clients:
            await c.aclose()
