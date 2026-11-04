"""The aiohttp client adds nothing of its own: the engine sees the SDK's headers and bytes, and the response bytes come back as written."""
import pytest

from .support import API_KEY, Harness, run, serve_uds


def test_the_aiohttp_client_adds_no_headers_of_its_own(tmp_path):
    """The engine sees what the SDK sent plus the hop's own framing: nothing of the client's
    (its User-Agent, Accept, Accept-Encoding), and a response body as the engine wrote it."""
    async def main():
        h = Harness(str(tmp_path), "aiohttp")
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sent = {"host": "sdk:8000", "x-api-key": API_KEY, "content-type": "application/json",
                        "accept": "*/*", "accept-encoding": "identity", "user-agent": "sdk-test/1"}
                r = await h.sdk.post("/api/v1/echo", content=b'{"a": 1}', headers=sent)
                assert r.status_code == 200 and r.content == b'{"a": 1}'
                seen = {k.lower(): v for k, v in h.engine.requests[-1]["headers"].items()}
                extra = set(seen) - set(sent) - {"content-length", "transfer-encoding", "connection"}   # the hop's own framing
                assert extra == set(), {k: seen[k] for k in extra}
                assert all(seen[k] == v for k, v in sent.items()), seen
                r = await h.sdk.post("/api/v1/echo", content=b"\x1f\x8bnot-really-gzip",
                                     headers={"X-API-Key": API_KEY, "content-type": "application/octet-stream", "content-encoding": "gzip"})
                assert r.status_code == 200 and r.content == b"\x1f\x8bnot-really-gzip"   # bytes through, not inflated
        finally:
            await h.close()
    run(main())


@pytest.mark.parametrize("client", ["aiohttp", "httpx"])
def test_a_percent_encoded_query_reaches_the_engine_as_sent(tmp_path, client):
    """`%41`, `%2F` and `%7E` are not decoded or re-quoted on the hop: the engine sees the SDK's bytes."""
    async def main():
        h = Harness(str(tmp_path), client)
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sent = []

                async def receive():
                    return {"type": "http.request", "body": b"", "more_body": False}

                async def send(message):
                    sent.append(message)

                scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
                         "path": "/api/v1/echo", "raw_path": b"/api/v1/echo", "query_string": b"a=%41&c=%2F&d=%7E", "root_path": "",
                         "headers": [(b"host", b"sdk:8000"), (b"x-api-key", API_KEY.encode())],
                         "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 1)}
                await h.app(scope, receive, send)
                assert sent[0]["status"] == 200, sent
                assert h.engine.requests[-1]["query"] == "a=%41&c=%2F&d=%7E"
        finally:
            await h.close()
    run(main())


@pytest.mark.parametrize("client", ["aiohttp", "httpx"])
def test_a_percent_encoded_path_reaches_the_engine_as_sent(tmp_path, client):
    """`a%3Fb` stays one path segment (not a query) and `a%20b` is a legal request line: the engine
    decodes the same bytes the SDK encoded."""
    async def main():
        h = Harness(str(tmp_path), client)
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                for raw, seen_path in ((b"/api/v1/echo/a%3Fb", "/api/v1/echo/a?b"), (b"/api/v1/echo/a%20b", "/api/v1/echo/a b")):
                    sent = []

                    async def receive():
                        return {"type": "http.request", "body": b"", "more_body": False}

                    async def send(message):
                        sent.append(message)

                    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
                             "path": seen_path, "raw_path": raw, "query_string": b"", "root_path": "",
                             "headers": [(b"host", b"sdk:8000"), (b"x-api-key", API_KEY.encode())],
                             "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 1)}
                    await h.app(scope, receive, send)
                    assert sent[0]["status"] == 200, (raw, sent)
                    assert (h.engine.requests[-1]["path"], h.engine.requests[-1]["query"]) == (seen_path, "")
        finally:
            await h.close()
    run(main())
