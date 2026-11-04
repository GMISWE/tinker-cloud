"""The two listeners are told apart by the scope's server address, in both shapes uvicorn has used; a request without Host is refused."""
import orjson

from training.gateway.app import build_app, is_unix_listener
from training.gateway.registry import Registry

from .support import API_KEY, config, run


def test_unix_listener_detection_covers_both_scope_shapes():
    assert is_unix_listener({"server": None})
    assert is_unix_listener({"server": ("/tmp/gateway.sock", None)})
    assert not is_unix_listener({"server": ("127.0.0.1", 8000)})
    assert not is_unix_listener({"server": ("gateway", None)})   # a TCP client that named no port (httpx's ASGI transport)


def _scripted(messages):
    it = iter(messages)

    async def receive():
        return next(it)
    return receive


def _scope(server, path="/engines", method="GET", headers=()):
    return {"type": "http", "method": method, "path": path, "query_string": b"", "raw_path": path.encode(), "headers": list(headers), "server": server}


def test_both_unix_scope_shapes_reach_the_engines_face():
    async def main():
        app = build_app(config("aiohttp"), Registry())
        for server in (None, ("/tmp/gateway.sock", None)):
            sent = []

            async def send(m):
                sent.append(m)
            await app(_scope(server), _scripted([{"type": "http.request", "body": b""}]), send)
            assert sent[0]["status"] == 200 and orjson.loads(sent[1]["body"]) == {"engines": []}, (server, sent)
    run(main())


def test_a_request_without_host_is_400_at_the_gateway():
    async def main():
        app = build_app(config("aiohttp"), Registry())
        sent = []

        async def send(m):
            sent.append(m)
        scope = _scope(("127.0.0.1", 8000), "/api/v1/create_model", "POST", [(b"x-api-key", API_KEY.encode())])
        await app(scope, _scripted([{"type": "http.request", "body": b"{}"}]), send)
        assert sent[0]["status"] == 400 and b"Host" in sent[1]["body"], sent
    run(main())
