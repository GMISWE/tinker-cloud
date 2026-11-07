"""An SDK that disconnects mid-body ends the exchange: the engine never receives a shorter body as if complete."""
import asyncio

import orjson
import pytest

from training.gateway.app import ClientDisconnected, _read_all, build_app
from training.gateway.registry import Registry
from training.gateway.sessions import SessionStore

from .support import API_KEY, config, run


def _scripted(messages):
    it = iter(messages)

    async def receive():
        return next(it)
    return receive


def test_read_all_raises_on_a_disconnect_mid_body():
    async def main():
        receive = _scripted([{"type": "http.request", "body": b"\n\x1cmodel_fake", "more_body": True},
                             {"type": "http.disconnect"}])
        with pytest.raises(ClientDisconnected):
            await _read_all(receive)
        receive = _scripted([{"type": "http.request", "body": b"ab", "more_body": True},
                             {"type": "http.request", "body": b"c"}])
        assert await _read_all(receive) == b"abc"
    run(main())


def test_the_sdk_face_answers_nobody_and_the_engine_gets_a_cut_off_body(tmp_path):
    async def main():
        registry = Registry()
        app = build_app(config(), registry, SessionStore(None))
        sock = str(tmp_path / "e.sock")
        seen = []

        async def engine(reader, writer):
            data = b""
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                data += chunk
            seen.append(data)
            writer.close()

        server = await asyncio.start_unix_server(engine, path=sock)
        registry.register("fake", "fake", None, sock, 1, False)
        sent = []

        async def send(message):
            sent.append(message)

        # a route the gateway streams (no id in its body): the engine is reading while the SDK leaves
        body = orjson.dumps({"request_ids": ["req_fake_0123456789abcdef"]})
        scope = {"type": "http", "method": "POST", "path": "/api/v1/cleanup_futures", "raw_path": b"/api/v1/cleanup_futures", "query_string": b"", "scheme": "http",
                 "http_version": "1.1", "client": ("sdk", 1), "server": ("sdk", 80),
                 "headers": [(b"host", b"sdk:8000"), (b"x-api-key", API_KEY.encode()), (b"content-type", b"application/json"),
                             (b"content-length", str(len(body) * 2).encode())]}
        receive = _scripted([{"type": "http.request", "body": body, "more_body": True}, {"type": "http.disconnect"}])
        try:
            await app(scope, receive, send)
            assert sent and sent[0]["status"] == 499, sent              # nobody is left to answer; no 2xx or 503 is invented
            await asyncio.sleep(0.1)
            assert len(seen) == 1 and body in seen[0], seen
            assert not seen[0].endswith(b"0\r\n\r\n")         # no terminating chunk: the engine knows it was cut off
        finally:
            server.close()
            await server.wait_closed()
            await app.state.gateway.stop()
    run(main())
