"""An SDK that disconnects mid-body ends the exchange: the engine never receives a shorter body as if complete."""
import asyncio

import orjson
import pytest

from training.gateway.app import ClientDisconnected, _read_all, asgi_sdk_face, build_app
from training.gateway.registry import Registry

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


def test_the_sdk_face_sends_nothing_and_the_engine_gets_a_cut_off_body(tmp_path):
    async def main():
        registry = Registry()
        app = build_app(config("asgi", "h11"), registry)
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
        registry.register("fake", "fake", None, sock, 1)
        sent = []

        async def send(message):
            sent.append(message)

        body = orjson.dumps({"session_id": "s"})
        scope = {"type": "http", "method": "POST", "path": "/api/v1/create_model", "query_string": b"",
                 "headers": [(b"host", b"sdk:8000"), (b"x-api-key", API_KEY.encode()), (b"content-type", b"application/json"),
                             (b"content-length", str(len(body) * 2).encode())], "server": ("sdk", 80)}
        receive = _scripted([{"type": "http.request", "body": body, "more_body": True}, {"type": "http.disconnect"}])
        try:
            await asgi_sdk_face(app._forwarder)(scope, receive, send)
            assert sent == []                                   # nobody is left to answer
            await asyncio.sleep(0.1)
            assert len(seen) == 1 and body in seen[0]
            assert not seen[0].endswith(b"0\r\n\r\n")         # no terminating chunk: the engine knows it was cut off
        finally:
            server.close()
            await server.wait_closed()
            await app._forwarder.aclose()
    run(main())
