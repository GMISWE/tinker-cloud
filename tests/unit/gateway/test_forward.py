"""Pass-through fidelity, prefix routing and an engine that is not there, for both engine clients (aiohttp, and the httpx baseline)."""
import asyncio
import os
import time

import orjson
import pytest

from training.proto import tinker_public_pb2 as pb

from .support import API_KEY, VARIANTS, Harness, run, serve_uds

HEX = "0123456789abcdef"
MODEL = f"model_fake_{HEX}"
OTHER = f"model_other_{HEX}"


@pytest.fixture(params=VARIANTS, ids=VARIANTS)
def harness(tmp_path, request):
    return Harness(str(tmp_path), request.param)


def test_status_body_and_content_type_pass_through_unchanged(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                for status in (200, 400, 404, 425):
                    r = await h.post(f"/api/v1/status/{status}", b"{}")
                    assert r.status_code == status
                    assert r.headers["content-type"] == "application/json"
                    assert r.json() == {"error": f"engine says {status}", "status_code": status, "path": f"/api/v1/status/{status}"}
                    assert "server" not in {k.lower() for k in r.headers} or r.headers.get("server") != "engine"
        finally:
            await h.close()
    run(main())


def test_request_bytes_headers_and_query_reach_the_engine_intact(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                proto = pb.ForwardBackwardRequest(model_id=MODEL, loss_fn="cross_entropy").SerializeToString()
                big = os.urandom(3 * 1024 * 1024)
                for body, ct in ((orjson.dumps({"a": 1}), "application/json"), (proto, "application/x-protobuf"),
                                 (big, "application/octet-stream")):
                    r = await h.sdk.post("/api/v1/echo?x=1&y=two", content=body,
                                         headers={"X-API-Key": API_KEY, "content-type": ct, "x-custom": "kept", "host": "sdk.example:8000"})
                    assert r.status_code == 200 and r.content == body and r.headers["content-type"] == ct
                    assert r.headers["x-engine"] == "yes"
                    seen = h.engine.requests[-1]
                    assert seen["body"] == body and seen["query"] == "x=1&y=two"
                    assert seen["headers"]["x-api-key"] == API_KEY and seen["headers"]["x-custom"] == "kept"
                    assert seen["headers"]["host"] == "sdk.example:8000"   # the engine builds download URLs from it
                    assert seen["headers"]["content-type"] == ct
        finally:
            await h.close()
    run(main())


def test_a_multi_megabyte_response_passes_through(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                r = await h.sdk.get("/api/v1/external_weights/x/y/z", headers={"X-API-Key": API_KEY})
                assert r.status_code == 200   # a public path: the engine's answer, whatever it is
                r = await h.sdk.post("/api/v1/stream/4", headers={"X-API-Key": API_KEY})
                assert r.status_code == 200 and len(r.content) == 4 * 1024 * 1024
                assert r.content[:256] == bytes(range(256))
        finally:
            await h.close()
    run(main())


def test_api_key_is_checked_at_the_gateway(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                r = await h.post("/api/v1/create_model", b"{}", key="")
                assert r.status_code == 401 and r.json()["path"] == "/api/v1/create_model" and "Missing API key" in r.json()["error"]
                r = await h.post("/api/v1/create_model", b"{}", key="wrong")
                assert r.status_code == 401 and r.json()["error"] == "Invalid API key"
                assert h.engine.requests == []
                for public in ("/health", "/api/v1/health", "/api/v1/healthz"):
                    assert (await h.sdk.get(public)).status_code == 200
                assert (await h.sdk.post("/api/v1/client/config", content=b"{}")).status_code == 200
                assert len(h.engine.requests) == 4
        finally:
            await h.close()
    run(main())


def test_an_id_naming_another_engine_is_404_at_the_gateway(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                proto_other = pb.ForwardBackwardRequest(model_id=OTHER).SerializeToString()
                cases = [
                    ("POST", f"/api/v1/retrieve_future/req_other_{HEX}", b"", "application/json", f"req_other_{HEX}"),
                    ("POST", "/api/v1/cancel_future", orjson.dumps({"request_id": f"req_other_{HEX}"}), "application/json", f"req_other_{HEX}"),
                    ("POST", "/api/v1/optim_step", orjson.dumps({"model_id": OTHER}), "application/json", OTHER),
                    ("POST", "/api/v1/forward_backward", proto_other, "application/x-protobuf", OTHER),
                    ("GET", f"/api/v1/training_runs/{OTHER}/checkpoints", b"", "application/json", OTHER),
                    ("GET", f"/api/v1/get_tokenizer?model_id={OTHER}", b"", "application/json", OTHER),
                    ("POST", "/api/v1/create_sampling_client", orjson.dumps({"model_path": f"tinker://{OTHER}/sampler_weights/s"}), "application/json", OTHER),
                    ("POST", "/api/v1/asample", orjson.dumps({"model_path": f"tinker://{OTHER}/weights/s", "prompt": {"tokens": [1]}}), "application/json", OTHER),
                    ("POST", "/api/v1/weights_info", orjson.dumps({"tinker_path": f"tinker://{OTHER}/weights/s"}), "application/json", OTHER),
                    ("POST", "/api/v1/load_weights", orjson.dumps({"model_id": OTHER, "path": f"tinker://{OTHER}/weights/s"}), "application/json", OTHER),
                ]
                for method, path, body, ct, id_ in cases:
                    r = await h.sdk.request(method, path, content=body, headers={"X-API-Key": API_KEY, "content-type": ct})
                    assert r.status_code == 404, (path, r.status_code, r.text)
                    err = r.json()
                    assert "'other'" in err["error"] and id_ in err["error"], err   # the body names the engine and the id
                    assert err["path"] == path.split("?")[0]
                assert h.engine.requests == []
                # the same ops naming the registered engine, or an id in another shape, reach it
                for method, path, body, ct in [
                    ("POST", f"/api/v1/retrieve_future/req_fake_{HEX}", b"", "application/json"),
                    ("POST", "/api/v1/optim_step", orjson.dumps({"model_id": MODEL}), "application/json"),
                    ("POST", "/api/v1/forward_backward", pb.ForwardBackwardRequest(model_id=MODEL).SerializeToString(), "application/x-protobuf"),
                    ("POST", "/api/v1/optim_step", orjson.dumps({"model_id": "model_nope"}), "application/json"),
                ]:
                    r = await h.sdk.request(method, path, content=body, headers={"X-API-Key": API_KEY, "content-type": ct})
                    assert r.status_code == 200, (path, r.text)
                assert len(h.engine.requests) == 4
        finally:
            await h.close()
    run(main())


def test_an_engine_whose_socket_is_gone_is_503_within_two_seconds(harness):
    async def main():
        h = harness
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                assert (await h.post("/api/v1/echo", b"x")).status_code == 200
            # the server is stopped and its socket file removed; the registration is still there
            t0 = time.monotonic()
            r = await h.post("/api/v1/echo", b"x")
            assert r.status_code == 503 and time.monotonic() - t0 < 2.0, r.text
            assert r.json()["error"].startswith("engine fake unreachable"), r.text
            # a socket file nothing accepts on
            if os.path.exists(h.engine_socket):
                os.unlink(h.engine_socket)
            open(h.engine_socket, "w").close()
            t0 = time.monotonic()
            r = await h.post("/api/v1/echo", b"x")
            assert r.status_code == 503 and time.monotonic() - t0 < 2.0, r.text
        finally:
            await h.close()
    run(main())


def test_an_engine_closing_before_it_answers_is_503_and_the_request_is_never_replayed(harness):
    async def main():
        h = harness
        accepted = []

        async def slam(reader, writer):
            accepted.append(1)
            await reader.read(1024)   # the engine read the request: a replay could run it twice
            writer.close()

        server = await asyncio.start_unix_server(slam, path=h.engine_socket)
        try:
            await h.register()
            for path, body in (("/api/v1/optim_step", orjson.dumps({"model_id": MODEL})),      # buffered body
                               ("/api/v1/create_model", orjson.dumps({"session_id": "s"}))):   # streamed body
                accepted.clear()
                t0 = time.monotonic()
                r = await h.post(path, body)
                assert r.status_code == 503 and time.monotonic() - t0 < 2.0, r.text
                assert r.json()["error"].startswith("engine fake unreachable"), r.text
                assert len(accepted) == 1, accepted
        finally:
            server.close()
            await server.wait_closed()
            await h.close()
    run(main())


def test_an_engine_dying_mid_response_is_503_not_a_truncated_200(harness):
    async def main():
        h = harness

        async def half_answer(reader, writer):
            await reader.read(65536)
            writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 1000\r\n\r\n{\"ok\": ")
            await writer.drain()
            writer.close()   # dies after the status line and 9 body bytes

        server = await asyncio.start_unix_server(half_answer, path=h.engine_socket)
        try:
            await h.register()
            t0 = time.monotonic()
            r = await h.post("/api/v1/optim_step", orjson.dumps({"model_id": MODEL}))
            assert r.status_code == 503 and time.monotonic() - t0 < 2.0, (r.status_code, r.text)
            assert r.json()["error"].startswith("engine fake unreachable"), r.text
        finally:
            server.close()
            await server.wait_closed()
            await h.close()
    run(main())


def test_the_aiohttp_client_reuses_an_idle_connection(tmp_path):
    async def main():
        h = Harness(str(tmp_path), "aiohttp")
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                for _ in range(3):
                    assert (await h.post("/api/v1/echo", b"x")).status_code == 200
                connector = h.gateway.pool._clients["fake"]._session.connector
                (key, idle), = connector._conns.items()
                assert len(idle) == 1
                proto, since = idle[0]
                idle[0] = (proto, since - h.cfg.gateway.keepalive_s - 1)   # idle past the keep-alive: dropped, not reused
                assert (await h.post("/api/v1/echo", b"x")).status_code == 200
                (key, idle), = connector._conns.items()
                assert len(idle) == 1 and idle[0][0] is not proto
        finally:
            await h.close()
    run(main())
