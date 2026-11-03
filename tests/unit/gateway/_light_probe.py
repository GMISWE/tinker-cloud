"""Run as a subprocess: import the gateway, serve one request of every route, then list what got imported.

Exit 0 and print LIGHT when none of the heavy packages is in sys.modules; otherwise print them and exit 1.
"""
import asyncio
import importlib
import os
import sys
import tempfile

HEAVY = ("ray", "torch", "numpy", "transformers", "vllm")
ROUTES = [
    ("GET", "/health"), ("GET", "/api/v1/health"), ("GET", "/api/v1/healthz"),
    ("POST", "/api/v1/client/config"), ("POST", "/api/v1/client/dynamic_config"),
    ("GET", "/api/v1/get_server_capabilities"), ("POST", "/api/v1/telemetry"),
    ("POST", "/api/v1/create_session"), ("POST", "/api/v1/session_heartbeat"),
    ("POST", "/api/v1/create_sampling_session"), ("POST", "/api/v1/join_sampling_session"),
    ("POST", "/api/v1/sessions/s/finish"), ("GET", "/api/v1/sessions"), ("GET", "/api/v1/sessions/s"),
    ("GET", "/api/v1/samplers/x"), ("POST", "/api/v1/create_model"), ("POST", "/api/v1/delete_model"),
    ("POST", "/api/v1/unload_model"), ("POST", "/api/v1/get_info"), ("GET", "/api/v1/get_tokenizer?model_id=m"),
    ("GET", "/api/v1/training_runs/model_fake_0123456789abcdef"), ("POST", "/api/v1/forward"),
    ("POST", "/api/v1/forward_backward"), ("POST", "/api/v1/optim_step"), ("POST", "/api/v1/asample"),
    ("POST", "/api/v1/sample"), ("POST", "/api/v1/create_sampling_client"), ("POST", "/api/v1/save_weights"),
    ("POST", "/api/v1/save_weights_for_sampler"), ("POST", "/api/v1/save_weights_external"),
    ("POST", "/api/v1/load_weights"), ("GET", "/api/v1/training_runs/model_fake_0123456789abcdef/checkpoints"),
    ("DELETE", "/api/v1/training_runs/model_fake_0123456789abcdef/checkpoints/weights/c"),
    ("GET", "/api/v1/training_runs/model_fake_0123456789abcdef/checkpoints/external_weights/n/external_weights_urls"),
    ("GET", "/api/v1/external_weights/model_fake_0123456789abcdef/n/adapter_config.json"),
    ("POST", "/api/v1/weights_info"), ("POST", "/api/v1/retrieve_future/req_fake_0123456789abcdef"),
    ("POST", "/api/v1/retrieve_future"), ("POST", "/api/v1/retrieve_futures"), ("POST", "/api/v1/cancel_future"),
    ("POST", "/api/v1/cleanup_futures"), ("GET", "/api/v1/billing/usage/checkpoints/current"),
]


async def main() -> int:
    import httpx
    import orjson
    importlib.import_module("training.gateway.__main__")   # the entry point's imports count too
    from training.gateway.app import build_app, engines_face
    from training.gateway.registry import Registry
    from .support import RecordingEngine, config, serve_uds

    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "e.sock")
    bodies = {"application/json": orjson.dumps({"model_id": "model_fake_0123456789abcdef", "request_id": "req_fake_0123456789abcdef"}),
              "application/x-protobuf": b"\n\x1cmodel_fake_0123456789abcdef"}
    engine = RecordingEngine()
    for server, client in (("asgi", "h11"), ("fastapi", "httpx")):
        registry = Registry()
        app = build_app(config(server, client), registry)
        async with serve_uds(engine, sock):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=engines_face(registry)), base_url="http://g") as eng:
                r = await eng.post("/engines/register", json={"name": "fake", "backend": "fake", "gpus": [], "socket": sock, "pid": 1})
                assert r.status_code == 200, r.text
                assert (await eng.get("/engines")).status_code == 200
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as sdk:
                for method, path in ROUTES:
                    for ct, body in bodies.items():
                        r = await sdk.request(method, path, content=body if method != "GET" else b"",
                                              headers={"X-API-Key": "gw-test-key", "content-type": ct})
                        assert r.status_code == 200, (server, method, path, r.status_code, r.text)
            await app._forwarder.aclose()
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in HEAVY or m.startswith("training.backends"))
    if loaded:
        print("HEAVY:", " ".join(loaded))
        return 1
    print("LIGHT")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
