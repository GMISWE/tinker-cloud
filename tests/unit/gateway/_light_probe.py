"""Run as a subprocess: import the gateway, serve one request of every route, then list what got imported.

Exit 0 and print LIGHT when none of the heavy packages is in sys.modules; otherwise print them and exit 1.
"""
import asyncio
import importlib
import os
import sys
import tempfile

HEAVY = ("ray", "torch", "numpy", "transformers", "vllm")
# The config's TINKERCLOUD_BACKEND check reads the allow-list from the factory, which imports
# no backend; these come with it. Every other training.backends module is a backend's.
BACKEND_SEAM = frozenset({"training.backends", "training.backends.base", "training.backends.env_config",
                          "training.backends.factory"})
MODEL = "model_fake_0123456789abcdef"
# forwarded routes, each sent once with a JSON body and once with a proto body
FORWARDED = [
    ("GET", "/health"), ("GET", "/api/v1/health"), ("GET", "/api/v1/healthz"),
    ("POST", "/api/v1/client/config"), ("POST", "/api/v1/client/dynamic_config"),
    ("GET", "/api/v1/get_server_capabilities"), ("POST", "/api/v1/telemetry"),
    ("POST", "/api/v1/create_model"), ("POST", "/api/v1/delete_model"),
    ("POST", "/api/v1/unload_model"), ("POST", "/api/v1/get_info"), ("GET", "/api/v1/get_tokenizer?model_id=m"),
    ("GET", f"/api/v1/training_runs/{MODEL}"), ("POST", "/api/v1/forward"),
    ("POST", "/api/v1/forward_backward"), ("POST", "/api/v1/optim_step"), ("POST", "/api/v1/asample"),
    ("POST", "/api/v1/sample"), ("POST", "/api/v1/create_sampling_client"), ("POST", "/api/v1/save_weights"),
    ("POST", "/api/v1/save_weights_for_sampler"), ("POST", "/api/v1/save_weights_external"),
    ("POST", "/api/v1/load_weights"), ("GET", f"/api/v1/training_runs/{MODEL}/checkpoints"),
    ("DELETE", f"/api/v1/training_runs/{MODEL}/checkpoints/weights/c"),
    ("GET", f"/api/v1/training_runs/{MODEL}/checkpoints/external_weights/n/external_weights_urls"),
    ("GET", f"/api/v1/external_weights/{MODEL}/n/adapter_config.json"),
    ("POST", "/api/v1/weights_info"), ("POST", "/api/v1/retrieve_future/req_fake_0123456789abcdef"),
    ("POST", "/api/v1/retrieve_future"), ("POST", "/api/v1/retrieve_futures"), ("POST", "/api/v1/cancel_future"),
    ("POST", "/api/v1/cleanup_futures"), ("GET", "/api/v1/billing/usage/checkpoints/current"),
]


async def main() -> int:
    import httpx
    import orjson
    importlib.import_module("training.gateway.__main__")   # the entry point's imports count too
    from training.gateway.app import build_app
    from training.gateway.registry import Registry
    from training.gateway.sessions import SessionStore
    from .support import RecordingEngine, config, serve_uds

    tmp = tempfile.mkdtemp()
    sock = os.path.join(tmp, "e.sock")
    bodies = {"application/json": orjson.dumps({"model_id": MODEL, "request_id": "req_fake_0123456789abcdef"}),
              "application/x-protobuf": b"\n\x1cmodel_fake_0123456789abcdef"}
    key = {"X-API-Key": "gw-test-key"}
    engine = RecordingEngine()
    for client in ("aiohttp", "httpx"):
        app = build_app(config(client), Registry(), SessionStore(None))
        async with serve_uds(engine, sock):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.state.engines), base_url="http://g", headers=key) as eng:
                r = await eng.post("/engines/register", json={"name": "fake", "backend": "fake", "gpus": [], "socket": sock, "pid": 1,
                                                            "base_model_sampling": False})
                assert r.status_code == 200, r.text
                assert (await eng.get("/engines")).status_code == 200
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as sdk:
                # the gateway's own routes, in the order their ids are minted
                r = await sdk.post("/api/v1/create_session", json={"tags": [], "user_metadata": {}, "sdk_version": "t"}, headers=key)
                assert r.status_code == 200, r.text
                sid = r.json()["session_id"]
                r = await sdk.post("/api/v1/create_sampling_session", headers=key,
                                   json={"session_id": sid, "sampling_session_seq_id": 0, "model_path": f"tinker://{MODEL}/sampler_weights/s"})
                assert r.status_code == 200, r.text
                sampler = r.json()["sampling_session_id"]
                for method, path, body in [
                    ("POST", "/api/v1/session_heartbeat", {"session_id": sid}),
                    ("POST", "/api/v1/join_sampling_session", {"sampling_session_id": sampler}),
                    ("GET", "/api/v1/sessions", None), ("GET", f"/api/v1/sessions/{sid}", None),
                    ("GET", f"/api/v1/samplers/{sampler}", None),
                    ("POST", f"/api/v1/sessions/{sid}/finish", {"reason": {"type": "success"}}),
                ]:
                    r = await sdk.request(method, path, json=body, headers=key)
                    assert r.status_code == 200, (client, method, path, r.status_code, r.text)
                for method, path in FORWARDED:
                    for ct, body in bodies.items():
                        r = await sdk.request(method, path, content=body if method != "GET" else b"",
                                              headers={**key, "content-type": ct})
                        assert r.status_code == 200, (client, method, path, r.status_code, r.text)
                r = await sdk.post("/api/v1/asample", headers={**key, "content-type": "application/json"},
                                   content=orjson.dumps({"sampling_session_id": sampler, "prompt": {"tokens": [1]}}))
                assert r.status_code == 200, r.text
            await app.state.gateway.stop()
    loaded = sorted(m for m in sys.modules
                    if m.split(".")[0] in HEAVY or (m.startswith("training.backends") and m not in BACKEND_SEAM))
    if loaded:
        print("HEAVY:", " ".join(loaded))
        return 1
    print("LIGHT")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
