"""/health asks the engine (R8.11): its model list is the engine's, the engines the gateway knows are listed,
and an engine that does not answer makes the gateway say so within 2 s."""
import time

from .support import Harness, run, serve_uds


def test_health_reports_the_engines_models_and_the_registry(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            r = await h.sdk.get("/health")
            assert r.status_code == 503 and r.json()["error"] == "no engine registered"
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                h.engine.model_ids = ["model_fake_0000000000000001", "model_fake_0000000000000002"]
                h.store.add_model("s", "model_fake_0000000000000009", 0, "fake/tiny", "fake")   # in the registry, not on the engine
                for path in ("/health", "/api/v1/health"):
                    r = await h.sdk.get(path)
                    assert r.status_code == 200, r.text
                    doc = r.json()
                    assert doc["status"] == "healthy" and doc["model_ids"] == h.engine.model_ids and doc["active_training_clients"] == 2
                    assert doc["engines"] == [{"name": "fake", "state": "UP"}]
            # the engine's socket is gone but its registration is still there
            t0 = time.monotonic()
            r = await h.sdk.get("/health")
            assert r.status_code == 503 and time.monotonic() - t0 < 2.0
            assert r.json()["error"].startswith("engine fake unreachable"), r.text
        finally:
            await h.close()
    run(main())
