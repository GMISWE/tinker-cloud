"""The session reaper runs in the gateway: a session silent past the timeout has its models deleted on
the engine, and a gateway restart frees nothing."""
import asyncio

import orjson

from .support import Harness, run, serve_uds


def _create(session_id: str) -> dict:
    return {"session_id": session_id, "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}


def test_an_expired_session_has_its_models_deleted_on_the_engine(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        h.cfg.session_timeout_s = 0.6
        h.cfg.session_reap_interval_s = 0.1
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                await h.gateway.start()
                quiet, live = await h.create_session(), await h.create_session()
                m_quiet = (await h.post_json("/api/v1/create_model", _create(quiet))).json()["model_id"]
                m_live = (await h.post_json("/api/v1/create_model", _create(live))).json()["model_id"]
                for _ in range(6):
                    await asyncio.sleep(0.2)
                    assert (await h.post_json("/api/v1/session_heartbeat", {"session_id": live})).status_code == 200
                deletes = [orjson.loads(q["body"])["model_id"] for q in h.engine.requests if q["path"] == "/api/v1/delete_model"]
                assert deletes == [m_quiet], deletes                      # within 2× the timeout; the live session's model untouched
                assert h.store.get_session(quiet) is None and h.store.model_ids() == [m_live]
                assert (await h.get(f"/api/v1/sessions/{live}")).json()["training_run_ids"] == [m_live]
                # the gateway stops: the engine's models stay (it keeps training through a gateway restart)
                await h.gateway.stop()
                assert [q for q in h.engine.requests if q["path"] == "/api/v1/delete_model"] and len(
                    [q for q in h.engine.requests if q["path"] == "/api/v1/delete_model"]) == 1
        finally:
            await h.close()
    run(main())


def test_two_workers_free_each_expired_model_once(tmp_path):
    """Two gateways over one store file stand in for two workers; a claim is one transaction."""
    from training.gateway.sessions import SessionStore

    async def main():
        path = tmp_path / "sessions.db"
        a = Harness(str(tmp_path), store=SessionStore(path))
        b = Harness(str(tmp_path / "b"), store=SessionStore(path))
        for h in (a, b):
            h.cfg.session_timeout_s = 0.5
            h.cfg.session_reap_interval_s = 0.05
        try:
            (tmp_path / "b").mkdir()
            async with serve_uds(a.engine, a.engine_socket):
                await a.register()
                b.registry.register("fake", "fake", [], a.engine_socket, 4242)
                sid = await a.create_session()
                models = [(await a.post_json("/api/v1/create_model", {**_create(sid), "model_seq_id": i})).json()["model_id"] for i in range(3)]
                await a.gateway.start()
                await b.gateway.start()
                await asyncio.sleep(1.2)
                deletes = sorted(orjson.loads(q["body"])["model_id"] for q in a.engine.requests if q["path"] == "/api/v1/delete_model")
                assert deletes == sorted(models), deletes   # each once, through whichever worker claimed the session
                assert a.store.model_ids() == [] and b.store.model_ids() == []
        finally:
            await a.close()
            await b.close()
    run(main())
