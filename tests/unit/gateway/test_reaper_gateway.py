"""The session reaper runs in the gateway: a session silent past the timeout has its models deleted on
the engine, and failed deletes remain queued for another tick."""
import asyncio

import orjson

from .support import MODEL, Harness, run, serve_uds


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


def test_a_model_created_by_load_is_deleted_with_its_session(tmp_path):
    """create-via-load's id is the SDK's `{session}:train:{seq}`, which names no engine: the delete goes to
    the only engine, as the load that created the model did, and is sent once."""
    async def main():
        h = Harness(str(tmp_path))
        h.cfg.session_timeout_s = 0.2
        h.cfg.session_reap_interval_s = 0.05
        deleted = asyncio.Event()

        def on_delete(request):
            if request["path"] == "/api/v1/delete_model":
                deleted.set()

        h.engine.on_request = on_delete
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                r = await h.post_json("/api/v1/load_weights", {"session_id": sid, "model_seq_id": 0,
                                                               "path": f"tinker://{MODEL}/weights/s"})
                assert r.status_code == 200, r.text
                model_id = f"{sid}:train:0"
                assert h.store.model_ids() == [model_id]
                await h.gateway.start()
                await asyncio.wait_for(deleted.wait(), timeout=2)
                await asyncio.sleep(0.2)   # more ticks: a confirmed delete is not sent again
                deletes = [orjson.loads(q["body"])["model_id"] for q in h.engine.requests if q["path"] == "/api/v1/delete_model"]
                assert deletes == [model_id], deletes
                assert h.store.get_session(sid) is None and h.store.claim_expired(-1) == []
        finally:
            await h.close()
    run(main())


def test_a_failed_engine_delete_is_retried(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        h.cfg.session_timeout_s = 0.2
        h.cfg.session_reap_interval_s = 0.05
        deletes = []
        retried = asyncio.Event()

        def fail_first_delete(request):
            if request["path"] != "/api/v1/delete_model":
                return
            deletes.append(orjson.loads(request["body"])["model_id"])
            if len(deletes) == 1:
                h.engine.answers[request["path"]] = 500
            else:
                h.engine.answers.pop(request["path"], None)
                retried.set()

        h.engine.on_request = fail_first_delete
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                session_id = await h.create_session()
                model_id = (await h.post_json("/api/v1/create_model", _create(session_id))).json()["model_id"]
                await h.gateway.start()
                await asyncio.wait_for(retried.wait(), timeout=2)
                await asyncio.sleep(0.1)
                assert deletes == [model_id, model_id]
                assert h.store.get_session(session_id) is None
        finally:
            await h.close()
    run(main())


def test_a_failed_reap_is_durable_and_not_claimed_twice(tmp_path):
    from training.gateway.sessions import SessionStore

    path = tmp_path / "sessions.db"
    first, second = SessionStore(path), SessionStore(path)
    session_id = first.create_session([], {}, "t", None)
    model_id = "model_fake_0123456789abcdef"
    first.add_model(session_id, model_id, 0, "fake/tiny", "fake")
    claimed = [(session_id, [model_id])]
    assert first.claim_expired(-1) == claimed
    assert second.claim_expired(-1) == []
    first.retry_reap(model_id)
    first.close()
    second.close()
    reopened = SessionStore(path)
    assert reopened.claim_expired(-1) == claimed
    reopened.remove_model(model_id)
    reopened.close()


def test_a_claim_its_worker_never_settled_is_taken_again_once_stale(tmp_path):
    """A worker killed mid-free neither acknowledges nor releases its claim: the claim stands for the
    lease, then the model is claimed again, and that claim stands in turn."""
    from training.gateway.sessions import SessionStore

    path = tmp_path / "sessions.db"
    killed = SessionStore(path)
    session_id = killed.create_session([], {}, "t", None)
    model_id = "model_fake_0123456789abcdef"
    killed.add_model(session_id, model_id, 0, "fake/tiny", "fake")
    claimed = [(session_id, [model_id])]
    assert killed.claim_expired(-1) == claimed
    killed.close()   # no remove_model, no retry_reap: the worker is gone
    survivor = SessionStore(path)
    assert survivor.claim_expired(-1) == []
    assert survivor.claim_expired(-1, lease_s=0) == claimed   # the lease runs out
    assert survivor.claim_expired(-1) == []
    survivor.remove_model(model_id)
    survivor.close()


def test_a_late_failure_releases_only_its_own_claim(tmp_path):
    """Worker A's delete outlives the lease and worker B claims the model again. A's failure, when it
    comes, leaves B's claim standing, so no third delete goes out; B's own failure releases it."""
    from training.gateway.sessions import SessionStore

    path = tmp_path / "sessions.db"
    a, b = SessionStore(path), SessionStore(path)
    session_id = a.create_session([], {}, "t", None)
    model_id = "model_fake_0123456789abcdef"
    a.add_model(session_id, model_id, 0, "fake/tiny", "fake")
    claimed = [(session_id, [model_id])]
    assert a.claim_expired(-1) == claimed
    assert b.claim_expired(-1, lease_s=0) == claimed   # A's delete is still running when the lease runs out
    a.retry_reap(model_id)
    assert a.claim_expired(-1) == [] and b.claim_expired(-1) == []
    b.retry_reap(model_id)
    assert a.claim_expired(-1) == claimed
    a.remove_model(model_id)
    a.close()
    b.close()


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
                b.registry.register("fake", "fake", [], a.engine_socket, 4242, False)
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
