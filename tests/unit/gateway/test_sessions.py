"""Sessions and samplers live in the gateway: the eight routes are answered there and the engine sees none of them."""
import orjson

from .support import API_KEY, Harness, run, serve_uds

SESSION_PATHS = ("/api/v1/create_session", "/api/v1/session_heartbeat", "/api/v1/create_sampling_session",
                 "/api/v1/join_sampling_session", "/api/v1/sessions", "/api/v1/samplers")


def test_session_lifecycle_is_the_gateways(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                assert (await h.post_json("/api/v1/session_heartbeat", {"session_id": sid})).status_code == 200
                r = await h.post_json("/api/v1/session_heartbeat", {"session_id": "nope"})
                assert r.status_code == 404 and r.json()["error"] == "Session not found: nope"
                assert sid in (await h.get("/api/v1/sessions")).json()["sessions"]
                r = await h.get(f"/api/v1/sessions/{sid}")
                assert r.status_code == 200 and r.json()["training_run_ids"] == [] and r.json()["sampler_ids"] == []
                assert (await h.get("/api/v1/sessions/nope")).status_code == 404
                first = await h.post_json(f"/api/v1/sessions/{sid}/finish", {"reason": {"type": "success"}, "detail": "done"})
                assert first.status_code == 200 and first.json()["reason"] == "success", first.text
                second = await h.post_json(f"/api/v1/sessions/{sid}/finish", {"reason": {"type": "errored"}, "detail": "crash"})
                assert second.json()["reason"] == "success" and second.json()["finished_at"] == first.json()["finished_at"]
                r = await h.post_json("/api/v1/session_heartbeat", {"session_id": sid})
                assert r.status_code == 410 and "finished (success)" in r.json()["error"]
                assert (await h.post_json("/api/v1/sessions/nope/finish", {"reason": {"type": "success"}})).status_code == 404
                assert (await h.post_json("/api/v1/create_session", {}, key="")).status_code == 401
                assert [q["path"] for q in h.engine.requests if q["path"].startswith(SESSION_PATHS)] == []
        finally:
            await h.close()
    run(main())


def test_samplers_are_minted_joined_and_read_at_the_gateway(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": "nope", "sampling_session_seq_id": 0, "model_path": "tinker://m/weights/x"})
                assert r.status_code == 404
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0, "base_model": "fake/tiny"})
                assert r.status_code == 400 and "base-model sampling is not supported" in r.json()["error"]
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0})
                assert r.status_code == 400
                assert h.engine.requests == []   # none of those reached the engine
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0, "model_path": "tinker://model_fake_0123456789abcdef/sampler_weights/s"})
                assert r.status_code == 200, r.text
                sampler_id = r.json()["sampling_session_id"]
                assert sampler_id.startswith(f"{sid}_0_")
                # the one engine call: which model the path names, at which version
                assert [q["path"] for q in h.engine.requests] == ["/api/v1/resolve_sampler_target"]
                assert orjson.loads(h.engine.requests[0]["body"]) == {"model_path": "tinker://model_fake_0123456789abcdef/sampler_weights/s"}
                assert h.engine.requests[0]["headers"]["x-api-key"] == API_KEY
                counters = [(await h.post_json("/api/v1/join_sampling_session", {"sampling_session_id": sampler_id})).json()["client_counter"] for _ in range(3)]
                assert counters == [1, 2, 3]
                assert (await h.post_json("/api/v1/join_sampling_session", {"sampling_session_id": "nope"})).status_code == 404
                r = await h.get(f"/api/v1/samplers/{sampler_id}")
                assert r.status_code == 200 and r.json()["base_model"] == "fake/tiny" and r.json()["model_path"].startswith("tinker://")
                assert (await h.get("/api/v1/samplers/nope")).status_code == 404
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["sampler_ids"] == [sampler_id]
                assert len(h.engine.requests) == 1
        finally:
            await h.close()
    run(main())


def test_the_engines_verdict_on_a_model_path_is_relayed(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                h.engine.answers["/api/v1/resolve_sampler_target"] = 404
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0, "model_path": "tinker://model_fake_0123456789abcdef/weights/x"})
                assert r.status_code == 404 and r.json()["error"] == "engine says 404", r.text
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["sampler_ids"] == []
        finally:
            await h.close()
    run(main())
