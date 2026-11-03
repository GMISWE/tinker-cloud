"""A sampling request naming a sampler reaches the engine with X-Model-Id and X-Pinned-Version and
its body untouched; a request naming no sampler carries neither; a sampler the gateway does not know is 404."""
import orjson

from .support import HEX, MODEL, Harness, run, serve_uds

PROMPT = {"prompt": {"tokens": [1, 2, 3]}, "num_samples": 1, "sampling_params": {"max_tokens": 2}}


def _sampler(h: Harness, sid: str, sampler_id: str = "s_1_abc", pinned=3):
    h.store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
    return h.store.add_sampler(sampler_id, sid, MODEL, "fake/tiny", f"tinker://{MODEL}/sampler_weights/s", pinned)


def test_a_sampler_travels_as_the_two_headers_with_the_body_untouched(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                _sampler(h, sid)
                for path, body in (("/api/v1/asample", {**PROMPT, "sampling_session_id": "s_1_abc", "seq_id": 7}),
                                   ("/api/v1/sample", {"prompts": [[1]], "num_samples": 1, "sampling_session_id": "s_1_abc"})):
                    raw = orjson.dumps(body)
                    r = await h.post(path, raw, **{"content-type": "application/json"})
                    assert r.status_code == 200, r.text
                    seen = h.engine.requests[-1]
                    assert seen["body"] == raw
                    assert seen["headers"]["x-model-id"] == MODEL and seen["headers"]["x-pinned-version"] == "3"
                # a sampler saved at no particular version: no pinned header
                h.store.add_sampler("s_1_live", sid, MODEL, "fake/tiny", None, None)
                await h.post_json("/api/v1/asample", {**PROMPT, "sampling_session_id": "s_1_live"})
                seen = h.engine.requests[-1]
                assert seen["headers"]["x-model-id"] == MODEL and "x-pinned-version" not in seen["headers"]
                # model_path and base_model requests are the engine's to resolve: no headers
                for body in ({**PROMPT, "model_path": f"tinker://{MODEL}/weights/s"}, {**PROMPT, "base_model": "fake/tiny"}, PROMPT):
                    await h.post_json("/api/v1/asample", body)
                    seen = h.engine.requests[-1]
                    assert "x-model-id" not in seen["headers"] and "x-pinned-version" not in seen["headers"]
                # the two headers an SDK forges are dropped; the engine sees the gateway's once, never a second copy
                await h.post("/api/v1/asample", orjson.dumps({**PROMPT, "sampling_session_id": "s_1_abc"}),
                             **{"content-type": "application/json", "X-Model-Id": "model_fake_ffffffffffffffff", "X-Pinned-Version": "99"})
                raw = h.engine.requests[-1]["raw_headers"]
                assert [v for k, v in raw if k == "x-model-id"] == [MODEL], raw
                assert [v for k, v in raw if k == "x-pinned-version"] == ["3"], raw
                await h.post("/api/v1/asample", orjson.dumps({**PROMPT, "sampling_session_id": "s_1_live"}),
                             **{"content-type": "application/json", "X-Pinned-Version": "99"})
                raw = h.engine.requests[-1]["raw_headers"]
                assert [v for k, v in raw if k == "x-model-id"] == [MODEL] and "x-pinned-version" not in dict(raw), raw
                # a request that names no sampler carries neither header, whatever the SDK sent
                await h.post("/api/v1/asample", orjson.dumps({**PROMPT, "model_path": f"tinker://{MODEL}/weights/s"}),
                             **{"content-type": "application/json", "X-Model-Id": "model_fake_ffffffffffffffff", "X-Pinned-Version": "99"})
                raw = h.engine.requests[-1]["raw_headers"]
                assert "x-model-id" not in dict(raw) and "x-pinned-version" not in dict(raw), raw
        finally:
            await h.close()
    run(main())


def test_an_unknown_sampler_is_404_at_the_gateway_and_the_engine_sees_nothing(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                for path, body in (("/api/v1/asample", {**PROMPT, "sampling_session_id": "nope"}),
                                   ("/api/v1/sample", {"prompts": [[1]], "num_samples": 1, "sampling_session_id": "nope"}),
                                   ("/api/v1/retrieve_futures", {"target": {"type": "sampling_session", "sampling_session_id": "nope"}, "prev_cursor": 0})):
                    r = await h.post_json(path, body)
                    assert r.status_code == 404 and r.json()["error"] == "Unknown sampling_session_id: nope", (path, r.text)
                assert h.engine.requests == []
                sid = await h.create_session()
                _sampler(h, sid)
                r = await h.post_json("/api/v1/retrieve_futures", {"target": {"type": "sampling_session", "sampling_session_id": "s_1_abc"}, "prev_cursor": 0})
                assert r.status_code == 200 and h.engine.requests[-1]["path"] == "/api/v1/retrieve_futures"
                # a pre-split sampler row with no model behind it
                h.store.add_sampler("s_1_none", sid, None, "fake/tiny", None, None)
                r = await h.post_json("/api/v1/asample", {**PROMPT, "sampling_session_id": "s_1_none"})
                assert r.status_code == 400 and "base-model sampling" in r.json()["error"]
        finally:
            await h.close()
    run(main())


def test_an_engines_ephemeral_sampler_is_recorded_through_the_engines_face(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                h.store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
                doc = {"sampler_id": f"{MODEL}_1_{HEX[:8]}", "model_id": MODEL, "pinned_version": 2,
                       "model_path": f"tinker://{MODEL}/sampler_weights/e", "base_model": None}
                r = await h.engines.post("/engines/fake/samplers", json=doc)
                assert r.status_code == 200 and r.json() == {"session_id": sid}, r.text
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["sampler_ids"] == [doc["sampler_id"]]
                assert (await h.get(f"/api/v1/samplers/{doc['sampler_id']}")).json()["base_model"] == "fake/tiny"   # from the model's row
                await h.post_json("/api/v1/asample", {**PROMPT, "sampling_session_id": doc["sampler_id"]})
                assert h.engine.requests[-1]["headers"]["x-pinned-version"] == "2"
                r = await h.engines.post("/engines/fake/samplers", json={**doc, "model_id": "model_fake_ffffffffffffffff"})
                assert r.status_code == 404   # a model no session owns
                assert (await h.engines.post("/engines/ghost/samplers", json=doc)).status_code == 404
                assert (await h.engines.post("/engines/fake/samplers", content=b"{}")).status_code == 400
        finally:
            await h.close()
    run(main())
