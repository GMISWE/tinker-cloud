"""A bare base_model sampler on an engine that registered base_model_sampling: the gateway asks the
engine whether it serves that base model, records a sampler with no model, and names the base model
(not a model id) on every sample request it routes for it."""
import orjson

from .support import Harness, run, serve_uds

PROMPT = {"prompt": {"tokens": [1, 2]}, "num_samples": 1, "sampling_params": {"max_tokens": 1}}


def test_a_base_model_sampler_routes_by_base_model(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register(base_model_sampling=True)
                sid = await h.create_session()
                h.engine.target = {"model_id": None, "pinned_version": None, "base_model": "fake/tiny"}
                r = await h.post_json("/api/v1/create_sampling_session",
                                      {"session_id": sid, "sampling_session_seq_id": 0, "base_model": "fake/tiny"})
                assert r.status_code == 200, r.text
                sampler = r.json()["sampling_session_id"]
                assert [q["path"] for q in h.engine.requests] == ["/api/v1/resolve_sampler_target"]
                assert orjson.loads(h.engine.requests[0]["body"]) == {"base_model": "fake/tiny"}

                for doc in ({**PROMPT, "sampling_session_id": sampler},   # sample()
                            {**PROMPT, "sampling_session_id": sampler, "prompt_logprobs": True}):   # compute_logprobs()
                    h.engine.requests.clear()
                    r = await h.post_json("/api/v1/asample", doc)
                    assert r.status_code == 200, r.text
                    (q,) = h.engine.requests
                    assert q["path"] == "/api/v1/asample"
                    assert q["headers"]["x-base-model"] == "fake/tiny"
                    assert "x-model-id" not in q["headers"] and "x-pinned-version" not in q["headers"]
        finally:
            await h.close()
    run(main())


def test_a_base_model_the_engine_does_not_serve_is_its_400(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register(base_model_sampling=True)
                sid = await h.create_session()
                h.engine.answers["/api/v1/resolve_sampler_target"] = 400
                r = await h.post_json("/api/v1/create_sampling_session",
                                      {"session_id": sid, "sampling_session_seq_id": 0, "base_model": "other/model"})
                assert r.status_code == 400, r.text
                assert orjson.loads(h.engine.requests[0]["body"]) == {"base_model": "other/model"}
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["sampler_ids"] == []
        finally:
            await h.close()
    run(main())


def test_an_sdks_own_base_model_header_never_reaches_the_engine(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register(base_model_sampling=True)
                sid = await h.create_session()
                h.engine.target = {"model_id": None, "pinned_version": None, "base_model": "fake/tiny"}
                r = await h.post_json("/api/v1/create_sampling_session",
                                      {"session_id": sid, "sampling_session_seq_id": 0, "base_model": "fake/tiny"})
                sampler = r.json()["sampling_session_id"]
                h.engine.requests.clear()
                # the engine reads the first X-Base-Model it gets: a forged one ahead of the gateway's would win
                r = await h.post("/api/v1/asample", orjson.dumps({**PROMPT, "sampling_session_id": sampler}),
                                 **{"content-type": "application/json", "X-Base-Model": "other/model"})
                assert r.status_code == 200, r.text
                (q,) = h.engine.requests
                assert [v for k, v in q["raw_headers"] if k == "x-base-model"] == ["fake/tiny"]
        finally:
            await h.close()
    run(main())
