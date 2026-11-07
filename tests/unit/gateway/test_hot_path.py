"""The sampling hot path does no I/O of the gateway's own: a warm sampler cache serves asample and
retrieve_futures with zero store reads, no proto parse and no pydantic body model; a fresh worker
reads the store once for a sampler it has not seen."""
import orjson
from fastapi.routing import APIRoute

from training.gateway import routing
from training.gateway.sessions import SessionStore

from .support import MODEL, Harness, run, serve_uds

PROMPT = {"prompt": {"tokens": [1, 2, 3]}, "num_samples": 1, "sampling_params": {"max_tokens": 2}}


class _CountingRoute:
    """Stands in for the proto skeleton class so a parse on the JSON path shows up."""
    parses = 0

    def ParseFromString(self, _: bytes) -> None:
        _CountingRoute.parses += 1
        raise AssertionError("the JSON hot path parsed a proto skeleton")


def test_a_warm_sampler_costs_no_store_read_no_proto_parse_no_body_model(tmp_path, monkeypatch):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                r = await h.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0, "model_path": f"tinker://{MODEL}/sampler_weights/s"})
                sampler_id = r.json()["sampling_session_id"]
                monkeypatch.setattr(routing, "_ModelRoute", _CountingRoute)
                asample = orjson.dumps({**PROMPT, "sampling_session_id": sampler_id})
                poll = orjson.dumps({"target": {"type": "sampling_session", "sampling_session_id": sampler_id}, "prev_cursor": 0})
                reads = h.store.reads
                for _ in range(100):
                    assert (await h.post("/api/v1/asample", asample, **{"content-type": "application/json"})).status_code == 200
                    assert (await h.post("/api/v1/retrieve_futures", poll, **{"content-type": "application/json"})).status_code == 200
                assert h.store.reads == reads, h.store.reads - reads
                assert _CountingRoute.parses == 0
                assert h.engine.requests[-2]["headers"]["x-model-id"] == MODEL   # the last asample; the poll came after
                # the catch-all route declares no body model: FastAPI parses nothing on the way through
                catch_all = [r for r in h.app.routes if isinstance(r, APIRoute) and r.path == "/{path:path}"]
                assert len(catch_all) == 1 and catch_all[0].dependant.body_params == []
        finally:
            await h.close()
    run(main())


def test_a_fresh_worker_reads_the_store_once_per_unseen_sampler(tmp_path):
    async def main():
        path = tmp_path / "sessions.db"
        a = Harness(str(tmp_path), store=SessionStore(path))
        (tmp_path / "b").mkdir()
        b = Harness(str(tmp_path / "b"), store=SessionStore(path))
        try:
            async with serve_uds(a.engine, a.engine_socket):
                await a.register()
                b.registry.register("fake", "fake", [], a.engine_socket, 4242, False)
                sid = await a.create_session()
                r = await a.post_json("/api/v1/create_sampling_session", {"session_id": sid, "sampling_session_seq_id": 0, "model_path": f"tinker://{MODEL}/sampler_weights/s"})
                sampler_id = r.json()["sampling_session_id"]
                body = {**PROMPT, "sampling_session_id": sampler_id}
                reads = b.store.reads
                assert (await b.post_json("/api/v1/asample", body)).status_code == 200
                assert b.store.reads == reads + 1          # the miss: one read, then cached
                for _ in range(10):
                    assert (await b.post_json("/api/v1/asample", body)).status_code == 200
                assert b.store.reads == reads + 1
                assert a.engine.requests[-1]["headers"]["x-model-id"] == MODEL
        finally:
            await a.close()
            await b.close()
    run(main())
