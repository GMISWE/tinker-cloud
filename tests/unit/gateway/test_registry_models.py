"""The model registry is written at submit (R8.11): the gateway mints the id, records the model under its
session before the engine answers, and drops the row when the engine has deleted the model or never had it."""
import re

import orjson

from .support import Harness, run, serve_uds

ID = re.compile(r"^model_fake_[0-9a-f]{16}$")


def _create(session_id: str) -> dict:
    return {"session_id": session_id, "model_seq_id": 0, "base_model": "fake/tiny", "lora_config": {"rank": 2}, "optimizer_config": {"type": "adamw"}}


def test_unknown_session_is_404_before_the_engine_sees_anything(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                r = await h.post_json("/api/v1/create_model", _create("nope"))
                assert r.status_code == 404 and r.json()["error"] == "Session not found: nope", r.text
                assert h.engine.requests == []
        finally:
            await h.close()
    run(main())


def test_the_model_is_recorded_before_the_engine_answers_and_dropped_on_its_404(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        recorded_at_request = []
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sid = await h.create_session()
                def at_request(_req):
                    session = h.store.get_session(sid)
                    recorded_at_request.append(session.model_ids if session is not None else None)
                h.engine.on_request = at_request
                r = await h.post_json("/api/v1/create_model", _create(sid))
                assert r.status_code == 200, r.text
                model_id = r.json()["model_id"]
                assert ID.match(model_id), model_id
                assert h.engine.requests[-1]["headers"]["x-model-id"] == model_id   # the engine creates it under this id
                assert orjson.loads(h.engine.requests[-1]["body"]) == _create(sid)   # the body is untouched
                assert recorded_at_request == [[model_id]]   # in the registry while the engine was still answering
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == [model_id]
                # a second model sorts by seq id
                r2 = await h.post_json("/api/v1/create_model", {**_create(sid), "model_seq_id": 1})
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == [model_id, r2.json()["model_id"]]
                # the engine never held it (its create failed): the delete gets the engine's 404 and the row goes
                h.engine.gone.add(model_id)
                r = await h.post_json("/api/v1/delete_model", {"model_id": model_id})
                assert r.status_code == 404
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == [r2.json()["model_id"]]
                # a delete the engine performed drops the row too
                r = await h.post_json("/api/v1/delete_model", {"model_id": r2.json()["model_id"]})
                assert r.status_code == 200
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == []
                # a create that fails inside the engine's task: the engine reports it and the row goes at once
                r3 = (await h.post_json("/api/v1/create_model", {**_create(sid), "model_seq_id": 2})).json()["model_id"]
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == [r3]
                assert (await h.engines.delete(f"/engines/fake/models/{r3}")).status_code == 204
                assert (await h.engines.delete(f"/engines/fake/models/{r3}")).status_code == 404
                assert (await h.get(f"/api/v1/sessions/{sid}")).json()["training_run_ids"] == []
        finally:
            await h.close()
    run(main())


def test_a_create_without_session_fields_is_the_engines_422(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                r = await h.post_json("/api/v1/create_model", {"base_model": "fake/tiny"})
                assert r.status_code == 200   # the recording engine answers 200; a real engine's pydantic says 422
                assert "x-model-id" not in h.engine.requests[-1]["headers"]
        finally:
            await h.close()
    run(main())
