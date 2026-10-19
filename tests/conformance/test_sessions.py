"""Session lifecycle endpoints the 0.33 SDK calls: finish (first-wins), join_sampling_session,
get_session user_metadata, and the checkpoint storage usage query."""
import pickle

import requests
from tinker import types

from .conftest import API_KEY


def _get(server, path, **params):
    return requests.get(server.base_url + path, headers={"X-API-Key": API_KEY}, params=params or None, timeout=30)


def _create_session(server, user_metadata=None):
    r = server.post("/api/v1/create_session", {"tags": [], "user_metadata": user_metadata or {}, "sdk_version": "t"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _finish(server, session_id, status, detail=None):
    return server.post(f"/api/v1/sessions/{session_id}/finish", {"reason": {"type": status}, "detail": detail})


# ---- finish ------------------------------------------------------------------

def test_finish_is_first_wins(server):
    sid = _create_session(server)
    first = _finish(server, sid, "success", "done")
    assert first.status_code == 200, first.text
    assert first.json()["reason"] == "success" and first.json()["detail"] == "done"
    second = _finish(server, sid, "errored", "crash")
    assert second.status_code == 200, second.text
    assert second.json()["reason"] == "success" and second.json()["detail"] == "done"
    assert second.json()["finished_at"] == first.json()["finished_at"]
    # the session stays readable; the SDK stops its heartbeat loop on 410
    assert _get(server, f"/api/v1/sessions/{sid}").status_code == 200
    hb = server.post("/api/v1/session_heartbeat", {"session_id": sid})
    assert hb.status_code == 410, hb.text


def test_finish_unknown_session_is_404(server):
    assert _finish(server, "no-such-session", "interrupted").status_code == 404


def test_finish_rejects_unknown_reason(server):
    sid = _create_session(server)
    assert _finish(server, sid, "cancelled").status_code == 422


def test_sdk_close_finishes_the_session(server):
    import tinker
    sc = tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY)
    tc = sc.create_lora_training_client(base_model="fake/tiny", rank=2)
    sid = sc._session_holder.get_session_id()
    sc.close("success", detail="protocol test").result()
    # models are untouched by finish
    assert _get(server, f"/api/v1/sessions/{sid}").json()["training_run_ids"] == [tc.model_id]
    assert server.post("/api/v1/session_heartbeat", {"session_id": sid}).status_code == 410


# ---- join_sampling_session -----------------------------------------------------

def test_join_counter_sequence_and_404(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sampler = tc.save_weights_and_get_sampling_client("join")
    sampler_id = sampler._sampling_session_id
    counters = [server.post("/api/v1/join_sampling_session", {"sampling_session_id": sampler_id}).json()["client_counter"]
                for _ in range(3)]
    assert counters == [1, 2, 3]
    r = server.post("/api/v1/join_sampling_session", {"sampling_session_id": "nope"})
    assert r.status_code == 404


def test_unpickled_sampling_client_joins(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sampler = tc.save_weights_and_get_sampling_client("clone")
    clone = pickle.loads(pickle.dumps(sampler))
    assert clone._cloned_sampler_id is None  # waits for the server-allocated id
    out = clone.sample(prompt=types.ModelInput.from_ints([1, 2]), num_samples=1,
                       sampling_params=types.SamplingParams(max_tokens=2)).result()
    assert len(out.sequences) == 1
    assert clone._cloned_sampler_id == 1
    assert pickle.loads(pickle.dumps(sampler)).sample(
        prompt=types.ModelInput.from_ints([1]), num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1)).result().sequences
    assert server.post("/api/v1/join_sampling_session", {"sampling_session_id": sampler._sampling_session_id}
                       ).json()["client_counter"] == 3


def test_client_config_advertises_join(server):
    r = server.post("/api/v1/client/config", {"sdk_version": "0.33.1"})
    assert r.json()["sample_join_sampling_session"] is True


# ---- get_session user_metadata ---------------------------------------------------

def test_get_session_user_metadata_as_strings(service_client, server):
    sid = _create_session(server, {"run": "exp-7", "seed": 3, "tags": ["a", "b"]})
    body = _get(server, f"/api/v1/sessions/{sid}").json()
    assert body["user_metadata"] == {"run": "exp-7", "seed": "3", "tags": '["a", "b"]'}
    # the SDK's GetSessionResponse validates dict[str, str]
    parsed = service_client.create_rest_client().get_session(sid).result()
    assert parsed.user_metadata == body["user_metadata"]
    empty = _create_session(server)
    assert _get(server, f"/api/v1/sessions/{empty}").json()["user_metadata"] == {}


# ---- billing --------------------------------------------------------------------

def test_checkpoint_storage_usage_aggregates_completed_saves(service_client, server):
    rest = service_client.create_rest_client()
    before = rest.get_current_checkpoint_storage_usage().result()
    assert len(before.data) == 1 and before.effective_rate_usd_per_gigabyte_month is None
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.save_state("u1").result()
    tc.save_weights_for_sampler("u2").result()
    after = rest.get_current_checkpoint_storage_usage(project_id="proj-x").result()
    row = after.data[0]
    assert row.checkpoint_count == before.data[0].checkpoint_count + 2
    assert row.size_bytes > before.data[0].size_bytes
    assert row.size_gigabytes == row.size_bytes / 2 ** 30
    assert row.project_id == "proj-x" and row.estimated_monthly_cost_usd is None
    raw = _get(server, "/api/v1/billing/usage/checkpoints/current")
    assert raw.status_code == 200 and raw.json()["data"][0]["project_id"] is None
