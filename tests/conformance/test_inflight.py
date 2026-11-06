"""kind="inflight" (SDK types/sampling_client_kind.py): every inflight client of a
training model samples the model's shared, moving weights — those saved when the
client was created, or newer. A snapshot client keeps the version it was saved at."""
from tinker import types

from .conftest import make_datum

PARAMS = types.SamplingParams(max_tokens=1)


def _sample_ops(server, model_id):
    return [t for t in server.trace() if t["op"] == "sample" and t["model_id"] == model_id]


def _step(tc):
    tc.forward_backward([make_datum([1, 2, 3, 4])], "cross_entropy")
    tc.optim_step(types.AdamParams(learning_rate=0.1)).result()


def _sample(client):
    return client.sample(prompt=types.ModelInput.from_ints([9]), num_samples=1, sampling_params=PARAMS).result()


def test_an_inflight_client_follows_later_weights(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    _step(tc)
    inflight = tc.save_weights_and_get_sampling_client(kind="inflight").result()
    snapshot = tc.save_weights_and_get_sampling_client().result()
    _step(tc)
    _step(tc)
    _sample(inflight)
    assert _sample_ops(server, tc.model_id)[-1]["pinned_version"] is None   # the live weights
    _sample(snapshot)
    assert _sample_ops(server, tc.model_id)[-1]["pinned_version"] == 1      # its own save


def test_each_inflight_save_is_its_own_session(service_client):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    a = tc.save_weights_and_get_sampling_client(kind="inflight").result()
    b = tc.save_weights_and_get_sampling_client(kind="inflight").result()
    assert a._sampling_session_id != b._sampling_session_id


def test_an_inflight_save_with_a_path_is_400(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    r = server.post("/api/v1/save_weights_for_sampler", {"model_id": tc.model_id, "path": "named", "kind": "inflight"})
    assert r.status_code == 400, r.text
    assert "inflight" in r.text
    r = server.post("/api/v1/save_weights_for_sampler", {"model_id": tc.model_id, "kind": "rolling"})
    assert r.status_code == 422, r.text
