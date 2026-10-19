"""A sampling request resolves to exactly the model its sampler was minted from; there is no fallback."""
import pytest
from tinker import types

from .conftest import make_datum


def _sample_ops(server, model_id):
    return [t for t in server.trace() if t["op"] == "sample" and t["model_id"] == model_id]


def test_sampler_routes_to_its_own_model(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    b = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    sa = a.save_weights_and_get_sampling_client("sa")
    sb = b.save_weights_and_get_sampling_client("sb")
    params = types.SamplingParams(max_tokens=3, temperature=1.0)
    prompt = types.ModelInput.from_ints([1, 2, 3])
    ra = sa.sample(prompt=prompt, num_samples=2, sampling_params=params).result()
    rb = sb.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
    assert len(ra.sequences) == 2 and len(rb.sequences) == 1
    assert [t["num_samples"] for t in _sample_ops(server, a.model_id)] == [2]
    assert [t["num_samples"] for t in _sample_ops(server, b.model_id)] == [1]


def test_model_path_names_the_model(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    b = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    path = b.save_weights_for_sampler("byname").result().path
    sc = service_client.create_sampling_client(model_path=path)  # create_sampling_session bound to b
    sc.sample(prompt=types.ModelInput.from_ints([4, 5]), num_samples=1,
              sampling_params=types.SamplingParams(max_tokens=2)).result()
    assert len(_sample_ops(server, b.model_id)) == 1
    assert _sample_ops(server, a.model_id) == []


def test_pinned_version_travels_with_the_sampler(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    a.forward_backward([make_datum([1, 2, 3, 4])], "cross_entropy")
    a.optim_step(types.AdamParams(learning_rate=0.1)).result()
    s1 = a.save_weights_and_get_sampling_client("v1")  # weight_version 1
    a.forward_backward([make_datum([1, 2, 3, 4])], "cross_entropy")
    a.optim_step(types.AdamParams(learning_rate=0.1)).result()
    s1.sample(prompt=types.ModelInput.from_ints([9]), num_samples=1,
              sampling_params=types.SamplingParams(max_tokens=1)).result()
    assert _sample_ops(server, a.model_id)[-1]["pinned_version"] == 1


def test_base_model_only_is_rejected(service_client, server):
    with pytest.raises(Exception) as ei:
        service_client.create_sampling_client(base_model="fake/tiny")
    assert "base-model sampling is not supported" in str(ei.value)
    r = server.post("/api/v1/asample", {"prompt": {"tokens": [1]}, "num_samples": 1, "base_model": "fake/tiny"})
    assert r.status_code == 400 and "base-model sampling" in r.json()["error"]
    r = server.post("/api/v1/sample", {"prompts": [[1]], "num_samples": 1, "base_model": "fake/tiny"})
    assert r.status_code == 400


def test_unknown_sampler_and_model_are_404(service_client, server):
    r = server.post("/api/v1/asample", {"prompt": {"tokens": [1]}, "num_samples": 1, "sampling_session_id": "nope"})
    assert r.status_code == 404
    r = server.post("/api/v1/asample", {"prompt": {"tokens": [1]}, "num_samples": 1, "model_path": "tinker://model_gone/weights/x"})
    assert r.status_code == 404
    r = server.post("/api/v1/create_sampling_client", {"model_path": "tinker://model_gone/weights/x"})
    assert r.status_code == 404
    r = server.post("/api/v1/asample", {"prompt": {"tokens": [1]}, "num_samples": 1})
    assert r.status_code == 400


def test_sampler_of_unloaded_model_is_404(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    sa = a.save_weights_and_get_sampling_client("gone")
    rid = server.post("/api/v1/unload_model", {"model_id": a.model_id}).json()["request_id"]
    assert server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=60).status_code == 200
    r = server.post("/api/v1/asample", {"prompt": {"tokens": [1]}, "num_samples": 1,
                                        "sampling_session_id": sa._sampling_session_id})
    assert r.status_code == 404
