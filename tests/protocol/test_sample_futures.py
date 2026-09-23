"""Sample futures live in memory: per-session /retrieve_futures poll with cursor acks, cancel_future,
and the SDK path with sample_use_retrieve_futures on."""
from tinker import types


def _asample(server, sampler_id, seq_id):
    r = server.post("/api/v1/asample", {
        "prompt": {"tokens": [1, 2, 3]}, "num_samples": 1,
        "sampling_params": {"max_tokens": 2, "temperature": 1.0},
        "sampling_session_id": sampler_id, "seq_id": seq_id,
    })
    assert r.status_code == 200, r.text
    return r.json()["request_id"]


def _poll(server, sampler_id, prev_cursor, timeout=0.2):
    r = server.post("/api/v1/retrieve_futures", {
        "target": {"type": "sampling_session", "sampling_session_id": sampler_id, "cloned_sampler_id": 0},
        "prev_cursor": prev_cursor, "timeout": timeout,
    })
    assert r.status_code == 200, r.text
    return r.json()


def test_client_config_advertises_the_session_poller(server):
    r = server.post("/api/v1/client/config", {"sdk_version": "0.27.1"})
    assert r.json()["sample_use_retrieve_futures"] is True
    r = server.post("/api/v1/client/dynamic_config", {})
    assert r.json()["sample_cancel_enabled"] is True


def test_sdk_samples_through_the_session_poller(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    s = a.save_weights_and_get_sampling_client("s")
    params = types.SamplingParams(max_tokens=2, temperature=1.0)
    futs = [s.sample(prompt=types.ModelInput.from_ints([1, 2, 3]), num_samples=1, sampling_params=params)
            for _ in range(8)]
    assert all(len(f.result().sequences) == 1 for f in futs)
    assert s._get_futures_poller() is not None            # the SDK took the poller path


def test_session_poll_cursor_semantics(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    s = a.save_weights_and_get_sampling_client("raw")
    sid = s._sampling_session_id
    ids = [_asample(server, sid, seq) for seq in (1, 2, 3)]
    body = _poll(server, sid, 0, timeout=5)
    got = body["completions"]
    assert body["cursor"] == 3 and sorted(c["request_id"] for c in got) == sorted(ids)
    assert all(c["state"] == "finished" and c["response_payload_uncompressed_size"] > 0 for c in got)
    # replay with the same cursor (lost response): the same batch
    assert len(_poll(server, sid, 0)["completions"]) == 3
    # acknowledging 2 drops the first two; then nothing new at cursor 3
    assert len(_poll(server, sid, 2)["completions"]) == 1
    body = _poll(server, sid, 3)
    assert body["completions"] == [] and body["cursor"] == 3
    # results are fetched by request_id, and more than once (retry grace)
    for rid in ids:
        r = server.post("/api/v1/retrieve_future", {"request_id": rid})
        assert r.status_code == 200 and len(r.json()["sequences"]) == 1
        assert server.post("/api/v1/retrieve_future", {"request_id": rid}).status_code == 200


def test_seq_id_retry_is_idempotent(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    s = a.save_weights_and_get_sampling_client("idem")
    sid = s._sampling_session_id
    first = _asample(server, sid, 11)
    assert _asample(server, sid, 11) == first
    assert _poll(server, sid, 0, timeout=5)["cursor"] == 1


def test_unknown_session_and_cancel_semantics(service_client, server):
    r = server.post("/api/v1/retrieve_futures", {
        "target": {"type": "sampling_session", "sampling_session_id": "nope"}, "prev_cursor": 0})
    assert r.status_code == 404
    assert server.post("/api/v1/cancel_future", {"request_id": "nope"}).status_code == 404
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    s = a.save_weights_and_get_sampling_client("c")
    rid = _asample(server, s._sampling_session_id, 1)
    _poll(server, s._sampling_session_id, 0, timeout=5)
    assert server.post("/api/v1/cancel_future", {"request_id": rid}).status_code == 204   # terminal: no-op
    # a training future is not a sample future
    rid = server.post("/api/v1/unload_model", {"model_id": a.model_id}).json()["request_id"]
    assert server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=60).status_code == 200
    assert server.post("/api/v1/cancel_future", {"request_id": rid}).status_code == 404
