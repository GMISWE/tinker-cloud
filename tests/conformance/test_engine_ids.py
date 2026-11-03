"""Every id the server mints names its engine: `model_<engine>_<hex16>`, `req_<engine>_<hex16>`.

The gateway routes a request by that name without a lookup; the one process (`single`)
is the engine named after its backend, so both modes mint the same shape.
"""
import re

ID = re.compile(r"^(?P<kind>model|req)_fake_[0-9a-f]{16}$")


def _session(server) -> str:
    r = server.post("/api/v1/create_session", {"tags": [], "user_metadata": {}, "sdk_version": "t"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _create_model(server, session_id: str, seq: int) -> dict:
    r = server.post("/api/v1/create_model", {
        "session_id": session_id, "model_seq_id": seq, "base_model": "fake/tiny",
        "lora_config": {"rank": 8}, "optimizer_config": {"type": "adamw"},
    })
    assert r.status_code == 200, r.text
    return r.json()


def _wait(server, request_id: str) -> dict:
    r = server.post(f"/api/v1/retrieve_future/{request_id}", {"request_id": request_id}, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def test_model_and_request_ids_carry_the_engine_name(server):
    sid = _session(server)
    created = _create_model(server, sid, 0)
    assert ID.match(created["model_id"]) and created["model_id"].startswith("model_"), created
    assert ID.match(created["request_id"]) and created["request_id"].startswith("req_"), created
    _wait(server, created["request_id"])
    model_id = created["model_id"]
    try:
        tokens = list(range(10, 20))
        datum = {"model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
                 "loss_fn_inputs": {"weights": {"data": [1.0] * 9, "dtype": "float32", "shape": [9]},
                                    "target_tokens": {"data": tokens[1:], "dtype": "int64", "shape": [9]}}}
        for path, body in (
            ("/api/v1/forward_backward", {"model_id": model_id, "seq_id": 1,
                                          "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"}}),
            ("/api/v1/optim_step", {"model_id": model_id, "seq_id": 2, "adam_params": {"learning_rate": 1e-4}}),
            ("/api/v1/save_weights", {"model_id": model_id, "seq_id": 3, "path": "ids-ckpt"}),
        ):
            r = server.post(path, body)
            assert r.status_code == 200, (path, r.text)
            rid = r.json()["request_id"]
            assert ID.match(rid) and rid.startswith("req_"), (path, rid)
            _wait(server, rid)
    finally:
        server.post("/api/v1/delete_model", {"model_id": model_id})


def test_sample_ids_derive_from_the_request_id(server):
    sid = _session(server)
    created = _create_model(server, sid, 0)
    _wait(server, created["request_id"])
    model_id = created["model_id"]
    try:
        r = server.post("/api/v1/save_weights_for_sampler", {"model_id": model_id, "seq_id": 1, "path": "ids-sampler"})
        assert r.status_code == 200, r.text
        _wait(server, r.json()["request_id"])
        r = server.post("/api/v1/create_sampling_session", {
            "session_id": sid, "model_path": f"tinker://{model_id}/sampler_weights/ids-sampler",
            "sampling_session_seq_id": 0,
        })
        assert r.status_code == 200, r.text
        sampler = r.json()["sampling_session_id"]
        r = server.post("/api/v1/asample", {"sampling_session_id": sampler, "prompt": {"tokens": [1, 2, 3]},
                                            "num_samples": 2, "sampling_params": {"max_tokens": 4}})
        assert r.status_code == 200, r.text
        doc = r.json()
        assert ID.match(doc["request_id"]), doc
        assert doc["sample_sequence_ids"] == [f"{doc['request_id']}-0", f"{doc['request_id']}-1"], doc
    finally:
        server.post("/api/v1/delete_model", {"model_id": model_id})
