"""`target_tokens` of shape [N, K] on a backend that reads one target per
position: the pass fails by name instead of being misread. Over HTTP on the
fake backend, where a failed pass is a 400 on its future."""
import pytest

TOKENS = list(range(3, 9))


def _body(model_id, key, k):
    n = len(TOKENS) - 1
    shape = [n] if k is None else [n, k]
    count = n * (k or 1)
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": TOKENS[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": [1.0] * count, "dtype": "float32", "shape": shape},
            "target_tokens": {"data": (TOKENS[1:] * (k or 1)), "dtype": "int64", "shape": shape},
        },
    }
    return {"model_id": model_id, key: {"data": [datum], "loss_fn": "cross_entropy"}}


def _run(server, path, body):
    r = server.post(path, body)
    assert r.status_code == 200, r.text
    return server.post("/api/v1/retrieve_future", {"request_id": r.json()["request_id"]}, timeout=60)


@pytest.fixture(scope="module")
def model(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=4)


@pytest.mark.parametrize("path, key", [
    ("/api/v1/forward", "forward_input"),
    ("/api/v1/forward_backward", "forward_backward_input"),
])
def test_nk_targets_are_400_naming_the_field_the_rank_and_the_backend(server, model, path, key):
    r = _run(server, path, _body(model.model_id, key, k=2))
    assert r.status_code == 400, (r.status_code, r.text)
    error = r.json()["error"]
    assert "target_tokens of rank 2" in error and "FakeBackend" in error, error
    # the refusal is the request's, not the model's: one target per position still runs
    r = _run(server, path, _body(model.model_id, key, k=None))
    assert r.status_code == 200, (r.status_code, r.text)
    assert r.json()["loss_fn_outputs"][0]["logprobs"]["shape"] == [len(TOKENS) - 1]
