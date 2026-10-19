"""Every mutating route requires the API key; the training routes used to be open."""
import pytest


@pytest.mark.parametrize("path,body", [
    ("/api/v1/forward", {"model_id": "model_x", "forward_input": {"data": [], "loss_fn": "cross_entropy"}}),
    ("/api/v1/forward_backward", {"model_id": "model_x", "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"}}),
    ("/api/v1/optim_step", {"model_id": "model_x", "adam_params": {"learning_rate": 1e-4}}),
    ("/api/v1/save_weights", {"model_id": "model_x", "path": "x"}),
    ("/api/v1/asample", {"prompt": {"tokens": [1, 2]}, "num_samples": 1}),
])
def test_missing_key_is_rejected(server, path, body):
    r = server.post(path, body, key=None)
    assert r.status_code in (401, 403), (path, r.status_code, r.text)
    r = server.post(path, body, key="wrong-key")
    assert r.status_code in (401, 403), (path, r.status_code, r.text)


def test_http_error_body_shape(server):
    r = server.post("/api/v1/optim_step", {"model_id": "model_nope", "adam_params": {"learning_rate": 1e-4}})
    assert r.status_code == 404
    assert r.json()["path"] == "/api/v1/optim_step"  # custom handler is bound on the served app
