"""Create-via-load: SDK >= 0.38 creates a client from a checkpoint with one
`load_weights` addressed by session_id + model_seq_id, and derives the model id
`{session_id}:train:{model_seq_id}` itself. The server creates that model with the
checkpoint's base model and LoRA config, then loads it."""
import pytest
from tinker import types

from .conftest import make_datum


def _session(server) -> str:
    r = server.post("/api/v1/create_session", {"tags": [], "user_metadata": {}, "sdk_version": "t"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _future(server, request_id, timeout=60):
    return server.post("/api/v1/retrieve_future", {"request_id": request_id}, timeout=timeout)


@pytest.fixture(scope="module")
def checkpoint(service_client):
    source = service_client.create_lora_training_client(base_model="fake/tiny", rank=8)
    source.forward_backward([make_datum([3, 4, 5, 6])], "cross_entropy").result()
    source.optim_step(types.AdamParams(learning_rate=1e-3)).result()
    return source.save_state("cvl").result().path


def _info(server, model_id):
    r = server.post("/api/v1/get_info", {"model_id": model_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_resume_with_optimizer_creates_the_derived_id(service_client, server, checkpoint):
    resumed = service_client.create_training_client_from_state_with_optimizer(checkpoint)
    session_id = service_client.holder.get_session_id()
    assert resumed.model_id.startswith(f"{session_id}:train:")
    info = _info(server, resumed.model_id)
    assert info["model_data"]["model_name"] == "fake/tiny"
    assert info["lora_rank"] == 8
    assert info["optimizer_config"] == {"type": "adamw"}
    resumed.forward_backward([make_datum([3, 4, 5, 6])], "cross_entropy").result()
    resumed.optim_step(types.AdamParams(learning_rate=1e-3)).result()


def test_weights_only_takes_the_declared_family(service_client, server, checkpoint):
    plain = service_client.create_training_client_from_state(checkpoint)
    assert _info(server, plain.model_id)["optimizer_config"] == {"type": "adamw"}
    dimuon = service_client.create_training_client_from_state(checkpoint, optimizer=types.DimuonOptimizerConfig())
    assert _info(server, dimuon.model_id)["optimizer_config"] == {"type": "dimuon", "version": 1}
    assert plain.model_id != dimuon.model_id


def test_restored_state_refuses_another_family(server, checkpoint):
    r = server.post("/api/v1/load_weights", {"session_id": _session(server), "model_seq_id": 0, "path": checkpoint,
                                              "optimizer": True, "optimizer_config": {"type": "dimuon", "version": 1}})
    assert r.status_code == 400, r.text
    assert "dimuon" in r.text and "adamw" in r.text


@pytest.mark.parametrize("extra, status, detail", [
    ({"session_id": "no-such-session", "model_seq_id": 0}, 404, "no-such-session"),
    ({"model_id": "m", "session_id": "s", "model_seq_id": 0}, 400, "not both"),
    ({"model_seq_id": 0}, 400, "needs both"),
    ({}, 400, "not both or neither"),
    ({"session_id": None, "model_seq_id": 0, "base_model": "other/base"}, 400, "other/base"),
    ({"session_id": None, "model_seq_id": 0, "weights_access_token": "t"}, 400, "account"),
])
def test_malformed_create_via_load_is_refused(server, checkpoint, extra, status, detail):
    body = {"path": checkpoint, "optimizer": False, **extra}
    if "session_id" in body and body["session_id"] is None:
        body["session_id"] = _session(server)
    r = server.post("/api/v1/load_weights", body)
    assert r.status_code == status, r.text
    assert detail in r.text


def test_the_new_model_steps_and_saves_under_its_id(server, checkpoint):
    session_id = _session(server)
    r = server.post("/api/v1/load_weights", {"session_id": session_id, "model_seq_id": 3,
                                              "path": checkpoint, "optimizer": True})
    assert r.status_code == 200, r.text
    model_id = f"{session_id}:train:3"
    assert r.json()["model_id"] == model_id
    assert _future(server, r.json()["request_id"]).status_code == 200
    r = server.post("/api/v1/save_weights", {"model_id": model_id, "path": "again"})
    assert r.status_code == 200, r.text
    path = _future(server, r.json()["request_id"]).json()["path"]
    assert path.startswith(f"tinker://{model_id}/")
    # the same session and seq again: the model exists
    r = server.post("/api/v1/load_weights", {"session_id": session_id, "model_seq_id": 3,
                                              "path": checkpoint, "optimizer": False})
    assert r.status_code == 400, r.text
    assert "already exists" in r.text


def test_a_retry_of_a_failed_create_via_load_leaves_no_model_in_the_session(server, checkpoint):
    """The first create fails and the gateway drops the model it recorded. The SDK then resends the load:
    the gateway records the model again, and the engine, answering with the failed future, has it dropped
    again. Another load naming the model is refused for the failed create."""
    import requests
    from .conftest import API_KEY
    session_id = _session(server)
    model_id = f"{session_id}:train:0"
    (server.gate / f"{model_id}.fail").touch()

    def runs():
        r = requests.get(f"{server.base_url}/api/v1/sessions/{session_id}", headers={"X-API-Key": API_KEY}, timeout=30)
        return r.json()["training_run_ids"]

    load = {"session_id": session_id, "model_seq_id": 0, "path": checkpoint, "optimizer": False}
    first = server.post("/api/v1/load_weights", load)
    assert first.status_code == 200 and first.json()["model_id"] == model_id, first.text
    assert _future(server, first.json()["request_id"]).status_code == 400
    assert runs() == []
    retry = server.post("/api/v1/load_weights", load)
    assert retry.status_code == 200 and retry.json() == first.json(), retry.text
    assert runs() == []
    r = server.post("/api/v1/load_weights", {**load, "optimizer": True})
    assert r.status_code == 400 and "failed" in r.text and "already exists" not in r.text, r.text
    assert runs() == []


def test_a_refused_create_via_load_leaves_no_model_in_the_session(server, checkpoint):
    import requests
    from .conftest import API_KEY
    session_id = _session(server)
    r = server.post("/api/v1/load_weights", {"session_id": session_id, "model_seq_id": 0, "path": checkpoint,
                                              "optimizer": False, "base_model": "other/base"})
    assert r.status_code == 400, r.text
    s = requests.get(f"{server.base_url}/api/v1/sessions/{session_id}", headers={"X-API-Key": API_KEY}, timeout=30)
    assert s.json()["training_run_ids"] == []
