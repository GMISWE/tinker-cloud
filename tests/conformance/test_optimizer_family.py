"""The optimizer family is fixed at create and travels with the model and its
checkpoints: the fake serves both families, so the wire's acceptance and every
refusal are checked without a GPU."""
import pytest
from tinker import types

from .conftest import make_datum

DIMUON = {"type": "dimuon", "version": 1}


def _session(server) -> str:
    r = server.post("/api/v1/create_session", {"tags": [], "user_metadata": {}, "sdk_version": "t"})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _create(server, session_id, seq, optimizer_config):
    return server.post("/api/v1/create_model", {
        "session_id": session_id, "model_seq_id": seq, "base_model": "fake/tiny",
        "lora_config": {"rank": 4}, "optimizer_config": optimizer_config,
    })


def _step(server, model_id, body):
    return server.post("/api/v1/optim_step", {"model_id": model_id, **body})


def _future(server, request_id, timeout=60):
    return server.post("/api/v1/retrieve_future", {"request_id": request_id}, timeout=timeout)


@pytest.fixture(scope="module")
def dimuon(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=4,
                                                      optimizer=types.DimuonOptimizerConfig())


@pytest.fixture(scope="module")
def adamw(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=4)


def test_create_records_and_echoes_the_family(server, dimuon, adamw):
    info = server.post("/api/v1/get_info", {"model_id": dimuon.model_id}).json()
    assert info["optimizer_config"] == DIMUON
    assert server.post("/api/v1/get_info", {"model_id": adamw.model_id}).json()["optimizer_config"] == {"type": "adamw"}
    created = [t for t in server.trace() if t["op"] == "create_model" and t["model_id"] == dimuon.model_id]
    assert created[-1]["optimizer"] == "dimuon"


@pytest.mark.parametrize("config,detail", [
    ({"type": "dimuon", "version": 2}, "version"),
    ({"type": "dimuon"}, "version"),
    ({"type": "adamw", "version": 1}, "version"),
    ({"type": "muon"}, "muon"),
])
def test_configs_the_wire_does_not_define_are_400(server, config, detail):
    r = _create(server, _session(server), 1, config)
    assert r.status_code == 400, (r.status_code, r.text)
    assert detail in r.json()["error"]


@pytest.mark.parametrize("version", [True, "1", 1.0])
def test_a_version_that_is_not_an_integer_is_a_type_error(server, version):
    r = _create(server, _session(server), 1, {"type": "dimuon", "version": version})
    assert r.status_code == 422 and "version" in r.text, (r.status_code, r.text)


def test_a_step_of_the_other_family_is_400_naming_both(server, dimuon, adamw):
    r = _step(server, adamw.model_id, {"optimizer_params": {"type": "dimuon", "learning_rate": 1e-3}})
    assert r.status_code == 400, r.text
    assert "adamw" in r.json()["error"] and "dimuon" in r.json()["error"]
    r = _step(server, dimuon.model_id, {"adam_params": {"learning_rate": 1e-3}})
    assert r.status_code == 400, r.text
    assert "adamw" in r.json()["error"] and "dimuon" in r.json()["error"]


def test_both_or_neither_parameter_key_is_400(server, dimuon):
    r = _step(server, dimuon.model_id, {"adam_params": {"learning_rate": 1e-3},
                                        "optimizer_params": {"type": "dimuon", "learning_rate": 1e-3}})
    assert r.status_code == 400 and "exactly one" in r.json()["error"], r.text
    r = _step(server, dimuon.model_id, {})
    assert r.status_code == 400 and "exactly one" in r.json()["error"], r.text


@pytest.mark.parametrize("params,detail", [
    ({"type": "dimuon", "learning_rate": -1}, "learning_rate"),
    ({"type": "dimuon", "learning_rate": 1e-3, "beta1": 1.0}, "beta1"),
    ({"type": "dimuon", "learning_rate": 1e-3, "beta2": 0.95}, "beta2"),
    ({"type": "dimuon"}, "learning_rate"),
    ({"type": "muon", "learning_rate": 1e-3}, "muon"),
    ({"learning_rate": 1e-3}, "type"),
])
def test_dimuon_fields_outside_the_sdk_constraints_are_400(server, dimuon, params, detail):
    r = _step(server, dimuon.model_id, {"optimizer_params": params})
    assert r.status_code == 400, (r.status_code, r.text)
    assert detail in r.json()["error"], r.text


def test_a_dimuon_step_reaches_the_backend_with_its_defaults(server, dimuon):
    dimuon.forward_backward([make_datum([3, 4, 5, 6])], "cross_entropy").result()
    r = _step(server, dimuon.model_id, {"optimizer_params": {"type": "dimuon", "learning_rate": 1e-3}})
    assert r.status_code == 200, r.text
    fut = _future(server, r.json()["request_id"])
    assert fut.status_code == 200, fut.text
    steps = [t for t in server.trace() if t["op"] == "apply_optimizer_step" and t["model_id"] == dimuon.model_id]
    assert steps[-1]["optim_params"] == {"type": "dimuon", "learning_rate": 1e-3, "beta1": 0.9, "grad_clip_norm": 0.0}
    assert steps[-1]["learning_rate"] == 1e-3
    # and through the SDK's own type
    dimuon.forward_backward([make_datum([3, 4, 5, 6])], "cross_entropy").result()
    dimuon.optim_step(types.DimuonParams(learning_rate=2e-3, beta1=0.8)).result()
    steps = [t for t in server.trace() if t["op"] == "apply_optimizer_step" and t["model_id"] == dimuon.model_id]
    assert steps[-1]["optim_params"] == {"type": "dimuon", "learning_rate": 2e-3, "beta1": 0.8, "grad_clip_norm": 0.0}


def test_the_family_travels_with_the_checkpoint(service_client, server, dimuon):
    path = dimuon.save_state("fam").result().path
    wi = server.post("/api/v1/weights_info", {"tinker_path": path}).json()
    assert wi == {"base_model": "fake/tiny", "is_lora": True, "lora_rank": 4, "optimizer_config": DIMUON}

    fresh_adamw = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    r = server.post("/api/v1/load_weights", {"model_id": fresh_adamw.model_id, "path": path, "optimizer": True})
    assert r.status_code == 400, r.text
    assert "dimuon" in r.json()["error"] and "adamw" in r.json()["error"]
    r = server.post("/api/v1/load_weights", {"model_id": fresh_adamw.model_id, "path": path, "optimizer": False,
                                              "optimizer_config": DIMUON})
    assert r.status_code == 400, r.text
    assert "dimuon" in r.json()["error"] and "adamw" in r.json()["error"]
    # weights only: the documented way to switch optimizers
    r = server.post("/api/v1/load_weights", {"model_id": fresh_adamw.model_id, "path": path, "optimizer": False})
    assert r.status_code == 200, r.text
    assert _future(server, r.json()["request_id"]).status_code == 200

    fresh_dimuon = service_client.create_lora_training_client(base_model="fake/tiny", rank=4,
                                                              optimizer=types.DimuonOptimizerConfig())
    r = server.post("/api/v1/load_weights", {"model_id": fresh_dimuon.model_id, "path": path, "optimizer": True,
                                              "optimizer_config": DIMUON})
    assert r.status_code == 200, r.text
    assert _future(server, r.json()["request_id"]).status_code == 200


def test_the_sdk_resumes_with_the_optimizer(service_client, server, dimuon):
    path = dimuon.save_state("resume").result().path
    resumed = service_client.create_training_client_from_state_with_optimizer(path)
    assert server.post("/api/v1/get_info", {"model_id": resumed.model_id}).json()["optimizer_config"] == DIMUON
    resumed.forward_backward([make_datum([3, 4, 5, 6])], "cross_entropy").result()
    resumed.optim_step(types.DimuonParams(learning_rate=1e-3)).result()
