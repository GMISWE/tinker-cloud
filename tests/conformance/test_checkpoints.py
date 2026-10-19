"""Checkpoint listing, typed delete, and load_weights as a first request only."""
import requests
from tinker import types

from .conftest import API_KEY, make_datum


def _delete(server, model_id, checkpoint_id, **q):
    return requests.delete(f"{server.base_url}/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}",
                           headers={"X-API-Key": API_KEY}, params=q or None, timeout=30)


def test_list_and_delete(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    p = tc.save_state("c1").result().path
    sp = tc.save_weights_for_sampler("c1").result().path  # same id, other kind
    rest = service_client.create_rest_client()
    listed = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert {(c.checkpoint_id, c.checkpoint_type, c.tinker_path) for c in listed} == {
        ("weights/c1", "training", p), ("sampler_weights/c1", "sampler", sp)}
    assert all(c.size_bytes and c.size_bytes > 0 for c in listed)

    # bare id is ambiguous -> 400; the SDK's own delete sends the typed form
    assert _delete(server, tc.model_id, "c1").status_code == 400
    rest.delete_checkpoint_from_tinker_path(sp).result()
    remaining = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert [c.checkpoint_id for c in remaining] == ["weights/c1"]
    assert server.post("/api/v1/weights_info", {"tinker_path": sp}).status_code == 404
    assert server.post("/api/v1/weights_info", {"tinker_path": p}).status_code == 200
    # bytes are gone for the deleted kind only
    assert not (server.checkpoint_base / tc.model_id / "sampler_weights" / "c1").exists()
    assert (server.checkpoint_base / tc.model_id / "weights" / "c1").exists()

    assert _delete(server, tc.model_id, "c1", checkpoint_type="training").status_code == 204
    assert rest.list_checkpoints(tc.model_id).result().checkpoints == []
    assert _delete(server, tc.model_id, "weights/c1").status_code == 404


def test_load_weights_first_request_rule(service_client, server):
    src = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    src.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
    src.optim_step(types.AdamParams(learning_rate=0.5)).result()
    p = src.save_state("w").result().path

    fresh = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": p, "optimizer": False, "seq_id": 1001})
    assert r.status_code == 200, r.text
    fut = server.post("/api/v1/retrieve_future", {"request_id": r.json()["request_id"]}, timeout=60)
    assert fut.status_code == 200 and fut.json()["path"] == p, fut.text
    # the loaded weights carry: w was 0.5 after src's step
    op = fresh.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert op.metrics["fake_w"] == 0.5

    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": p, "optimizer": False, "seq_id": 1003})
    assert r.status_code == 400 and "not permitted" in r.json()["error"], r.text

    trained = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    trained.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
    r = server.post("/api/v1/load_weights", {"model_id": trained.model_id, "path": p, "optimizer": False})
    assert r.status_code == 400
    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": "tinker://nope/weights/x", "optimizer": False})
    assert r.status_code in (400, 404)
    r = server.post("/api/v1/load_weights", {"model_id": "model_missing", "path": p, "optimizer": False})
    assert r.status_code == 404


def test_load_weights_honours_the_optimizer_flag(service_client, server):
    """weights-only leaves the optimizer fresh; with-optimizer restores it; a
    checkpoint without optimizer state refuses rather than partially resumes.
    The fake backend's optimizer state is its step counter, reported as grad_norm."""
    src = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    for _ in range(2):
        src.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
        src.optim_step(types.AdamParams(learning_rate=0.5)).result()
    p = src.save_state("opt").result().path  # w = 1.0, step_count = 2

    def load(model_id, optimizer, seq_id=1001):  # clear of the SDK's own seq_ids
        r = server.post("/api/v1/load_weights", {"model_id": model_id, "path": p, "optimizer": optimizer, "seq_id": seq_id})
        assert r.status_code == 200, r.text
        return server.post("/api/v1/retrieve_future", {"request_id": r.json()["request_id"]}, timeout=60)

    weights_only = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    assert load(weights_only.model_id, False).status_code == 200
    op = weights_only.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert op.metrics["fake_w"] == 1.0 and op.metrics["grad_norm"] == 1.0  # first step of a fresh optimizer

    resumed = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    assert load(resumed.model_id, True).status_code == 200
    op = resumed.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert op.metrics["fake_w"] == 1.0 and op.metrics["grad_norm"] == 3.0  # continues from step 2

    # the SDK's two resume constructors map onto the two flags
    assert service_client.create_training_client_from_state(p).optim_step(
        types.AdamParams(learning_rate=0.0)).result().metrics["grad_norm"] == 1.0
    assert service_client.create_training_client_from_state_with_optimizer(p).optim_step(
        types.AdamParams(learning_rate=0.0)).result().metrics["grad_norm"] == 3.0

    # a checkpoint that carries no optimizer state: weights-only fine, with-optimizer refused
    import json
    state = server.checkpoint_base / src.model_id / "weights" / "opt" / "fake_state.json"
    st = json.loads(state.read_text())
    st.pop("optimizer")
    state.write_text(json.dumps(st))
    fresh = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    fut = load(fresh.model_id, True)
    assert fut.status_code == 400 and "no optimizer state" in fut.text, fut.text
    fresh2 = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    assert load(fresh2.model_id, False).status_code == 200


def test_a_save_in_flight_is_pending_not_missing(service_client, server):
    """weights_info / load_weights on a checkpoint whose save has not returned
    answer 425, never 404; the record exists before the bytes do."""
    import time
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    fut = tc.save_state("slow-one")                       # the fake writes slow-* after a delay
    p = f"tinker://{tc.model_id}/weights/slow-one"
    seen = None
    for _ in range(40):
        seen = server.post("/api/v1/weights_info", {"tinker_path": p}).status_code
        if seen == 425:
            break
        time.sleep(0.05)
    assert seen == 425
    assert fut.result().path == p
    assert server.post("/api/v1/weights_info", {"tinker_path": p}).status_code == 200


def test_kind_and_shape_are_checked_at_admission(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sp = tc.save_weights_for_sampler("k1").result().path
    fresh = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    # a sampler checkpoint is not a training resume
    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": sp, "optimizer": False, "seq_id": 1001})
    assert r.status_code == 400 and "weights required" in r.json()["error"], r.text
    # only the tinker://<model>/<kind>/<name> grammar exists
    for bad in ("/data/checkpoints/x", f"tinker://{tc.model_id}", f"tinker://{tc.model_id}/weights/a/b",
                f"tinker://{tc.model_id}/native/x"):
        assert server.post("/api/v1/weights_info", {"tinker_path": bad}).status_code == 400, bad
        r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": bad, "optimizer": False})
        assert r.status_code == 400, (bad, r.text)


def test_sampler_from_a_checkpoint_path_is_pinned_to_its_version(service_client, server):
    """A sampling client built from a sampler_weights path serves the weight
    version that checkpoint was saved at, not the live weights."""
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
    tc.optim_step(types.AdamParams(learning_rate=0.5)).result()          # version 1
    sp = tc.save_weights_for_sampler("v1").result().path
    tc.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
    tc.optim_step(types.AdamParams(learning_rate=0.5)).result()          # live is version 2
    sc = service_client.create_sampling_client(model_path=sp)
    sc.sample(prompt=types.ModelInput.from_ints([1, 2]), num_samples=1,
              sampling_params=types.SamplingParams(max_tokens=2)).result()
    pinned = [t["pinned_version"] for t in server.trace() if t["op"] == "sample" and t["model_id"] == tc.model_id]
    assert pinned and pinned[-1] == 1, pinned
