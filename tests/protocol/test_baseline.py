"""Regression floor: the plain SDK training loop must work before any protocol change lands."""
import pytest
from tinker import types

from .conftest import make_datum


def test_train_save_unload(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=8)
    assert tc.model_id.startswith("model_")

    tokens = list(range(10, 20))
    fb = tc.forward_backward([make_datum(tokens)] * 2, "cross_entropy")
    op = tc.optim_step(types.AdamParams(learning_rate=0.5))
    fb_out = fb.result()
    op_out = op.result()

    # one logprob per model_input token, deterministic values (float32 on the proto wire)
    assert len(fb_out.loss_fn_outputs) == 2
    lp = list(fb_out.loss_fn_outputs[0]["logprobs"].data)
    assert lp == pytest.approx([-(t % 7) / 7.0 for t in tokens[:-1]], rel=1e-6)
    assert op_out.metrics["grad_norm"] == 1.0  # first optimizer step on this model

    fwd = tc.forward([make_datum(tokens)], "cross_entropy").result()
    assert list(fwd.loss_fn_outputs[0]["logprobs"].data) == lp

    path = tc.save_state("baseline").result().path
    assert path == f"tinker://{tc.model_id}/weights/baseline"

    ops = [t["op"] for t in server.trace() if t["model_id"] == tc.model_id]
    assert ops[:4] == ["create_model", "forward_backward", "apply_optimizer_step", "forward"]
    assert "save_checkpoint" in ops

    r = server.post("/api/v1/unload_model", {"model_id": tc.model_id})
    assert r.status_code == 200
    rid = r.json()["request_id"]
    r = server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=60)
    assert r.status_code == 200, r.text
    assert server.post("/api/v1/get_info", {"model_id": tc.model_id}).status_code == 404


def test_load_from_state_round_trips_weights(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=4)
    tokens = list(range(1, 8))
    tc.forward_backward([make_datum(tokens)], "cross_entropy")
    tc.optim_step(types.AdamParams(learning_rate=0.25)).result()
    path = tc.save_state("rt").result().path

    tc2 = service_client.create_training_client_from_state(path)
    assert tc2.model_id != tc.model_id
    # create, then load_weights: nothing loads at create (D9, specs/025)
    mine = [t for t in server.trace() if t["model_id"] == tc2.model_id]
    ops = [t["op"] for t in mine]
    assert "create_model" in ops and "load_checkpoint" in ops, ops
    # the loaded model continues from the saved weights: w = 0.25 * 1 pending microbatch
    op = tc2.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert op.metrics["fake_w"] == 0.25
