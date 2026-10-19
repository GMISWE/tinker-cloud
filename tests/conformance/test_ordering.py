"""seq_id idempotency and per-model program order over the HTTP protocol."""
import concurrent.futures

from tinker import types

from .conftest import make_datum


def _fb(server, model_id, seq_id, tokens):
    return server.post("/api/v1/forward_backward", {
        "model_id": model_id, "seq_id": seq_id,
        "forward_backward_input": {"data": [{
            "model_input": {"tokens": tokens[:-1]},
            "loss_fn_inputs": {"weights": {"data": [1.0] * (len(tokens) - 1), "dtype": "float32", "shape": [len(tokens) - 1]},
                               "target_tokens": {"data": tokens[1:], "dtype": "int64", "shape": [len(tokens) - 1]}}}],
            "loss_fn": "cross_entropy"}})


def _optim(server, model_id, seq_id, lr=1.0):
    return server.post("/api/v1/optim_step", {"model_id": model_id, "seq_id": seq_id, "adam_params": {"learning_rate": lr}})


def _wait(server, rid):
    r = server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


def test_retry_with_same_seq_id_is_idempotent(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    # raw seq_ids live above the SDK client's own counter (which starts at 1)
    r1 = _fb(server, tc.model_id, 1001, [1, 2, 3, 4]); r2 = _fb(server, tc.model_id, 1001, [1, 2, 3, 4])
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["request_id"] == r2.json()["request_id"]
    _wait(server, r1.json()["request_id"])
    fbs = [t for t in server.trace() if t["op"] == "forward_backward" and t["model_id"] == tc.model_id]
    assert len(fbs) == 1  # the retry ran nothing
    # a reused seq_id with a different payload / operation is a client error (400: the SDK would retry a 409)
    r = _fb(server, tc.model_id, 1001, [9, 9, 9, 9])
    assert r.status_code == 400 and "sequence number 1001 was reused" in r.json()["error"], r.text
    assert _optim(server, tc.model_id, 1001).status_code == 400
    # the SDK's own retries reuse seq_id transparently
    op = tc.optim_step(types.AdamParams(learning_rate=0.5)).result()
    assert op.metrics["fake_w"] == 0.5  # exactly one pending microbatch was applied


def test_program_order_is_kept_under_concurrent_submission(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    m = tc.model_id
    # fire fb(1) optim(2) fb(3) optim(4) fb(5) save(6) without waiting for any result
    calls = [lambda: _fb(server, m, 1, [1, 2, 3]), lambda: _optim(server, m, 2, 1.0),
             lambda: _fb(server, m, 3, [4, 5, 6]), lambda: _fb(server, m, 4, [7, 8, 9]), lambda: _optim(server, m, 5, 1.0),
             lambda: server.post("/api/v1/save_weights", {"model_id": m, "path": "ord", "seq_id": 6})]
    rids = [c().json()["request_id"] for c in calls]
    results = [_wait(server, rid) for rid in rids]
    # each optimizer step saw exactly the microbatches submitted before it: w = 1*1, then 1*1 + 1*2
    assert results[1]["metrics"]["fake_w"] == 1.0
    assert results[4]["metrics"]["fake_w"] == 3.0
    ops = [t["op"] for t in server.trace() if t["model_id"] == m and t["op"] != "create_model"]
    assert ops == ["forward_backward", "apply_optimizer_step", "forward_backward", "forward_backward",
                   "apply_optimizer_step", "save_checkpoint"]


def test_parallel_clients_do_not_interleave_each_others_steps(service_client, server):
    a = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    b = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)

    def loop(tc, lr):
        for i in range(3):
            tc.forward_backward([make_datum([1, 2, 3, 4])], "cross_entropy")
            tc.optim_step(types.AdamParams(learning_rate=lr)).result()
        return tc.optim_step(types.AdamParams(learning_rate=0.0)).result().metrics["fake_w"]

    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        wa, wb = ex.map(lambda args: loop(*args), [(a, 1.0), (b, 10.0)])
    assert wa == 3.0 and wb == 30.0
