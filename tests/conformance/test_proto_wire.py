"""The proto wire paths of SDK >= 0.25 over HTTP: proto forward_backward bodies
(forward_only, zstd, seq_id 0), proto result views on Accept, the client
config handshake, and the healthz alias. Bodies are built on the vendored
schema so the suite runs whichever SDK is installed beside it; the JSON paths
(older SDKs) stay covered by the rest of the suite."""
import math

import numpy as np
import pytest

from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.wire import PROTO_CONTENT_TYPE, zstd_available

PROTO = {"Content-Type": PROTO_CONTENT_TYPE}
ACCEPT_PROTO = {"Accept": PROTO_CONTENT_TYPE}


def _logprobs(tokens):  # the fake backend's per-token logprob
    return [-(t % 7) / 7.0 for t in tokens]


def _fb_body(model_id, tokens, seq_id=0, forward_only=False, loss_fn="cross_entropy", config=None):
    msg = pb.ForwardBackwardRequest(model_id=model_id, seq_id=seq_id, loss_fn=loss_fn, forward_only=forward_only)
    for k, v in (config or {}).items():
        if isinstance(v, str):
            msg.loss_fn_config_v2[k].text = v
        else:
            msg.loss_fn_config_v2[k].number = v
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = np.asarray(tokens[:-1], dtype=np.int32).tobytes()
    n = len(tokens) - 1
    d.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_INT64, shape=[n],
                                                         dense=np.asarray(tokens[1:], dtype=np.int64).tobytes()))
    d.loss_fn_inputs["weights"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_FLOAT32, shape=[n],
                                                   dense=np.ones(n, dtype=np.float32).tobytes()))
    return msg.SerializeToString()


def _retrieve(server, rid, headers=None):
    r = server.post_raw("/api/v1/retrieve_future", b'{"request_id": "%s", "allow_metadata_only": true}' % rid.encode(),
                        {"Content-Type": "application/json", **(headers or {})}, timeout=60)
    assert r.status_code == 200, r.text
    return r


def _fb_output(r):
    assert r.headers["content-type"].startswith(PROTO_CONTENT_TYPE), r.headers
    out = pb.ForwardBackwardOutput()
    out.ParseFromString(r.content)
    return out


def _first_logprobs(out):
    bt = out.loss_fn_outputs[0].fields["logprobs"]
    offsets = np.frombuffer(bt.offsets, dtype=np.int64)
    buf = np.frombuffer(bt.data, dtype=np.float32)
    return buf[: int(offsets[1]) // 4].tolist()


@pytest.fixture
def model(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=2).model_id


def test_healthz_and_client_config(server):
    import requests
    r = requests.get(server.base_url + "/api/v1/healthz", timeout=5)
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    r = requests.post(server.base_url + "/api/v1/client/config", json={"sdk_version": "0.27.0"}, timeout=5)
    assert r.status_code == 200, r.text
    flags = r.json()
    assert flags["pjwt_auth_enabled"] is False
    assert flags["create_model_via_load_weights"] is False
    assert flags["proto_compress_fwdbwd"] is zstd_available()


def test_proto_forward_backward_and_proto_result(server, model):
    tokens = list(range(20, 30))
    r = server.post_raw("/api/v1/forward_backward", _fb_body(model, tokens, seq_id=1), PROTO)
    assert r.status_code == 200, r.text
    rid = r.json()["request_id"]
    out = _fb_output(_retrieve(server, rid, ACCEPT_PROTO))
    assert out.loss_fn_output_type == "cross_entropy"
    assert out.loss_fn_outputs[0].num_datums == 1
    np.testing.assert_allclose(_first_logprobs(out), _logprobs(tokens[:-1]), rtol=1e-6)
    assert "loss:mean" in out.metrics
    # the same future answers JSON to a client that does not accept proto
    j = _retrieve(server, rid).json()
    assert j["loss_fn_outputs"][0]["logprobs"]["data"] == _logprobs(tokens[:-1])
    assert [t["op"] for t in server.trace() if t["model_id"] == model][-1] == "forward_backward"


def test_forward_only_runs_a_forward_pass(server, model):
    tokens = list(range(5, 12))
    r = server.post_raw("/api/v1/forward_backward", _fb_body(model, tokens, seq_id=1, forward_only=True), PROTO)
    assert r.status_code == 200, r.text
    out = _fb_output(_retrieve(server, r.json()["request_id"], ACCEPT_PROTO))
    np.testing.assert_allclose(_first_logprobs(out), _logprobs(tokens[:-1]), rtol=1e-6)
    ops = [t["op"] for t in server.trace() if t["model_id"] == model]
    assert ops[-1] == "forward" and "forward_backward" not in ops


def test_seq_id_zero_is_unset_not_a_retry(server, model):
    tokens = list(range(1, 6))
    rids = {server.post_raw("/api/v1/forward_backward", _fb_body(model, tokens, seq_id=0), PROTO).json()["request_id"]
            for _ in range(2)}
    assert len(rids) == 2
    for rid in rids:
        _retrieve(server, rid, ACCEPT_PROTO)
    # a real seq_id is idempotent across formats: the proto retry returns the first request
    rid1 = server.post_raw("/api/v1/forward_backward", _fb_body(model, tokens, seq_id=7), PROTO).json()["request_id"]
    rid2 = server.post_raw("/api/v1/forward_backward", _fb_body(model, tokens, seq_id=7), PROTO).json()["request_id"]
    assert rid1 == rid2
    _retrieve(server, rid1, ACCEPT_PROTO)


def test_loss_fn_config_v2(server, model):
    tokens = list(range(1, 6))
    r = server.post_raw("/api/v1/forward_backward",
                        _fb_body(model, tokens, seq_id=1, loss_fn="ppo", config={"clip_low_threshold": 0.7}), PROTO)
    assert r.status_code == 200, r.text
    out = _fb_output(_retrieve(server, r.json()["request_id"], ACCEPT_PROTO))
    assert out.metrics["clip_low_threshold:mean"] == pytest.approx(0.7)
    r = server.post_raw("/api/v1/forward_backward",
                        _fb_body(model, tokens, seq_id=2, loss_fn="ppo", config={"clip_low_threshold": "0.7"}), PROTO)
    assert r.status_code == 400 and "must be a number" in r.text


def test_malformed_and_unsupported_proto_bodies_are_422(server, model):
    r = server.post_raw("/api/v1/forward_backward", b"\xff\xff", PROTO)
    assert r.status_code == 422, r.text
    msg = pb.ForwardBackwardRequest(model_id=model, seq_id=1, loss_fn="cross_entropy")
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = np.asarray([1, 2], dtype=np.int32).tobytes()
    d.loss_fn_inputs["weights"].dtype = pb.DTYPE_FLOAT32
    d.loss_fn_inputs["weights"].sparse_csr.values = b"\x00\x00\x00\x00"
    r = server.post_raw("/api/v1/forward_backward", msg.SerializeToString(), PROTO)
    assert r.status_code == 422 and "sparse" in r.text
    # unknown model still 404s on a well-formed proto body
    r = server.post_raw("/api/v1/forward_backward", _fb_body("model_nope", [1, 2, 3], seq_id=1), PROTO)
    assert r.status_code == 404


@pytest.mark.skipif(not zstd_available(), reason="zstandard not installed")
def test_zstd_compressed_body(server, model):
    import zstandard
    tokens = list(range(3, 9))
    body = zstandard.ZstdCompressor().compress(_fb_body(model, tokens, seq_id=1))
    r = server.post_raw("/api/v1/forward_backward", body, {**PROTO, "Content-Encoding": "zstd"})
    assert r.status_code == 200, r.text
    out = _fb_output(_retrieve(server, r.json()["request_id"], ACCEPT_PROTO))
    np.testing.assert_allclose(_first_logprobs(out), _logprobs(tokens[:-1]), rtol=1e-6)
    r = server.post_raw("/api/v1/forward_backward", b"junk", {**PROTO, "Content-Encoding": "zstd"})
    assert r.status_code == 422


def test_json_forward_backward_still_served(server, model):
    tokens = list(range(1, 6))
    r = server.post("/api/v1/forward_backward", {
        "model_id": model, "seq_id": 1,
        "forward_backward_input": {"data": [{
            "model_input": {"tokens": tokens[:-1]},
            "loss_fn_inputs": {"weights": {"data": [1.0] * 4, "dtype": "float32", "shape": [4]},
                               "target_tokens": {"data": tokens[1:], "dtype": "int64", "shape": [4]}}}],
            "loss_fn": "cross_entropy"}})
    assert r.status_code == 200, r.text
    rid = r.json()["request_id"]
    assert _retrieve(server, rid).json()["loss_fn_outputs"][0]["logprobs"]["data"] == _logprobs(tokens[:-1])
    # a JSON-submitted pass still has the proto view
    _fb_output(_retrieve(server, rid, ACCEPT_PROTO))
    r = server.post("/api/v1/forward_backward", {"model_id": model, "forward_backward_input": {"data": "x"}})
    assert r.status_code == 422


def test_sample_result_as_proto(server, model, service_client):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    path = tc.save_weights_for_sampler("s0").result().path
    r = server.post("/api/v1/asample", {
        "model_path": path, "prompt": {"tokens": [3, 4, 5, 6]}, "num_samples": 2, "prompt_logprobs": True,
        "sampling_params": {"max_tokens": 3, "seed": 1},
    })
    assert r.status_code == 200, r.text
    rid = r.json()["request_id"]
    assert len(r.json()["sample_sequence_ids"]) == 2  # fixed at submission, one per sequence
    json_out = _retrieve(server, rid).json()
    r = _retrieve(server, rid, ACCEPT_PROTO)
    assert r.headers["content-type"].startswith(PROTO_CONTENT_TYPE)
    out = pb.SampleResponse()
    out.ParseFromString(r.content)
    assert len(out.sequences) == 2
    for seq, jseq in zip(out.sequences, json_out["sequences"]):
        assert np.frombuffer(seq.tokens, dtype=np.int32).tolist() == jseq["tokens"]
        np.testing.assert_allclose(np.frombuffer(seq.logprobs, dtype=np.float32), jseq["logprobs"], rtol=1e-6)
        assert seq.stop_reason == {"stop": pb.STOP_REASON_STOP, "length": pb.STOP_REASON_LENGTH}[jseq["stop_reason"]]
    plp = np.frombuffer(out.prompt_logprobs, dtype=np.float32)
    assert math.isnan(plp[0]) and plp[1:].tolist() == pytest.approx(json_out["prompt_logprobs"][1:])
    # an operation without a proto view ignores Accept
    rid = server.post("/api/v1/optim_step", {"model_id": tc.model_id, "adam_params": {"learning_rate": 0.1}}).json()["request_id"]
    r = _retrieve(server, rid, ACCEPT_PROTO)
    assert r.headers["content-type"].startswith("application/json")
