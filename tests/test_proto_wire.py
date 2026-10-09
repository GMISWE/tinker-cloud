"""Proto wire codec (training/proto/wire.py) against the schema the SDK ships.

Messages are built directly on the vendored schema, mirroring byte-for-byte
what the SDK's request_conv writes and what its response_conv reads;
test_proto_wire_sdk.py runs the same round trips through the SDK's own
converters when a tinker >= 0.25 is importable.
"""
import math

import numpy as np
import pytest

from tinkercloud.training.core import loss_registry
from tinkercloud.training.models.requests import ForwardBackwardRequest
from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.wire import (
    WireError, decompress_zstd, parse_forward_backward_request, serialize_result, zstd_available,
)


def _fb_msg(seq_id=3, loss_fn="cross_entropy", forward_only=False):
    msg = pb.ForwardBackwardRequest(model_id="model_abc", seq_id=seq_id, loss_fn=loss_fn, forward_only=forward_only)
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = np.asarray([1, 2, 3, 4], dtype=np.int32).tobytes()
    d.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_INT64, shape=[4],
                                                         dense=np.asarray([2, 3, 4, 5], dtype=np.int64).tobytes()))
    d.loss_fn_inputs["weights"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_FLOAT32, shape=[4],
                                                   dense=np.asarray([1, .5, 1, 0], dtype=np.float32).tobytes()))
    return msg


def _batched(bt):
    """SDK-side read of a BatchedTensor: per-datum flat arrays."""
    np_dtype = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64}[bt.dtype]
    offsets = np.frombuffer(bt.offsets, dtype=np.int64)
    buf = np.frombuffer(bt.data, dtype=np_dtype)
    return [buf[int(offsets[i]) // buf.itemsize: int(offsets[i + 1]) // buf.itemsize] for i in range(len(offsets) - 1)]


# ---------------------------------------------------------------- requests

def test_parse_request_yields_the_json_request_shape():
    req, forward_only = parse_forward_backward_request(_fb_msg().SerializeToString())
    assert forward_only is False
    parsed = ForwardBackwardRequest.model_validate(req)
    assert parsed.model_id == "model_abc" and parsed.seq_id == 3
    batch = parsed.forward_backward_input
    assert batch.loss_fn == "cross_entropy" and batch.loss_fn_config is None
    datum = batch.data[0]
    assert datum.model_input.chunks[0].tokens == [1, 2, 3, 4]
    assert datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5]
    assert datum.loss_fn_inputs["target_tokens"].dtype == "int64"
    assert datum.loss_fn_inputs["weights"].data == [1.0, 0.5, 1.0, 0.0]
    assert datum.loss_fn_inputs["weights"].shape == [4]


def test_forward_only_flag_and_unset_seq_id():
    req, forward_only = parse_forward_backward_request(_fb_msg(seq_id=0, forward_only=True).SerializeToString())
    assert forward_only is True
    assert req["seq_id"] is None  # 0 on the wire is "unset", never a retry of seq 0


def test_loss_fn_config_v2_wins_and_carries_text():
    msg = _fb_msg(loss_fn="ppo")
    msg.loss_fn_config["clip_low_threshold"] = 0.7
    msg.loss_fn_config_v2["clip_low_threshold"].number = 0.8
    msg.loss_fn_config_v2["note"].text = "x"
    req, _ = parse_forward_backward_request(msg.SerializeToString())
    assert req["forward_backward_input"]["loss_fn_config"] == {"clip_low_threshold": 0.8, "note": "x"}
    # legacy float map only
    msg = _fb_msg(loss_fn="ppo")
    msg.loss_fn_config["clip_high_threshold"] = 1.3
    req, _ = parse_forward_backward_request(msg.SerializeToString())
    assert req["forward_backward_input"]["loss_fn_config"] == {"clip_high_threshold": 1.3}


def test_registry_rejects_text_on_a_numeric_key():
    with pytest.raises(ValueError, match="must be a number"):
        loss_registry.validate("ppo", {"clip_low_threshold": "0.8"})
    loss_registry.validate("ppo", {"clip_low_threshold": 0.8})


def test_image_chunk_is_base64_json():
    msg = _fb_msg()
    c = msg.data[0].model_input.add()
    c.image.data, c.image.format, c.image.expected_tokens = b"\x89PNG", "png", 7
    req, _ = parse_forward_backward_request(msg.SerializeToString())
    chunk = req["forward_backward_input"]["data"][0]["model_input"]["chunks"][1]
    assert chunk == {"type": "image", "data": "iVBORw==", "format": "png", "expected_tokens": 7}


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.data[0].model_input.add().dmel.__setattr__("dmel", b"x"), "chunk type"),
    (lambda m: m.data[0].loss_fn_inputs["weights"].sparse_csr.__setattr__("values", b"\x00\x00\x00\x00"), "sparse"),
    (lambda m: m.data[0].loss_fn_inputs["weights"].__setattr__("dtype", pb.DTYPE_BFLOAT16), "dtype"),
])
def test_unsupported_bodies_are_wire_errors(mutate, match):
    msg = _fb_msg()
    mutate(msg)
    with pytest.raises(WireError, match=match):
        parse_forward_backward_request(msg.SerializeToString())


def test_malformed_bytes_are_a_wire_error():
    with pytest.raises(WireError, match="malformed"):
        parse_forward_backward_request(b"\xff\xff\xff")


@pytest.mark.skipif(not zstd_available(), reason="zstandard not installed")
def test_zstd_round_trip_and_garbage():
    import zstandard
    raw = _fb_msg().SerializeToString()
    assert decompress_zstd(zstandard.ZstdCompressor().compress(raw)) == raw
    with pytest.raises(WireError, match="zstd"):
        decompress_zstd(b"not zstd")


# ----------------------------------------------------------------- results

def test_forward_backward_result_ragged_datums():
    result = {
        "loss_fn_output_type": "cross_entropy",
        "loss_fn_outputs": [
            {"logprobs": {"data": [-0.1, -0.2, -0.3], "shape": [3], "dtype": "float32"},
             "ids": {"data": [5, 6, 7], "shape": [3], "dtype": "int64"}},
            {"logprobs": {"data": [-0.4], "shape": [1], "dtype": "float32"},
             "ids": {"data": [8], "shape": [1], "dtype": "int64"}},
        ],
        "metrics": {"loss:mean": 0.25},
    }
    out = pb.ForwardBackwardOutput()
    out.ParseFromString(serialize_result("forward_backward", result))
    assert out.loss_fn_output_type == "cross_entropy" and dict(out.metrics) == {"loss:mean": 0.25}
    (record,) = out.loss_fn_outputs
    assert record.num_datums == 2
    lp = _batched(record.fields["logprobs"])
    np.testing.assert_allclose(lp[0], [-0.1, -0.2, -0.3], rtol=1e-6)
    np.testing.assert_allclose(lp[1], [-0.4], rtol=1e-6)
    ids = _batched(record.fields["ids"])
    assert ids[0].tolist() == [5, 6, 7] and ids[1].tolist() == [8]
    assert record.fields["ids"].dtype == pb.DTYPE_INT64
    assert list(record.fields["logprobs"].trailing_shape) == []


def test_forward_backward_result_empty_and_missing_field():
    out = pb.ForwardBackwardOutput()
    out.ParseFromString(serialize_result("forward", {"loss_fn_output_type": "x", "loss_fn_outputs": [], "metrics": {}}))
    assert not out.loss_fn_outputs
    result = {"loss_fn_output_type": "x", "metrics": {},
              "loss_fn_outputs": [{"logprobs": {"data": [-1.0], "shape": [1], "dtype": "float32"}}, {}]}
    out.ParseFromString(serialize_result("forward_backward", result))
    lp = _batched(out.loss_fn_outputs[0].fields["logprobs"])
    assert lp[0].tolist() == [-1.0] and lp[1].tolist() == []


def test_forward_backward_result_mixed_shapes_rejected():
    result = {"loss_fn_output_type": "x", "metrics": {}, "loss_fn_outputs": [
        {"f": {"data": [1, 2], "shape": [1, 2], "dtype": "float32"}},
        {"f": {"data": [1, 2, 3], "shape": [1, 3], "dtype": "float32"}}]}
    with pytest.raises(ValueError, match="trailing shapes"):
        serialize_result("forward_backward", result)


def test_sample_result_sequences_prompt_logprobs_and_topk():
    result = {
        "sequences": [{"stop_reason": "stop", "tokens": [1, 2, 3], "logprobs": [-0.5, -1.25, -2.0], "text": None},
                      {"stop_reason": "length", "tokens": [7], "logprobs": None}],
        "prompt_logprobs": [None, -0.25],
        "topk_prompt_logprobs": [None, [(11, -0.1), (12, -0.2)], [(13, -0.3)]],
        "weight_version": 3,
    }
    out = pb.SampleResponse()
    out.ParseFromString(serialize_result("asample", result))
    s0, s1 = out.sequences
    assert s0.stop_reason == pb.STOP_REASON_STOP and s1.stop_reason == pb.STOP_REASON_LENGTH
    assert np.frombuffer(s0.tokens, dtype=np.int32).tolist() == [1, 2, 3]
    assert np.frombuffer(s0.logprobs, dtype=np.float32).tolist() == [-0.5, -1.25, -2.0]
    assert np.frombuffer(s1.tokens, dtype=np.int32).tolist() == [7] and s1.logprobs == b""
    plp = np.frombuffer(out.prompt_logprobs, dtype=np.float32)
    assert math.isnan(plp[0]) and plp[1] == np.float32(-0.25)
    topk = out.topk_prompt_logprobs
    assert (topk.length, topk.k) == (3, 2)
    ids = np.frombuffer(topk.token_ids, dtype=np.int32).reshape(3, 2)
    lps = np.frombuffer(topk.logprobs, dtype=np.float32).reshape(3, 2)
    assert ids.tolist() == [[0, 0], [11, 12], [13, 0]]
    assert lps[0].tolist() == [-99999.0, -99999.0] and lps[2, 1] == -99999.0


def test_sample_result_topk_sampled_logprobs_per_sequence():
    result = {"sequences": [
        {"stop_reason": "length", "tokens": [1, 2, 3], "logprobs": [-0.5, -1.0, -2.0],
         "topk_sample_logprobs": [[(1, -0.5), (9, -0.7)], None, [(3, -2.0)]]},
        {"stop_reason": "stop", "tokens": [4], "logprobs": [-0.1]}]}
    out = pb.SampleResponse()
    out.ParseFromString(serialize_result("asample", result))
    s0, s1 = out.sequences
    assert s0.HasField("topk_sampled_logprobs") and not s1.HasField("topk_sampled_logprobs")
    topk = s0.topk_sampled_logprobs
    assert (topk.length, topk.k) == (3, 2)
    ids = np.frombuffer(topk.token_ids, dtype=np.int32).reshape(3, 2)
    lps = np.frombuffer(topk.logprobs, dtype=np.float32).reshape(3, 2)
    assert ids.tolist() == [[1, 9], [0, 0], [3, 0]]          # None row and short row sentinel-filled
    assert lps[1].tolist() == [-99999.0, -99999.0] and lps[2].tolist() == [np.float32(-2.0), -99999.0]
    assert not out.HasField("topk_prompt_logprobs")


def test_sample_result_without_optional_fields():
    out = pb.SampleResponse()
    out.ParseFromString(serialize_result("sample", {"sequences": [{"stop_reason": "stop", "tokens": [], "logprobs": []}]}))
    assert out.prompt_logprobs == b"" and not out.HasField("topk_prompt_logprobs")


def test_no_proto_view_for_other_operations():
    with pytest.raises(ValueError, match="no proto view"):
        serialize_result("optim_step", {})
