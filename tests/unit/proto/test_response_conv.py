"""Proto result conversion (training/proto/response_conv.py) against the schema the SDK ships.

Messages are read directly on the vendored schema, mirroring byte-for-byte what the SDK's
response_conv reads; test_sdk_round_trip.py runs the same conversions through the SDK's own
converters when a tinker >= 0.25 is importable.
"""
import math

import numpy as np
import pytest

from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.response_conv import serialize_result


def _batched(bt):
    """SDK-side read of a BatchedTensor: per-datum flat arrays."""
    np_dtype = {pb.DTYPE_FLOAT32: np.float32, pb.DTYPE_INT64: np.int64}[bt.dtype]
    offsets = np.frombuffer(bt.offsets, dtype=np.int64)
    buf = np.frombuffer(bt.data, dtype=np_dtype)
    return [buf[int(offsets[i]) // buf.itemsize: int(offsets[i + 1]) // buf.itemsize] for i in range(len(offsets) - 1)]


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
