"""Proto request conversion (training/proto/request_conv.py) against the schema the SDK ships.

Messages are built directly on the vendored schema, mirroring byte-for-byte what the SDK's
request_conv writes; test_sdk_round_trip.py runs the same conversions through the SDK's own
converters when a tinker >= 0.25 is importable.
"""
import numpy as np
import pytest

from tinkercloud.training.models.requests import ForwardBackwardRequest
from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.request_conv import WireError, parse_forward_backward_request


def _fb_msg(seq_id=3, loss_fn="cross_entropy", forward_only=False):
    msg = pb.ForwardBackwardRequest(model_id="model_abc", seq_id=seq_id, loss_fn=loss_fn, forward_only=forward_only)
    d = msg.data.add()
    d.model_input.add().encoded_text.tokens = np.asarray([1, 2, 3, 4], dtype=np.int32).tobytes()
    d.loss_fn_inputs["target_tokens"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_INT64, shape=[4],
                                                         dense=np.asarray([2, 3, 4, 5], dtype=np.int64).tobytes()))
    d.loss_fn_inputs["weights"].CopyFrom(pb.Tensor(dtype=pb.DTYPE_FLOAT32, shape=[4],
                                                   dense=np.asarray([1, .5, 1, 0], dtype=np.float32).tobytes()))
    return msg


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
