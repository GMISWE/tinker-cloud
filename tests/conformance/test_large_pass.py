"""A forward_backward request of millions of tensor values over HTTP: a retry is recognised by every
value of the request, however its body is written, and a JSON client reads the result's tensors."""
import numpy as np
import pytest

from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto import PROTO_CONTENT_TYPE

from .conftest import ACCEPT_PROTO

VALUES = 2_000_000
PROTO = {"Content-Type": PROTO_CONTENT_TYPE}


@pytest.fixture
def model(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=2).model_id


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    return bytes(out + bytes([value]))


def _body(model_id: str, weights: np.ndarray, seq_id: int, weights_first: bool) -> bytes:
    """The request's proto bytes. The datum's two loss inputs are map entries, which a writer may
    emit in either order; `weights_first` picks the order this body has."""
    inputs = {"target_tokens": pb.Tensor(dtype=pb.DTYPE_INT64, shape=[VALUES], dense=np.ones(VALUES, dtype=np.int64).tobytes()),
              "weights": pb.Tensor(dtype=pb.DTYPE_FLOAT32, shape=[VALUES], dense=weights.tobytes())}
    tokens = pb.Datum()
    tokens.model_input.add().encoded_text.tokens = (np.arange(VALUES, dtype=np.int32) % 1000).tobytes()
    # a message is the concatenation of its fields: one single-entry datum per input, in the wanted order
    datum = tokens.SerializeToString() + b"".join(
        pb.Datum(loss_fn_inputs={name: inputs[name]}).SerializeToString() for name in sorted(inputs, reverse=weights_first))
    data = pb.ForwardBackwardRequest.DESCRIPTOR.fields_by_name["data"]
    head = pb.ForwardBackwardRequest(model_id=model_id, seq_id=seq_id, loss_fn="cross_entropy").SerializeToString()
    return head + _varint(data.number << 3 | 2) + _varint(len(datum)) + datum


def _retrieve(server, request_id: str, accept: dict):
    reply = server.post_raw("/api/v1/retrieve_future", b'{"request_id": "%s"}' % request_id.encode(),
                            {"Content-Type": "application/json", **accept}, timeout=120)
    assert reply.status_code == 200, reply.text
    return reply


def test_a_large_request_is_matched_by_its_whole_body_and_read_back_as_json(server, model):
    """Two bodies that differ in one of 2,000,000 weights share a seq_id: a digest that does not
    cover every value files the second as a retry of the first. The SDK builds a resend's body
    anew, and a proto map has no fixed order on the wire: a digest of the body's bytes refuses a
    resend whose inputs are written in another order. A JSON client's read of the result must
    carry the logprobs the proto view carries: the two views are built apart."""
    weights = np.ones(VALUES, dtype=np.float32)
    body = _body(model, weights, 7001, weights_first=False)
    first = server.post_raw("/api/v1/forward_backward", body, PROTO, timeout=120)
    assert first.status_code == 200, first.text
    resend = _body(model, weights, 7001, weights_first=True)
    assert resend != body, "both bodies are the same bytes: the map order is not exercised"
    again = server.post_raw("/api/v1/forward_backward", resend, PROTO, timeout=120)
    assert again.status_code == 200 and again.json()["request_id"] == first.json()["request_id"], again.text
    weights[VALUES // 2] = 0.5
    other = server.post_raw("/api/v1/forward_backward", _body(model, weights, 7001, weights_first=False), PROTO, timeout=120)
    assert other.status_code == 400 and "sequence number 7001 was reused" in other.json()["error"], other.text

    request_id = first.json()["request_id"]
    as_json = _retrieve(server, request_id, {}).json()
    as_proto = pb.ForwardBackwardOutput()
    as_proto.ParseFromString(_retrieve(server, request_id, ACCEPT_PROTO).content)
    sent = np.frombuffer(as_proto.loss_fn_outputs[0].fields["logprobs"].data, dtype=np.float32)
    logprobs = as_json["loss_fn_outputs"][0]["logprobs"]
    assert logprobs["shape"] == [VALUES] and sent.size == VALUES
    assert np.array_equal(np.asarray(logprobs["data"], dtype=np.float32), sent)
    assert as_json["loss_fn_output_type"] == as_proto.loss_fn_output_type
