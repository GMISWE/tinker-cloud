"""Sparse CSR `loss_fn_inputs` over HTTP on the fake backend: the request the SDK sends for a mostly
zero 2-D tensor is served as its dense form is."""
import numpy as np
import pytest

from tinkercloud.training.proto import tinker_public_pb2 as pb

from tests.datums import write_csr

from .conftest import ACCEPT_PROTO
from .test_proto_wire import PROTO, _fb_output, _retrieve

TOKENS = list(range(3, 9))
N, K = len(TOKENS) - 1, 4
# [N, K] with zero rows and zeros inside the others
WEIGHTS = np.zeros((N, K), dtype=np.float32)
WEIGHTS[1, 0], WEIGHTS[1, 2], WEIGHTS[3, 3] = 0.5, -0.25, 1.0
TARGETS = np.zeros((N, K), dtype=np.int64)
TARGETS[0, 1], TARGETS[2, 0], TARGETS[2, 3] = 7, 5, 9


@pytest.fixture(scope="module")
def model(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=2).model_id


def _csr(tensor: pb.Tensor, dense: np.ndarray, crow=None) -> None:
    rows, cols = np.nonzero(dense)
    pointers = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=dense.shape[0]))]) if crow is None else crow
    write_csr(tensor, dense.shape, dense[rows, cols], pointers, cols)


def _body(model_id, seq_id, *, sparse_weights=False, sparse_targets=False, crow=None):
    msg = pb.ForwardBackwardRequest(model_id=model_id, seq_id=seq_id, loss_fn="cross_entropy")
    datum = msg.data.add()
    datum.model_input.add().encoded_text.tokens = np.asarray(TOKENS[:-1], dtype=np.int32).tobytes()
    targets = datum.loss_fn_inputs["target_tokens"]
    targets.dtype = pb.DTYPE_INT64
    if sparse_targets:
        _csr(targets, TARGETS)
    else:
        targets.shape.extend([N])
        targets.dense = np.asarray(TOKENS[1:], dtype=np.int64).tobytes()
    weights = datum.loss_fn_inputs["weights"]
    weights.dtype = pb.DTYPE_FLOAT32
    if sparse_weights:
        _csr(weights, WEIGHTS, crow)
    else:
        weights.shape.extend(WEIGHTS.shape)
        weights.dense = WEIGHTS.tobytes()
    return msg.SerializeToString()


def _result(server, body):
    r = server.post_raw("/api/v1/forward_backward", body, PROTO)
    assert r.status_code == 200, r.text
    return _fb_output(_retrieve(server, r.json()["request_id"], ACCEPT_PROTO))


def test_sparse_weights_are_served_as_the_dense_request_is(server, model):
    """The POST was a 422 before the decode, so a custom loss on [N, K] targets never reached a backend.
    This guards the acceptance over HTTP: the fake backend's outputs do not depend on the weights, and the
    decoded values are `tests/unit/proto/test_wire_sparse.py`'s to check."""
    dense = _result(server, _body(model, seq_id=1))
    sparse = _result(server, _body(model, seq_id=2, sparse_weights=True))
    assert sparse.loss_fn_outputs == dense.loss_fn_outputs
    assert dict(sparse.metrics) == dict(dense.metrics)


def test_sparse_targets_reach_the_rank_check_with_their_dense_shape(server, model):
    """Sparse targets are parsed like dense ones and reach the service's rank check, which refuses
    [N, K] on a backend that reads one target per position: a 400 of that check, not the wire's 422."""
    r = server.post_raw("/api/v1/forward_backward", _body(model, seq_id=3, sparse_targets=True), PROTO)
    assert r.status_code == 400, (r.status_code, r.text)
    assert f"shape [{N}, {K}]" in r.json()["error"], r.text


def test_an_invalid_csr_body_is_a_422_naming_the_input(server, model):
    """Row pointers one short of the shape: over HTTP the wire error is the client's 422, not a 500."""
    r = server.post_raw("/api/v1/forward_backward",
                        _body(model, seq_id=4, sparse_weights=True, crow=[0] * N), PROTO)
    assert r.status_code == 422, (r.status_code, r.text)
    assert "loss_fn_inputs['weights']" in r.text and "sparse" in r.text
