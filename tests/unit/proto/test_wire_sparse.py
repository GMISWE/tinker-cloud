"""Sparse CSR `loss_fn_inputs` on the proto path. The SDK sends a 2-D `target_tokens` or `weights`
as CSR whenever the CSR holds fewer elements than the dense tensor, so the same client call arrives in
either form depending on how many zeros the tensor holds."""
import numpy as np
import pytest

from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto import wire
from tinkercloud.training.proto.wire import WireError, parse_forward_backward_request

from tests.datums import write_csr

ROWS, COLS = 6, 4


def _mostly_zero(numpy_dtype):
    """[ROWS, COLS] with zero rows and zeros inside the others: the shape a custom loss's weights have
    (prompt rows, clipped ids)."""
    dense = np.zeros((ROWS, COLS), dtype=numpy_dtype)
    dense[2, 0], dense[2, 3], dense[4, 1], dense[5, 2] = 3, 5, 7, 2
    return dense


def _parsed(msg, name):
    request, _ = parse_forward_backward_request(msg.SerializeToString())
    return request["forward_backward_input"]["data"][0]["loss_fn_inputs"][name]


@pytest.mark.parametrize("name, numpy_dtype", [("weights", np.float32), ("target_tokens", np.int64)])
def test_the_sdks_sparse_tensor_parses_to_the_tensor_the_client_passed(name, numpy_dtype):
    """Before the decode every such request was a 422: a custom loss on [N, K] targets failed on its
    first short sequence. The request is the SDK encoder's own bytes, so a layout the two sides read
    differently (row pointers, index width, value order) shows as a wrong dense tensor."""
    sdk_req = pytest.importorskip("tinker.proto.request_conv", reason="tinker >= 0.25 not installed")
    torch = pytest.importorskip("torch")
    import tinker.types as t

    dense = torch.from_numpy(_mostly_zero(numpy_dtype))
    datum = t.Datum(model_input=t.ModelInput.from_ints(list(range(ROWS))), loss_fn_inputs={name: dense})
    sent = datum.loss_fn_inputs[name]
    assert sent.sparse_crow_indices is not None, "the SDK sent this tensor dense: the test no longer reaches the decode"
    msg = sdk_req.forward_backward_request_to_proto(t.ForwardBackwardRequest(
        model_id="m", seq_id=1, forward_backward_input=t.ForwardBackwardInput(data=[datum], loss_fn="cross_entropy")))
    got = _parsed(msg, name)
    assert got["shape"] == [ROWS, COLS]
    np.testing.assert_array_equal(np.asarray(got["data"]).reshape(ROWS, COLS), dense.numpy())


def _csr_msg(shape=(ROWS, COLS), crow=(0, 0, 0, 2, 2, 3, 4), col=(0, 3, 1, 2), values=(3.0, 5.0, 7.0, 2.0),
             value_bytes=None):
    msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
    datum = msg.data.add()
    datum.model_input.add().encoded_text.tokens = np.arange(ROWS, dtype=np.int32).tobytes()
    tensor = datum.loss_fn_inputs["weights"]
    tensor.dtype = pb.DTYPE_FLOAT32
    write_csr(tensor, shape, np.asarray(values, dtype=np.float32), crow, col)
    if value_bytes is not None:
        tensor.sparse_csr.values = value_bytes
    return msg


def test_the_hand_built_csr_of_the_invalid_cases_is_itself_valid():
    """The base of every invalid case below: if it stopped parsing, those cases would pass for the
    wrong reason."""
    got = _parsed(_csr_msg(), "weights")
    assert np.asarray(got["data"]).reshape(ROWS, COLS).tolist() == _mostly_zero(np.float32).tolist()


@pytest.mark.parametrize("change", [
    pytest.param(dict(shape=()), id="no_shape"),
    pytest.param(dict(shape=(ROWS * COLS,)), id="shape_of_rank_1"),
    pytest.param(dict(shape=(-ROWS, COLS)), id="negative_dimension"),
    pytest.param(dict(value_bytes=b"\0" * 15), id="values_cut_inside_an_item"),
    pytest.param(dict(crow=(0, 0, 0, 2, 2, 3)), id="one_row_pointer_short"),
    pytest.param(dict(crow=(1, 1, 1, 2, 2, 3, 4)), id="row_pointers_not_from_0"),
    pytest.param(dict(crow=(0, 0, 3, 2, 2, 3, 4)), id="row_pointers_decreasing"),
    pytest.param(dict(col=(0, COLS, 1, 2)), id="column_past_the_row"),
    pytest.param(dict(col=(0, -1, 1, 2)), id="column_negative"),
    pytest.param(dict(values=(3.0, 5.0, 7.0)), id="fewer_values_than_entries"),
    pytest.param(dict(col=(0, 3, 1)), id="fewer_columns_than_values"),
    pytest.param(dict(col=(3, 0, 1, 2)), id="columns_not_increasing_in_a_row"),
])
def test_a_csr_that_does_not_fit_its_shape_is_refused_by_name(change):
    """Each of these would otherwise write values to the wrong cells or raise an IndexError inside
    numpy: the client gets a 422 that names the input, never a tensor it did not send."""
    with pytest.raises(WireError, match=r"loss_fn_inputs\['weights'\]: sparse"):
        parse_forward_backward_request(_csr_msg(**change).SerializeToString())


def test_the_sparse_tensors_of_a_request_share_one_bound_on_their_dense_bytes(monkeypatch):
    """A dense tensor is as large as its bytes in the body; a CSR of a few bytes can name any shape, and
    the decode would allocate it. The bound is the request's: two tensors that each fit do not fit twice."""
    msg = _csr_msg()
    msg.data.add().CopyFrom(msg.data[0])
    monkeypatch.setattr(wire, "MAX_DECOMPRESSED_BODY_BYTES", 2 * ROWS * COLS * 4)
    parse_forward_backward_request(msg.SerializeToString())
    monkeypatch.setattr(wire, "MAX_DECOMPRESSED_BODY_BYTES", 2 * ROWS * COLS * 4 - 1)
    with pytest.raises(WireError, match="densifies past"):
        parse_forward_backward_request(msg.SerializeToString())
