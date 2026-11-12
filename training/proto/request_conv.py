"""Proto request bodies of the tinker SDK (>= 0.25.0) as the server's request dicts: the inverse of
the SDK's `proto/request_conv.py`.

`forward_backward` bodies arrive as protobuf (`Content-Type: application/x-protobuf`, optionally
`Content-Encoding: zstd`); a forward-only pass rides the same endpoint via the `forward_only` flag.
`parse_forward_backward_request` yields the JSON-shaped dict the existing pydantic
`ForwardBackwardRequest` validates, so the routers keep one request model and one service call; a
tensor's values stay the array its bytes decode to.
"""
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.protobuf.message import DecodeError

from . import tinker_public_pb2 as pb
from .dtypes import DTYPE_TO_NUMPY, PROTO_TO_DTYPE


class WireError(ValueError):
    """A body the server cannot accept (malformed, or a feature it does not serve)."""


# Bound on a request body once decompressed, and on the dense bytes its sparse tensors may fill. The
# SDK caps a chunk at the bytes the server hands it (TrainingBackend.FWDBWD_MAX_CHUNK_BYTES) but sends
# a larger datum alone in its chunk, so a legitimate body can be far bigger: one 128k-token datum with
# top-64 soft targets is ~98 MB.
MAX_DECOMPRESSED_BODY_BYTES = 1 << 30


def parse_forward_backward_request(body: bytes) -> Tuple[Dict[str, Any], bool]:
    """Proto `ForwardBackwardRequest` bytes -> (request dict, forward_only).

    The dict validates against `models.requests.ForwardBackwardRequest`.
    `seq_id` is non-optional on the wire and the SDK encodes an unset value as
    0, which maps back to None here so it is never mistaken for a retry of
    sequence 0. `loss_fn_config_v2` (numbers or text) wins over the legacy
    float-only map when both are present.
    """
    msg = pb.ForwardBackwardRequest()
    try:
        msg.ParseFromString(body)
    except DecodeError as e:
        raise WireError(f"malformed proto forward_backward body: {e}") from None

    data = []
    sparse_room = _SparseRoom()
    for datum in msg.data:
        chunks = []
        for chunk in datum.model_input:
            which = chunk.WhichOneof("chunk")
            if which == "encoded_text":
                chunks.append({
                    "type": "encoded_text",
                    "tokens": np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist(),
                })
            elif which == "image":
                image = {
                    "type": "image",
                    "data": base64.b64encode(chunk.image.data).decode(),
                    "format": chunk.image.format,
                }
                if chunk.image.HasField("expected_tokens"):
                    image["expected_tokens"] = chunk.image.expected_tokens
                chunks.append(image)
            else:
                raise WireError(f"unsupported model_input chunk type: {which}")
        loss_fn_inputs = {name: _tensor_from_proto(name, t, sparse_room) for name, t in datum.loss_fn_inputs.items()}
        data.append({"model_input": {"chunks": chunks}, "loss_fn_inputs": loss_fn_inputs})

    loss_fn_config: Optional[Dict[str, Any]] = None
    if msg.loss_fn_config_v2:
        loss_fn_config = {
            k: (v.text if v.WhichOneof("value") == "text" else v.number)
            for k, v in msg.loss_fn_config_v2.items()
        }
    elif msg.loss_fn_config:
        loss_fn_config = dict(msg.loss_fn_config)

    request = {
        "model_id": msg.model_id,
        "seq_id": msg.seq_id or None,
        "forward_backward_input": {"data": data, "loss_fn": msg.loss_fn, "loss_fn_config": loss_fn_config},
    }
    return request, msg.forward_only


class _SparseRoom:
    """The bytes a request's sparse tensors may still fill once dense. A dense tensor is as large as its
    bytes in the body; a sparse one of a few bytes can name any shape, so together their dense forms get
    the bound a decompressed body has."""

    def __init__(self) -> None:
        self.bytes_left = MAX_DECOMPRESSED_BODY_BYTES

    def take(self, name: str, shape: List[int], nbytes: int) -> None:
        """Raises WireError when `nbytes` more would pass the bound."""
        if nbytes > self.bytes_left:
            raise WireError(f"loss_fn_inputs[{name!r}]: sparse tensor of shape {shape} densifies past the "
                            f"{MAX_DECOMPRESSED_BODY_BYTES} bytes a request's sparse tensors may fill")
        self.bytes_left -= nbytes


def _tensor_from_proto(name: str, tensor: pb.Tensor, sparse_room: _SparseRoom) -> Dict[str, Any]:
    dtype = PROTO_TO_DTYPE.get(tensor.dtype)
    if dtype is None:
        raise WireError(f"loss_fn_inputs[{name!r}]: unsupported tensor dtype {tensor.dtype}")
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        values = np.frombuffer(tensor.dense, dtype=DTYPE_TO_NUMPY[dtype])
    elif encoding == "sparse_csr":
        values = _dense_from_csr(name, tensor, DTYPE_TO_NUMPY[dtype], sparse_room)
    else:
        raise WireError(f"loss_fn_inputs[{name!r}]: the tensor carries neither dense nor sparse data")
    out: Dict[str, Any] = {"data": values, "dtype": dtype}
    if tensor.shape:
        out["shape"] = list(tensor.shape)
    return out


def _dense_from_csr(name: str, tensor: pb.Tensor, numpy_dtype, room: _SparseRoom) -> np.ndarray:
    """A sparse CSR tensor as its dense values in row order; an entry the CSR leaves out is 0.

    The SDK sends a 2-D `target_tokens` or `weights` this way when the CSR holds fewer elements than the
    dense tensor:
    `values` and int64 `col_indices` hold the listed entries row after row, and row i owns the entries
    `crow_indices[i]:crow_indices[i + 1]`.

    Raises:
        WireError: the message is not a valid CSR for its shape, or its dense form does not fit in `room`.
    """
    def refuse(why: str) -> WireError:
        return WireError(f"loss_fn_inputs[{name!r}]: sparse tensor {why}")

    shape = list(tensor.shape)
    if len(shape) != 2 or min(shape) < 0:
        raise refuse(f"needs a 2-D shape, got {shape}")
    rows, cols = shape
    room.take(name, shape, rows * cols * np.dtype(numpy_dtype).itemsize)
    csr = tensor.sparse_csr
    try:
        values = np.frombuffer(csr.values, dtype=numpy_dtype)
        crow = np.frombuffer(csr.crow_indices, dtype=np.int64)
        col = np.frombuffer(csr.col_indices, dtype=np.int64)
    except ValueError as e:   # a buffer that is not a whole number of items
        raise refuse(f"has a truncated buffer: {e}") from None
    per_row = np.diff(crow)
    if crow.size != rows + 1 or crow[0] != 0 or (per_row < 0).any():
        raise refuse(f"needs {rows + 1} non-decreasing row pointers from 0, got {crow.size}")
    if values.size != crow[-1] or col.size != values.size:
        raise refuse(f"lists {values.size} values and {col.size} column indices for {int(crow[-1])} entries")
    row = np.repeat(np.arange(rows), per_row)
    if col.size and (col.min() < 0 or col.max() >= cols):
        raise refuse(f"has a column index outside [0, {cols})")
    if ((np.diff(col) <= 0) & (row[1:] == row[:-1])).any():
        raise refuse("has column indices that do not increase within a row")
    dense = np.zeros((rows, cols), dtype=numpy_dtype)
    dense[row, col] = values
    return dense.reshape(-1)
