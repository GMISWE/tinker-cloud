"""Proto wire format for the two binary paths of the tinker SDK (>= 0.25.0).

Requests: `forward_backward` bodies arrive as protobuf (`Content-Type:
application/x-protobuf`, optionally `Content-Encoding: zstd`); a forward-only
pass rides the same endpoint via the `forward_only` flag. `parse_forward_backward_request`
yields the JSON-shaped dict the existing pydantic `ForwardBackwardRequest`
validates, so the routers keep one request model and one service call.

Results: the SDK retrieves sample and forward/forward_backward results with
`Accept: application/x-protobuf` and rejects JSON for those two types.
`serialize_result` is the inverse of the SDK's `proto/response_conv.py`:
tokens as little-endian int32 bytes, logprobs as float32 bytes, undefined
prompt logprobs as NaN, undefined top-k slots (prompt and sampled)
sentinel-filled, and per-datum
loss outputs concatenated into one `BatchedTensor` with int64 byte offsets.

Everything else on the API stays JSON.
"""
import base64
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from google.protobuf.message import DecodeError

from . import tinker_public_pb2 as pb


class WireError(ValueError):
    """A body the server cannot accept (malformed, or a feature it does not serve)."""


try:
    import zstandard
except ImportError:  # optional: the server then advertises proto_compress_fwdbwd=False
    zstandard = None

PROTO_CONTENT_TYPE = "application/x-protobuf"

# Operations (TaskManager names) whose completed result has a proto view.
PROTO_RESULT_OPERATIONS = frozenset({"forward", "forward_backward", "sample", "asample"})

# Bound on a decompressed body: far above any chunk the SDK sends (it caps
# chunks at ~5 MB before compression), so only a crafted body ever hits it.
MAX_DECOMPRESSED_BODY_BYTES = 1 << 30


def zstd_available() -> bool:
    return zstandard is not None


def decompress_zstd(body: bytes) -> bytes:
    """`Content-Encoding: zstd` body -> raw bytes. ASGI servers do not decode request bodies."""
    if zstandard is None:
        raise WireError("zstd request bodies are not supported on this server (zstandard not installed)")
    try:
        return zstandard.ZstdDecompressor().decompress(body, max_output_size=MAX_DECOMPRESSED_BODY_BYTES)
    except zstandard.ZstdError as e:
        raise WireError(f"zstd decompression failed: {e}") from None

# Undefined top-k slot; must match MASK_LOGPROB in the SDK's sample_response.py.
_TOPK_MASK_TOKEN_ID = 0
_TOPK_MASK_LOGPROB = -99999.0

_STOP_REASON_TO_PROTO = {"stop": pb.STOP_REASON_STOP, "length": pb.STOP_REASON_LENGTH}
_DTYPE_TO_PROTO = {"float32": pb.DTYPE_FLOAT32, "int64": pb.DTYPE_INT64}
_DTYPE_TO_NUMPY = {"float32": np.float32, "int64": np.int64}
# The SDK collapses request tensors to {float32, int64} before writing.
_PROTO_DTYPE_TO_DTYPE = {pb.DTYPE_FLOAT32: "float32", pb.DTYPE_INT64: "int64"}


# ------------------------------------------------------------------ requests

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
        loss_fn_inputs = {name: _tensor_from_proto(name, t) for name, t in datum.loss_fn_inputs.items()}
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


def _tensor_from_proto(name: str, tensor: pb.Tensor) -> Dict[str, Any]:
    dtype = _PROTO_DTYPE_TO_DTYPE.get(tensor.dtype)
    if dtype is None:
        raise WireError(f"loss_fn_inputs[{name!r}]: unsupported tensor dtype {tensor.dtype}")
    if tensor.WhichOneof("encoding") != "dense":
        raise WireError(f"loss_fn_inputs[{name!r}]: sparse tensors are not supported")
    out: Dict[str, Any] = {
        "data": np.frombuffer(tensor.dense, dtype=_DTYPE_TO_NUMPY[dtype]).tolist(),
        "dtype": dtype,
    }
    if tensor.shape:
        out["shape"] = list(tensor.shape)
    return out


# ------------------------------------------------------------------- results

def serialize_result(operation: str, result: Dict[str, Any]) -> bytes:
    """Completed result dict (already validated at the futures boundary) -> proto bytes."""
    if operation in ("sample", "asample"):
        return _serialize_sample(result)
    if operation in ("forward", "forward_backward"):
        return _serialize_forward_backward(result)
    raise ValueError(f"no proto view for operation {operation!r}")


def _serialize_sample(result: Dict[str, Any]) -> bytes:
    proto = pb.SampleResponse()
    for seq in result["sequences"]:
        stop_reason = _STOP_REASON_TO_PROTO.get(seq["stop_reason"])
        if stop_reason is None:
            raise ValueError(f"stop_reason {seq['stop_reason']!r} has no proto value")
        logprobs = seq.get("logprobs")
        msg = pb.SampledSequence(
            stop_reason=stop_reason,
            tokens=np.asarray(seq["tokens"], dtype=np.int32).tobytes(),
            logprobs=np.asarray(logprobs, dtype=np.float32).tobytes() if logprobs is not None else b"",
        )
        rows = seq.get("topk_sample_logprobs")
        if rows is not None:
            msg.topk_sampled_logprobs.CopyFrom(_topk_message(rows))
        proto.sequences.append(msg)

    prompt_logprobs = result.get("prompt_logprobs")
    if prompt_logprobs is not None:
        proto.prompt_logprobs = np.array(
            [np.nan if lp is None else lp for lp in prompt_logprobs], dtype=np.float32
        ).tobytes()

    rows = result.get("topk_prompt_logprobs")
    if rows is not None:
        proto.topk_prompt_logprobs.CopyFrom(_topk_message(rows))
    return proto.SerializeToString()


def _topk_message(rows: List[Optional[List[Tuple[int, float]]]]) -> pb.TopkLogprobs:
    """Rows of (token_id, logprob), best first, None for an undefined position
    -> dense N x k matrices, undefined slots sentinel-filled (the SDK's
    topk_to_lists reads a sentinel-led row back as None). k is not recorded in
    the result; recover it from the widest row and keep k=1 when every row is
    undefined so length still encodes."""
    k = max((len(row) for row in rows if row), default=1)
    token_ids = np.full((len(rows), k), _TOPK_MASK_TOKEN_ID, dtype=np.int32)
    logprobs = np.full((len(rows), k), _TOPK_MASK_LOGPROB, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, (token_id, logprob) in enumerate(row or ()):
            token_ids[i, j] = token_id
            logprobs[i, j] = logprob
    return pb.TopkLogprobs(
        length=len(rows), k=k, token_ids=token_ids.tobytes(), logprobs=logprobs.tobytes(),
    )


def _serialize_forward_backward(result: Dict[str, Any]) -> bytes:
    proto = pb.ForwardBackwardOutput()
    proto.loss_fn_output_type = result.get("loss_fn_output_type") or ""
    for name, value in (result.get("metrics") or {}).items():
        proto.metrics[name] = float(value)

    outputs: List[Dict[str, Dict[str, Any]]] = result.get("loss_fn_outputs") or []
    if not outputs:
        return proto.SerializeToString()

    record = proto.loss_fn_outputs.add()
    record.num_datums = len(outputs)
    # Field names in first-seen order. A datum missing a field encodes as an
    # empty slice (equal adjacent offsets), which the SDK reads back as [].
    for name in dict.fromkeys(n for datum in outputs for n in datum):
        tensors = [datum.get(name) for datum in outputs]
        dtypes = {t.get("dtype") or "float32" for t in tensors if t is not None}
        if len(dtypes) > 1:
            raise ValueError(f"field {name!r} has mixed dtypes across datums: {sorted(dtypes)}")
        dtype = dtypes.pop()
        if dtype not in _DTYPE_TO_PROTO:
            raise ValueError(f"field {name!r}: dtype {dtype!r} has no proto value")
        # One BatchedTensor carries one trailing shape: datums may be ragged
        # only in the leading dimension.
        trailing = {tuple((t.get("shape") or [len(t["data"])])[1:]) for t in tensors if t is not None}
        if len(trailing) > 1:
            raise ValueError(f"field {name!r} has mixed trailing shapes across datums: {sorted(trailing)}")
        np_dtype = _DTYPE_TO_NUMPY[dtype]
        arrays = [np.asarray(t["data"] if t is not None else [], dtype=np_dtype).ravel() for t in tensors]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([a.nbytes for a in arrays], out=offsets[1:])
        record.fields[name].CopyFrom(pb.BatchedTensor(
            data=b"".join(a.tobytes() for a in arrays),
            offsets=offsets.tobytes(),
            dtype=_DTYPE_TO_PROTO[dtype],
            trailing_shape=trailing.pop(),
        ))
    return proto.SerializeToString()
