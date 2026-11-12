"""Completed results as the proto messages the tinker SDK (>= 0.25.0) reads: the inverse of the
SDK's `proto/response_conv.py`.

The SDK retrieves sample and forward/forward_backward results with `Accept: application/x-protobuf`
and rejects JSON for those two types. `serialize_result` writes tokens as little-endian int32 bytes,
logprobs as float32 bytes, undefined prompt logprobs as NaN, undefined top-k slots (prompt and
sampled) sentinel-filled, and per-datum loss outputs concatenated into one `BatchedTensor` with int64
byte offsets. `stored_forward_backward_result` and `forward_backward_json` are the form a futures row
keeps such a result in, and its way back to JSON.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.array_values import is_array
from ..models.tensor_values import DEFAULT_TENSOR_DTYPE, batched_layout
from . import tinker_public_pb2 as pb
from .dtypes import DTYPE_TO_NUMPY, DTYPE_TO_PROTO, PROTO_TO_DTYPE

# Operations (TaskManager names) whose completed result has a proto view.
FORWARD_BACKWARD_OPERATIONS = frozenset({"forward", "forward_backward"})
PROTO_RESULT_OPERATIONS = FORWARD_BACKWARD_OPERATIONS | {"sample", "asample"}

# Undefined top-k slot; must match MASK_LOGPROB in the SDK's types/topk_logprobs.py.
_TOPK_MASK_TOKEN_ID = 0
_TOPK_MASK_LOGPROB = -99999.0

_STOP_REASON_TO_PROTO = {"stop": pb.STOP_REASON_STOP, "length": pb.STOP_REASON_LENGTH}


def serialize_result(operation: str, result: Dict[str, Any]) -> bytes:
    """Completed result dict (already validated at the futures boundary) -> proto bytes."""
    if operation in ("sample", "asample"):
        return _serialize_sample(result)
    if operation in FORWARD_BACKWARD_OPERATIONS:
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


def stored_forward_backward_result(result: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    """A validated forward or forward_backward result as a futures row keeps it: (the result, its
    proto view). A per-datum tensor the backend returned as an array is in the proto view only: in
    the result it keeps its dtype and shape and has no `data`. A tensor returned as a list stays
    whole, with the backend's own numbers. `forward_backward_json` is the inverse."""
    result_proto = _serialize_forward_backward(result)
    outputs = [{name: {key: value for key, value in tensor.items() if key != "data"} if is_array(tensor["data"]) else tensor
                for name, tensor in datum.items()} for datum in result["loss_fn_outputs"]]
    return {**result, "loss_fn_outputs": outputs}, result_proto


def forward_backward_json(result: Dict[str, Any], result_proto: bytes) -> Dict[str, Any]:
    """A stored forward or forward_backward result with every tensor's `data` in place, as lists: a
    tensor stored without it reads its values from `result_proto`."""
    outputs = result["loss_fn_outputs"]
    if all("data" in tensor for datum in outputs for tensor in datum.values()):
        return result
    proto = pb.ForwardBackwardOutput()
    proto.ParseFromString(result_proto)
    (record,) = proto.loss_fn_outputs   # `_serialize_forward_backward` writes one record for all datums
    filled = []
    for index, datum in enumerate(outputs):
        filled.append({})
        for name, tensor in datum.items():
            if "data" not in tensor:
                batched = record.fields[name]
                start, end = np.frombuffer(batched.offsets, dtype=np.int64)[index:index + 2]
                values = np.frombuffer(batched.data[start:end], dtype=DTYPE_TO_NUMPY[PROTO_TO_DTYPE[batched.dtype]])
                tensor = {**tensor, "data": values.tolist()}
            filled[-1][name] = tensor
    return {**result, "loss_fn_outputs": filled}


def _serialize_forward_backward(result: Dict[str, Any]) -> bytes:
    proto = pb.ForwardBackwardOutput()
    proto.loss_fn_output_type = result["loss_fn_output_type"]
    for name, value in (result.get("metrics") or {}).items():
        proto.metrics[name] = float(value)

    outputs: List[Dict[str, Dict[str, Any]]] = result["loss_fn_outputs"]
    if not outputs:
        return proto.SerializeToString()

    record = proto.loss_fn_outputs.add()
    record.num_datums = len(outputs)
    # Field names in first-seen order. A datum missing a field encodes as an
    # empty slice (equal adjacent offsets), which the SDK reads back as [].
    for name in dict.fromkeys(n for datum in outputs for n in datum):
        tensors = [datum.get(name) for datum in outputs]
        # a validated result is dumped without a dtype its backend did not state
        dtype, trailing = batched_layout(name, ((t.get("dtype") or DEFAULT_TENSOR_DTYPE, t["shape"])
                                                for t in tensors if t is not None))
        np_dtype = DTYPE_TO_NUMPY[dtype]
        arrays = [np.asarray(t["data"] if t is not None else [], dtype=np_dtype).ravel() for t in tensors]
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([a.nbytes for a in arrays], out=offsets[1:])
        record.fields[name].CopyFrom(pb.BatchedTensor(
            data=b"".join(a.tobytes() for a in arrays),
            offsets=offsets.tobytes(),
            dtype=DTYPE_TO_PROTO[dtype],
            trailing_shape=trailing,
        ))
    return proto.SerializeToString()
