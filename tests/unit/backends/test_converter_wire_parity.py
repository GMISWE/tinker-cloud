"""A datum that arrived as a proto body carries its tensors as arrays; one that arrived as JSON
carries lists. Every backend's converter must read both to the same backend input: a converter that
slices, concatenates or truth-tests the values treats an array differently from a list. The last case
holds the wire name of those values, which the two arrivals share."""
import dataclasses
import sys
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from tinkercloud.training.models.requests import ForwardBackwardRequest
from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.wire import parse_forward_backward_request

TOKENS = [11, 12, 13, 14, 15, 16]
N = len(TOKENS)
_NUMPY = {"int64": np.int64, "float32": np.float32}
_PROTO = {"int64": pb.DTYPE_INT64, "float32": pb.DTYPE_FLOAT32}

CE = {"target_tokens": ("int64", [N], [12, 13, 14, 15, 16, 17]),
      "weights": ("float32", [N], [0.0, 0.0, 1.0, 1.0, 0.5, 1.0])}
CE_NK = {"target_tokens": ("int64", [N, 2], list(range(20, 20 + 2 * N))),
         "weights": ("float32", [N, 2], [0.25 * i for i in range(2 * N)])}
RL = {"target_tokens": ("int64", [N], [12, 13, 14, 15, 16, 17]),
      "logprobs": ("float32", [N], [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]),
      "advantages": ("float32", [N], [0.0, 0.0, 1.0, -1.0, 0.5, 2.0])}
LABELS = {"labels": ("int64", [1], [3])}


def _both_arrivals(inputs):
    """The same two-datum request as (datums from a proto body, datums from a JSON body)."""
    message = pb.ForwardBackwardRequest(model_id="m", loss_fn="cross_entropy")
    json_data = []
    for tokens in (TOKENS, TOKENS[::-1]):
        datum = message.data.add()
        datum.model_input.add().encoded_text.tokens = np.asarray(tokens, dtype=np.int32).tobytes()
        json_inputs = {}
        for name, (dtype, shape, values) in inputs.items():
            datum.loss_fn_inputs[name].CopyFrom(pb.Tensor(
                dtype=_PROTO[dtype], shape=shape, dense=np.asarray(values, dtype=_NUMPY[dtype]).tobytes()))
            json_inputs[name] = {"data": values, "dtype": dtype, "shape": shape}
        json_data.append({"model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                          "loss_fn_inputs": json_inputs})
    from_proto, _ = parse_forward_backward_request(message.SerializeToString())
    from_json = {"model_id": "m", "forward_backward_input": {"data": json_data, "loss_fn": "cross_entropy"}}
    proto_data = ForwardBackwardRequest.model_validate(from_proto).forward_backward_input.data
    assert isinstance(proto_data[0].loss_fn_inputs[next(iter(inputs))].values, np.ndarray), \
        "the proto body's tensors are lists again: this test no longer compares two arrivals"
    return proto_data, ForwardBackwardRequest.model_validate(from_json).forward_backward_input.data


def _assert_same(a, b, where="result"):
    """Equal keys, shapes, dtypes and values, through dicts, sequences and dataclasses."""
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape, where
        assert torch.equal(a, b), where
    elif dataclasses.is_dataclass(a):
        assert type(a) is type(b), where
        for field in dataclasses.fields(a):
            _assert_same(getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}")
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), where
        for key in a:
            _assert_same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{where}[{i}]")
    else:
        assert type(a) is type(b) and a == b, where


def _verl(data, loss_fn):
    from tinkercloud.training.backends.verl.converter import VerlDataConverter
    return VerlDataConverter().forward_backward_to_backend(data, loss_fn, None)


def _automodel(data, loss_fn):
    from tinkercloud.training.backends.automodel.converter import convert_request
    return convert_request(data, loss_fn)


def _nemo_rl(data, loss_fn):
    from tinkercloud.training.backends.nemo_rl.converter import NemoRLDataConverter
    return NemoRLDataConverter().forward_backward_to_backend(data, loss_fn, {})


def _megatron_bridge(data, loss_fn):
    from tinkercloud.training.backends.megatron_bridge.converter import NeMo2ClassificationDataConverter
    return NeMo2ClassificationDataConverter().forward_backward_to_backend(data, loss_fn, None)


def _miles(data, loss_fn):
    from tinkercloud.training.backends.miles.rollout_data import TinkerDataConverter
    return TinkerDataConverter().forward_backward_to_rollout(data, is_rl=loss_fn != "cross_entropy")


@pytest.mark.parametrize("convert,loss_fn,inputs", [
    (_verl, "cross_entropy", CE), (_verl, "cross_entropy", CE_NK), (_verl, "importance_sampling", RL),
    (_automodel, "cross_entropy", CE), (_automodel, "importance_sampling", RL),
    (_nemo_rl, "cross_entropy", CE), (_nemo_rl, "importance_sampling", RL),
    (_megatron_bridge, "cross_entropy", LABELS),
    (_miles, "cross_entropy", CE), (_miles, "importance_sampling", RL),
], ids=lambda value: value.__name__.lstrip("_") if callable(value) else None)
def test_a_proto_datum_and_a_json_datum_convert_alike(convert, loss_fn, inputs, monkeypatch):
    """A converter that truth-tests, slices or concatenates a tensor's values reads a proto body's
    array differently from a JSON body's list, or raises on it; the unit suites build every datum
    from lists and cannot see that."""
    if "nemo_rl.distributed.batched_data_dict" not in sys.modules:
        # the nemo_rl converter needs the batch's dict interface only, not NeMo RL itself
        monkeypatch.setitem(sys.modules, "nemo_rl.distributed.batched_data_dict",
                            SimpleNamespace(BatchedDataDict=dict))
    proto_data, json_data = _both_arrivals(inputs)
    _assert_same(convert(proto_data, loss_fn), convert(json_data, loss_fn))


def test_a_tensor_is_named_data_on_the_wire_only():
    """The model keeps a tensor's values under `values`, which is not a wire name: a body that sends
    `values` and no `data` is as malformed as one that sends neither."""
    body = {"model_id": "m", "forward_backward_input": {"loss_fn": "cross_entropy", "data": [{
        "model_input": {"tokens": TOKENS}, "loss_fn_inputs": {"weights": {"values": [1.0] * N, "dtype": "float32"}}}]}}
    with pytest.raises(ValidationError) as refused:
        ForwardBackwardRequest.model_validate(body)
    (error,) = refused.value.errors()
    assert error["type"] == "missing" and error["loc"][-2:] == ("weights", "data")
