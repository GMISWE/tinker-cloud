"""
Harness tests for the classification backend (feature 004).

GPU-free. Covers:
- BackendFactory creates MegatronBridgeBackend.
- NeMo2ClassificationDataConverter produces the evo2_classifier
  {input_ids, pool_mask, labels} batch for toy seq-cls Datums.
- Generation methods (sample / get_logprobs / update_inference_weights /
  prepare_for_generation) raise BackendError.

See specs/004-bionemo-classification/plan.md.
"""
import asyncio

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.models.requests import Datum
from tinkercloud.training.backends.factory import BackendFactory
from tinkercloud.training.backends.megatron_bridge.backend import (
    MegatronBridgeBackend,
    MegatronBridgeHandle,
)
from tinkercloud.training.backends.megatron_bridge.converter import (
    PAD_TOKEN_ID,
    NeMo2ClassificationDataConverter,
)
from tinkercloud.training.backends.objectives import (
    CLASSIFICATION_CE,
    Objective,
    is_classification,
)


def _datum(tokens, labels):
    return Datum.model_validate({
        "model_input": {"tokens": tokens},
        "loss_fn_inputs": {"labels": {"data": labels}},
    })


# --- factory ---------------------------------------------------------------

def test_factory_creates_megatron_bridge():
    assert isinstance(
        BackendFactory.create("megatron_bridge"), MegatronBridgeBackend
    )


def test_factory_unknown_lists_new_backends():
    with pytest.raises(ValueError, match="megatron_bridge"):
        BackendFactory.create("does_not_exist")


# --- objectives ------------------------------------------------------------

def test_objectives():
    assert CLASSIFICATION_CE == "classification_ce"
    assert not is_classification(Objective.LANGUAGE_MODELING)
    assert is_classification(Objective.SEQUENCE_CLASSIFICATION)
    assert is_classification("token_classification")


# --- converter -------------------------------------------------------------

def test_seq_cls_batch_shapes():
    conv = NeMo2ClassificationDataConverter()
    data = [_datum([5, 6, 7], [1]), _datum([8, 9], [0])]
    batch = conv.forward_backward_to_backend(data, CLASSIFICATION_CE, {})
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["pool_mask"].shape) == (2, 3)
    assert batch["labels"].tolist() == [1, 0]                 # [B]
    # pool_mask marks real length; sample 1 has 2 real tokens + 1 pad.
    assert batch["pool_mask"][1].tolist() == [1.0, 1.0, 0.0]
    assert batch["input_ids"][1].tolist() == [8, 9, PAD_TOKEN_ID]


def test_fixed_seq_length_pads_and_truncates():
    conv = NeMo2ClassificationDataConverter()
    data = [_datum([5, 6, 7], [1])]
    assert batch_len(conv.forward_to_backend(data, {"seq_length": 5})) == 5
    assert batch_len(conv.forward_to_backend(data, {"seq_length": 2})) == 2


def batch_len(batch):
    return batch["input_ids"].shape[1]


def test_forward_only_without_labels():
    """Forward-only inputs may carry no labels: the batch still builds."""
    conv = NeMo2ClassificationDataConverter()
    data = [Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {}})]
    batch = conv.forward_to_backend(data, {})
    assert batch["labels"].tolist() == [0]
    assert batch["pool_mask"][0].tolist() == [1.0, 1.0, 1.0]


# --- generation methods not supported --------------------------------------

_BACKENDS = [
    (MegatronBridgeBackend, MegatronBridgeHandle, "megatron_bridge"),
]


@pytest.mark.parametrize("backend_cls,handle_cls,backend_type", _BACKENDS)
def test_generation_methods_raise(backend_cls, handle_cls, backend_type):
    backend = backend_cls()
    handle = handle_cls(model_id="m", backend_type=backend_type)

    calls = {
        "sample": backend.sample(handle, "req", [1, 2], 1),
        "get_logprobs": backend.get_logprobs(handle, []),
        "update_inference_weights": backend.update_inference_weights(handle),
        "prepare_for_generation": backend.prepare_for_generation(handle),
    }
    for name, coro in calls.items():
        with pytest.raises(BackendError, match="only valid for the language_modeling objective"):
            asyncio.run(coro)
