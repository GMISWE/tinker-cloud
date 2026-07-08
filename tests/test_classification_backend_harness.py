"""
Harness tests for the classification backends (feature 004).

GPU-free. Covers:
- BackendFactory creates AutomodelBackend + MegatronBridgeBackend.
- ClassificationDataConverter produces correct {input_ids, attention_mask,
  labels} shapes for toy seq-cls ([B]) and token-cls ([B,S], -100 ignore) Datums.
- Generation methods (sample / get_logprobs / update_inference_weights /
  prepare_for_generation) raise BackendError on both backends.

See specs/004-bionemo-classification/plan.md.
"""
import asyncio

import pytest

from tinkercloud.training.backends.automodel.backend import (
    AutomodelBackend,
    AutomodelHandle,
)
from tinkercloud.training.backends.automodel.converter import (
    ClassificationDataConverter,
    IGNORE_INDEX,
)
from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.factory import BackendFactory
from tinkercloud.training.backends.megatron_bridge.backend import (
    MegatronBridgeBackend,
    MegatronBridgeHandle,
)
from tinkercloud.training.backends.megatron_bridge.converter import (
    NeMo2ClassificationDataConverter,
)
from tinkercloud.training.backends.objectives import (
    CLASSIFICATION_CE,
    Objective,
    is_classification,
)


def _datum(tokens, labels):
    return {
        "model_input": {"tokens": tokens},
        "loss_fn_inputs": {"labels": {"data": labels}},
    }


# --- factory ---------------------------------------------------------------

def test_factory_creates_automodel():
    assert isinstance(BackendFactory.create("automodel"), AutomodelBackend)


def test_factory_creates_megatron_bridge():
    assert isinstance(
        BackendFactory.create("megatron_bridge"), MegatronBridgeBackend
    )


def test_factory_unknown_lists_new_backends():
    with pytest.raises(ValueError, match="automodel|megatron_bridge"):
        BackendFactory.create("does_not_exist")


# --- objectives ------------------------------------------------------------

def test_objectives():
    assert CLASSIFICATION_CE == "classification_ce"
    assert not is_classification(Objective.LANGUAGE_MODELING)
    assert is_classification(Objective.SEQUENCE_CLASSIFICATION)
    assert is_classification("token_classification")


# --- converter -------------------------------------------------------------

@pytest.mark.parametrize(
    "converter_cls",
    [ClassificationDataConverter, NeMo2ClassificationDataConverter],
)
def test_seq_cls_batch_shapes(converter_cls):
    conv = converter_cls()
    data = [_datum([5, 6, 7], [1]), _datum([8, 9], [0])]
    batch = conv.forward_backward_to_backend(
        data, CLASSIFICATION_CE, {"objective": "sequence_classification"}
    )
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert tuple(batch["attention_mask"].shape) == (2, 3)
    assert tuple(batch["labels"].shape) == (2,)          # [B]
    assert batch["labels"].tolist() == [1, 0]
    # attention_mask marks real length; sample 1 has 2 real tokens + 1 pad.
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["input_ids"][1].tolist() == [8, 9, 0]


def test_token_cls_batch_shapes():
    conv = ClassificationDataConverter()
    data = [_datum([5, 6, 7], [1, 2, 0]), _datum([8, 9], [2, 1])]
    batch = conv.forward_backward_to_backend(
        data, CLASSIFICATION_CE, {"objective": "token_classification"}
    )
    assert tuple(batch["labels"].shape) == (2, 3)        # [B, S]
    assert batch["labels"][0].tolist() == [1, 2, 0]
    # Padded token position uses the CE ignore index.
    assert batch["labels"][1].tolist() == [2, 1, IGNORE_INDEX]


def test_empty_batch():
    conv = ClassificationDataConverter()
    batch = conv.forward_backward_to_backend([], CLASSIFICATION_CE, {})
    assert batch["input_ids"].numel() == 0
    assert batch["labels"].numel() == 0


def test_forward_only_without_labels_omits_labels():
    """Forward-only inputs may carry no labels — the batch must still build
    (input_ids/attention_mask only), not IndexError on the seq-cls path."""
    conv = ClassificationDataConverter()
    data = [
        {"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {}},
        {"model_input": {"tokens": [8, 9]}, "loss_fn_inputs": {}},
    ]
    batch = conv.forward_to_backend(data, {"objective": "sequence_classification"})
    assert "labels" not in batch
    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]


# --- generation methods not supported --------------------------------------

_BACKENDS = [
    (AutomodelBackend, AutomodelHandle, "automodel"),
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
