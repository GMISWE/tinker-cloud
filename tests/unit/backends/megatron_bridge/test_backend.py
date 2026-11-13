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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    batch = conv.forward_backward_to_backend(data, CLASSIFICATION_CE, {"num_labels": 2})
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


@pytest.mark.parametrize("labels", [
    None, [], [1, 0], [1.9], [True], ["1"], [[1]], [None], [float("nan")], [-1], [2], [7],
])
def test_training_refuses_a_label_it_would_have_to_invent_cut_or_guess(labels):
    """The label is the training target. Missing, extra, fractional, non-numeric and out-of-range classes are
    refused with the datum's index; a 2-class model takes 0 and 1 only."""
    conv = NeMo2ClassificationDataConverter()
    inputs = {} if labels is None else {"labels": {"data": labels}}
    bad = Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": inputs})
    with pytest.raises(ValueError, match="datum 1: "):
        conv.forward_backward_to_backend([_datum([5, 6], [1]), bad], CLASSIFICATION_CE, {"num_labels": 2})


@pytest.mark.parametrize("labels, expected", [([1], 1), ([0], 0), ([1.0], 1)])
def test_training_builds_the_batch_from_the_checked_classes(labels, expected):
    conv = NeMo2ClassificationDataConverter()
    batch = conv.forward_backward_to_backend([_datum([5, 6, 7], labels), _datum([8], [0])], CLASSIFICATION_CE,
                                             {"num_labels": 2})
    assert batch["labels"].tolist() == [expected, 0]


def test_the_backend_gives_the_converter_the_models_class_count(monkeypatch):
    """forward_backward passes handle.num_labels, so an out-of-range class never reaches the worker."""
    from tinkercloud.training.backends.megatron_bridge import backend as mb

    async def inline(ref):
        return ref

    monkeypatch.setattr(mb, "_get", inline)
    sent = []
    worker = SimpleNamespace(forward_backward=SimpleNamespace(remote=lambda batch: sent.append(batch) or batch))
    handle = MegatronBridgeHandle(model_id="m", backend_type="megatron_bridge", num_labels=2, seq_length=4,
                                  worker=worker)
    backend = MegatronBridgeBackend()
    asyncio.run(backend.forward_backward(handle, [_datum([5, 6], [1])], CLASSIFICATION_CE))
    assert sent[0]["labels"].tolist() == [1]
    with pytest.raises(ValueError, match=r"class 2 is outside \[0, 2\)"):
        asyncio.run(backend.forward_backward(handle, [_datum([5, 6], [2])], CLASSIFICATION_CE))
    assert len(sent) == 1   # the bad batch never reached the worker



def test_training_needs_the_class_count_so_a_missing_one_is_an_error_not_an_unchecked_range():
    """The backend always passes num_labels. A caller that leaves it out would otherwise get no range check and
    no way to tell: 'range not checked' must not look like 'range checked'."""
    conv = NeMo2ClassificationDataConverter()
    for args in ({}, None, {"seq_length": 4}):
        with pytest.raises((KeyError, TypeError, ValueError)):
            conv.forward_backward_to_backend([_datum([5, 6], [1])], CLASSIFICATION_CE, args)


def test_an_integer_too_large_for_a_float_is_refused_with_the_datum_index():
    """float(10**400) raises OverflowError before any check, which would carry no datum index."""
    conv = NeMo2ClassificationDataConverter()
    bad = Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {"labels": {"data": [10 ** 400]}}})
    with pytest.raises(ValueError, match="datum 1: "):
        conv.forward_backward_to_backend([_datum([5, 6], [1]), bad], CLASSIFICATION_CE, {"num_labels": 2})


def test_the_error_does_not_echo_a_huge_integer_back_to_the_client():
    conv = NeMo2ClassificationDataConverter()
    for value in (10 ** 400, float(10 ** 300)):
        bad = Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {"labels": {"data": [value]}}})
        with pytest.raises(ValueError, match="datum 0: class .* is outside \\[0, 2\\)") as caught:
            conv.forward_backward_to_backend([bad], CLASSIFICATION_CE, {"num_labels": 2})
        assert len(str(caught.value)) < 120


def test_the_error_does_not_echo_a_long_label_list_back_to_the_client():
    """A client that sends per-token labels would get thousands of values echoed in the message."""
    conv = NeMo2ClassificationDataConverter()
    long = list(range(5000))
    bad = Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {"labels": {"data": long}}})
    with pytest.raises(ValueError) as caught:  # noqa: PT011  (#193)
        conv.forward_backward_to_backend([bad], CLASSIFICATION_CE, {"num_labels": 2})
    msg = str(caught.value)
    assert "datum 0" in msg and "5000" in msg and len(msg) < 300   # the count says what was wrong; the values stay out


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


def _fake_worker_module():
    worker = MagicMock()
    worker.ready.remote.return_value = None
    cls = MagicMock()
    cls.remote.return_value = worker
    mod = MagicMock()
    mod.MegatronBridgeWorker = cls
    return mod


def test_token_classification_is_refused_before_worker_starts():
    backend = MegatronBridgeBackend()
    config = {"architectures": ["RobertaForTokenClassification"], "num_labels": 8}
    with patch("tinkercloud.training.backends.megatron_bridge.backend.read_raw_hf_config", return_value=config):
        with pytest.raises(BackendError, match="token_classification"):
            asyncio.run(backend.create_model(
                model_id="m", request_id="r", base_model="tm", num_gpus=1,
                optimizer="adamw",
            ))


def test_sequence_classification_is_accepted():
    backend = MegatronBridgeBackend()
    config = {"architectures": ["RobertaForSequenceClassification"], "num_labels": 3}
    ray_mod = MagicMock()
    ray_mod.is_initialized.return_value = True
    with patch.dict("sys.modules", {"ray": ray_mod, "tinkercloud.training.backends.megatron_bridge.worker": _fake_worker_module()}):
        with patch("tinkercloud.training.backends.megatron_bridge.backend.read_raw_hf_config", return_value=config):
            with patch("tinkercloud.training.backends.megatron_bridge.backend._get", return_value=None):
                handle = asyncio.run(backend.create_model(
                    model_id="m", request_id="r", base_model="tm", num_gpus=1,
                    optimizer="adamw",
                ))
    assert isinstance(handle, MegatronBridgeHandle)
    assert handle.objective == Objective.SEQUENCE_CLASSIFICATION.value
