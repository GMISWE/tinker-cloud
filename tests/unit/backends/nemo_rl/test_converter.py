"""NemoRLDataConverter on synthetic wire datums: batched shapes, variable lengths, the deferred result."""
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from tinkercloud.training.models.requests import ModelInput
from tests.datums import ce_datum, make_synthetic_batch, make_synthetic_datum

HAS_NEMO_RL = importlib.util.find_spec("nemo_rl") is not None


@pytest.fixture
def sft_converter(monkeypatch):
    from tinkercloud.training.backends.nemo_rl.converter import NemoRLDataConverter
    if not HAS_NEMO_RL:
        # Conversion only needs the dict interface, not NeMo's distributed methods.
        monkeypatch.setitem(sys.modules, "nemo_rl.distributed.batched_data_dict",
                            SimpleNamespace(BatchedDataDict=dict))
    return NemoRLDataConverter()


@pytest.mark.parametrize("field,count", [
    ("target_tokens", 0), ("target_tokens", 2), ("target_tokens", 4),
    ("weights", 0), ("weights", 2), ("weights", 4),
])
def test_sft_rejects_mismatched_lengths(sft_converter, field, count):
    datum = ce_datum([10, 20, 30], [20, 30, 40], [1.0, 1.0, 1.0])
    datum.loss_fn_inputs[field].data = [1] * count
    with pytest.raises(ValueError, match=f"datum 1: {field} has {count} entries for 3 tokens"):
        sft_converter.forward_backward_to_backend([ce_datum([1]), datum], "cross_entropy", {})


@pytest.mark.parametrize("model_input", [
    {"tokens": [10, 20, 30]},
    {"input_ids": [10, 20, 30]},
    {"chunks": [{"tokens": [10]}, {"tokens": [20, 30]}]},
])
def test_sft_preserves_weights_and_batch_padding(sft_converter, model_input):
    datum = ce_datum([10, 20, 30], [20, 30, 40], [0.0, 0.5, -0.25])
    datum.model_input = ModelInput.model_validate(model_input)
    result = sft_converter.forward_backward_to_backend(
        [datum, ce_datum([50], [60], [1.0])], "cross_entropy", {},
    )
    assert result["input_ids"].tolist() == [[10, 20, 30, 40], [50, 60, 0, 0]]
    assert result["input_lengths"].tolist() == [4, 2]
    assert result["token_mask"].tolist() == [[0.0, 0.0, 0.5, -0.25], [0.0, 1.0, 0.0, 0.0]]
    assert result["sample_mask"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("missing", ["target_tokens", "weights"])
def test_sft_keeps_missing_field_behavior(sft_converter, missing):
    datum = ce_datum([10, 20, 30], [20, 30, 40], [0.5, 1.0, 0.0])
    del datum.loss_fn_inputs[missing]
    result = sft_converter.forward_backward_to_backend([datum], "cross_entropy", {})
    if missing == "weights":
        assert result["token_mask"].tolist() == [[0.0, 1.0, 1.0, 1.0]]
        assert result["sample_mask"].tolist() == [1.0]
    else:
        assert result["input_lengths"].tolist() == [3]
        assert result["sample_mask"].tolist() == [0.0]


@pytest.mark.parametrize("count", [3, 4, 5])
def test_sft_counts_expanded_image_tokens(sft_converter, count):
    datum = ce_datum([10, 20], [0, 0, 20, 30], [0.0] * count)
    datum.model_input = ModelInput.model_validate({"chunks": [
        {"tokens": [10]}, {"type": "image", "expected_tokens": 2}, {"tokens": [20]},
    ]})
    image_preprocessor = SimpleNamespace(image_token_id=99)
    if count != 4:
        with pytest.raises(ValueError, match=f"datum 0: weights has {count} entries for 4 tokens"):
            sft_converter.forward_backward_to_backend(
                [datum], "cross_entropy", {}, image_preprocessor=image_preprocessor,
            )
    else:
        result = sft_converter.forward_backward_to_backend(
            [datum], "cross_entropy", {}, image_preprocessor=image_preprocessor,
        )
        assert result["input_ids"].tolist() == [[10, 99, 99, 20, 30]]
        assert result["input_lengths"].tolist() == [5]


@pytest.mark.skipif(not HAS_NEMO_RL, reason="NeMo RL not installed")
class TestNemoRLDataConverter:
    """Test NemoRLDataConverter with synthetic data."""

    @pytest.fixture
    def converter(self):
        from tinkercloud.training.backends.nemo_rl.converter import NemoRLDataConverter
        return NemoRLDataConverter()

    def test_forward_to_backend_shapes(self, converter):
        data = make_synthetic_batch(batch_size=4, seq_len=128)
        config = {}
        result = converter.forward_to_backend(data, config)

        assert "input_ids" in result
        assert "input_lengths" in result
        assert "token_mask" in result
        assert result["input_ids"].shape[0] == 4  # batch size
        assert result["input_ids"].shape[1] == 128  # max seq len

    def test_forward_backward_to_backend_shapes(self, converter):
        data = make_synthetic_batch(batch_size=4, seq_len=128)
        config = {}
        result = converter.forward_backward_to_backend(data, "importance_sampling", config)

        expected_keys = {
            "input_ids", "input_lengths", "token_mask",
            "advantages", "prev_logprobs", "generation_logprobs",
            "reference_policy_logprobs", "sample_mask",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["input_ids"].shape == (4, 128)
        assert result["sample_mask"].shape == (4,)
        assert result["sample_mask"].sum().item() == 4.0

    def test_forward_backward_empty_data(self, converter):
        result = converter.forward_backward_to_backend([], "importance_sampling", {})
        assert result["input_ids"].shape[0] == 0

    def test_variable_length_sequences(self, converter):
        data = [
            make_synthetic_datum(seq_len=64, response_len=32),
            make_synthetic_datum(seq_len=128, response_len=64),
        ]
        result = converter.forward_backward_to_backend(data, "importance_sampling", {})
        assert result["input_ids"].shape == (2, 128)
        assert result["input_ids"][0, 65:].sum().item() == 0

    def test_deferred_result_conversion(self, converter):
        result = converter.backend_to_forward_backward_result(None, [], loss_fn="importance_sampling")
        assert result["deferred"] is True
        assert result["metrics"] == {}
        assert result["loss_fn_output_type"] == "importance_sampling"
