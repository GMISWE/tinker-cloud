"""NemoRLDataConverter on synthetic wire datums: batched shapes, variable lengths, the deferred result."""
import importlib.util

import pytest

from tests.datums import make_synthetic_batch, make_synthetic_datum

HAS_NEMO_RL = importlib.util.find_spec("nemo_rl") is not None


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
