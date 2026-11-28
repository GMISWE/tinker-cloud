"""Sampling parameter validation at the request boundary."""
import pytest
from pydantic import ValidationError

from tinkercloud.training.models.requests import ASampleRequest, SamplingParams


@pytest.mark.parametrize("value", [[], "bad", 7])
def test_non_object_sampling_params_are_validation_errors(value):
    with pytest.raises(ValidationError, match="sampling_params") as exc:
        ASampleRequest.model_validate({
            "prompt": {"tokens": [1, 2]}, "sampling_params": value,
        })
    assert exc.value.errors()[0]["input"] == value


def test_integer_stop_tokens_are_still_converted():
    params = SamplingParams.model_validate({
        "max_tokens": 8, "stop": [1, 2], "stop_token_ids": [3],
    })
    assert params.stop is None
    assert params.stop_token_ids == [3, 1, 2]
