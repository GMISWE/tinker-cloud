"""SamplingParams.stop takes the forms the SDK's SamplingParams.stop declares: one string, strings, or token ids."""
from tinkercloud.training.models.requests import ASampleRequest, SamplingParams


def test_a_single_stop_string_is_one_stop_sequence():
    params = SamplingParams.model_validate({"max_tokens": 8, "stop": "\n\n"})
    assert params.stop == ["\n\n"]
    assert params.stop_token_ids is None


def test_a_single_stop_string_reaches_the_sample_request():
    request = ASampleRequest.model_validate({
        "prompt": {"chunks": [{"tokens": [1, 2]}]},
        "sampling_params": {"max_tokens": 8, "stop": "</s>"},
    })
    assert request.sampling_params.stop == ["</s>"]


def test_lists_keep_their_meaning():
    assert SamplingParams.model_validate({"max_tokens": 8, "stop": ["a", "b"]}).stop == ["a", "b"]
    ids = SamplingParams.model_validate({"max_tokens": 8, "stop": [5, 6]})
    assert ids.stop is None and ids.stop_token_ids == [5, 6]
