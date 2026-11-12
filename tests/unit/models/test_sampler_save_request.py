"""Which sampler save is ephemeral."""
import pytest

from tinkercloud.training.models.requests import SaveWeightsForSamplerRequest


@pytest.mark.parametrize("fields, ephemeral", [
    ({"sampling_session_seq_id": 0}, True),
    ({"sampling_session_seq_id": 0, "path": "p"}, False),
    ({"sampling_session_seq_id": 0, "name": "n"}, False),
    ({}, False),
])
def test_a_sampler_save_is_ephemeral_when_it_has_a_session_and_names_no_checkpoint(fields, ephemeral):
    """The route and the service each held a copy of this rule; the request now holds the one."""
    assert SaveWeightsForSamplerRequest(model_id="m", **fields).ephemeral is ephemeral
