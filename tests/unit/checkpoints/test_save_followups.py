"""The duplicate-save error's place among the store's errors, and which sampler save is ephemeral."""
import pytest

from tinkercloud.training.checkpoints import CheckpointError, CheckpointExists
from tinkercloud.training.models.requests import SaveWeightsForSamplerRequest


def test_a_duplicate_save_is_a_checkpoint_error_the_client_caused():
    """It was a bare Exception, outside the errors the application maps to a status."""
    assert issubclass(CheckpointExists, CheckpointError)
    assert CheckpointExists.status_code == 400


@pytest.mark.parametrize("fields, ephemeral", [
    ({"sampling_session_seq_id": 0}, True),
    ({"sampling_session_seq_id": 0, "path": "p"}, False),
    ({"sampling_session_seq_id": 0, "name": "n"}, False),
    ({}, False),
])
def test_a_sampler_save_is_ephemeral_when_it_has_a_session_and_names_no_checkpoint(fields, ephemeral):
    """The route and the service each held a copy of this rule; the request now holds the one."""
    assert SaveWeightsForSamplerRequest(model_id="m", **fields).ephemeral is ephemeral
