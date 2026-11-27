"""Loss configuration keeps value kinds for the registry's boundary check."""
import pytest

from tinkercloud.training.core import loss_registry
from tinkercloud.training.models.requests import ForwardBackwardInput, ForwardInput


@pytest.mark.parametrize("model", [ForwardInput, ForwardBackwardInput])
@pytest.mark.parametrize("value", [True, False, "1.0"])
def test_non_numeric_loss_config_values_are_rejected_after_request_validation(model, value):
    request = model.model_validate({
        "data": [], "loss_fn": "ppo", "loss_fn_config": {"clip_high_threshold": value},
    })
    with pytest.raises(ValueError, match=r"clip_high_threshold.*must be a number"):
        loss_registry.validate(request.loss_fn, request.loss_fn_config)


@pytest.mark.parametrize("model", [ForwardInput, ForwardBackwardInput])
@pytest.mark.parametrize("value", [1.0, 1])
def test_numeric_loss_config_values_are_still_accepted(model, value):
    request = model.model_validate({
        "data": [], "loss_fn": "ppo", "loss_fn_config": {"clip_high_threshold": value},
    })
    loss_registry.validate(request.loss_fn, request.loss_fn_config)
    assert request.loss_fn_config == {"clip_high_threshold": 1.0}
