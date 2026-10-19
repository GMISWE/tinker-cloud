"""TrainingConfig: malformed SUPPORTED_MODELS is a startup error."""
import pytest


def test_malformed_supported_models_is_a_startup_error(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    monkeypatch.setenv("SUPPORTED_MODELS", "{not json")
    with pytest.raises(ValueError, match="SUPPORTED_MODELS"):
        TrainingConfig._get_default_models()
    monkeypatch.setenv("SUPPORTED_MODELS", '[{"model_name": "m", "max_context_length": 8}]')
    assert TrainingConfig._get_default_models()[0].model_name == "m"
