"""TrainingConfig: malformed SUPPORTED_MODELS is a startup error; every field validator
checks the value the environment supplies, not only one passed in explicitly."""
import pytest


@pytest.mark.parametrize("key", ["tml-dev-key", ""])
def test_production_refuses_the_default_or_an_empty_api_key(monkeypatch, key):
    from tinkercloud.training.config import AuthConfig
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("TINKER_API_KEY", key)
    with pytest.raises(ValueError, match="Production requires a secure API key"):
        AuthConfig()


def test_production_takes_an_operator_api_key_and_development_the_default(monkeypatch):
    from tinkercloud.training.config import AuthConfig
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("TINKER_API_KEY", "an-operator-key")
    assert AuthConfig().api_key == "an-operator-key"
    monkeypatch.delenv("ENV")
    monkeypatch.delenv("TINKER_API_KEY")
    assert AuthConfig().api_key == "tml-dev-key"


def test_public_base_url_from_the_environment_loses_its_trailing_slash(monkeypatch):
    from tinkercloud.training.config import ExternalWeightsConfig
    monkeypatch.setenv("TINKERCLOUD_PUBLIC_BASE_URL", "https://tinker.example.com/")
    assert ExternalWeightsConfig().public_base_url == "https://tinker.example.com"


def test_an_unknown_backend_in_the_environment_is_a_startup_error(monkeypatch):
    from tinkercloud.training.config import BackendConfig
    monkeypatch.setenv("TINKERCLOUD_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unsupported backend: 'bogus'"):
        BackendConfig()
    monkeypatch.setenv("TINKERCLOUD_BACKEND", "fake")
    assert BackendConfig().backend_type == "fake"


def test_the_factory_serves_exactly_the_config_backend_names():
    from tinkercloud.training.backends.factory import SUPPORTED_BACKENDS
    from tinkercloud.training.config import BACKEND_NAMES
    assert SUPPORTED_BACKENDS == BACKEND_NAMES


def test_ray_address_explicit_then_kubernetes_service_then_local_head(monkeypatch):
    from tinkercloud.training.config import RayConfig
    for var in ("RAY_ADDRESS", "RAY_HEAD_SERVICE_HOST", "RAY_HEAD_SERVICE_PORT_CLIENT"):
        monkeypatch.delenv(var, raising=False)
    assert RayConfig().address is None   # ray.init(address=None): a local head
    monkeypatch.setenv("RAY_HEAD_SERVICE_HOST", "ray-head")
    assert RayConfig().address == "ray://ray-head:10001"
    monkeypatch.setenv("RAY_HEAD_SERVICE_PORT_CLIENT", "20001")
    assert RayConfig().address == "ray://ray-head:20001"
    monkeypatch.setenv("RAY_ADDRESS", "ray://elsewhere:10001")
    assert RayConfig().address == "ray://elsewhere:10001"


def test_malformed_supported_models_is_a_startup_error(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    monkeypatch.setenv("SUPPORTED_MODELS", "{not json")
    with pytest.raises(ValueError, match="SUPPORTED_MODELS"):
        TrainingConfig._get_default_models()
    monkeypatch.setenv("SUPPORTED_MODELS", '[{"model_name": "m", "max_context_length": 8}]')
    assert TrainingConfig._get_default_models()[0].model_name == "m"
