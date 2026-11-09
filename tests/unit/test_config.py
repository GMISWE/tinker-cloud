"""TrainingConfig: malformed SUPPORTED_MODELS is a startup error; every field validator
checks the value the environment supplies, not only one passed in explicitly."""
import pytest


@pytest.mark.parametrize("key", ["tml-dev-key", "", " , ", "real-key,tml-dev-key"])
def test_production_refuses_a_key_list_that_is_empty_or_holds_the_dev_key(monkeypatch, key):
    from tinkercloud.training.config import AuthConfig
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("TINKER_API_KEY", key)
    with pytest.raises(ValueError, match="Production requires a secure API key"):
        AuthConfig()


def test_production_takes_operator_api_keys_and_development_the_default(monkeypatch):
    from tinkercloud.training.config import AuthConfig
    monkeypatch.setenv("ENV", "production")
    for key in ("an-operator-key", "key-a, key-b"):
        monkeypatch.setenv("TINKER_API_KEY", key)
        assert AuthConfig().api_key == key
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


def test_ray_address_is_ray_address_or_a_local_head(monkeypatch):
    from tinkercloud.training.config import RayConfig
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("RAY_HEAD_SERVICE_HOST", "ray-head")   # Kubernetes sets it beside any Service named ray-head
    assert RayConfig().address is None   # ray.init(address=None): a local head
    monkeypatch.setenv("RAY_ADDRESS", "ray://elsewhere:10001")
    assert RayConfig().address == "ray://elsewhere:10001"


def test_malformed_supported_models_is_a_startup_error(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    monkeypatch.setenv("SUPPORTED_MODELS", "{not json")
    with pytest.raises(ValueError, match="SUPPORTED_MODELS"):
        TrainingConfig._get_default_models()
    monkeypatch.setenv("SUPPORTED_MODELS", '[{"model_name": "m", "max_context_length": 8}]')
    assert TrainingConfig._get_default_models()[0].model_name == "m"


def _production(monkeypatch):
    """ENV=production with the other production settings valid, so only Ray's auth mode decides."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("TINKER_API_KEY", "an-operator-key")
    monkeypatch.setenv("TINKERCLOUD_URL_SIGNING_KEY", "a-signing-key")


@pytest.mark.parametrize("mode", [None, "", "disabled"])
def test_production_refuses_ray_auth_off(monkeypatch, mode):
    from tinkercloud.training.config import TrainingConfig
    _production(monkeypatch)
    if mode is None:
        monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    else:
        monkeypatch.setenv("RAY_AUTH_MODE", mode)
    with pytest.raises(ValueError, match="RAY_AUTH_MODE"):
        TrainingConfig()


def test_production_takes_ray_auth_token(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    _production(monkeypatch)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    assert TrainingConfig().ray.auth_mode == "token"


def test_development_takes_ray_auth_unset(monkeypatch):
    from tinkercloud.training.config import TrainingConfig
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    assert TrainingConfig().ray.auth_mode is None
