"""What the gateway refuses to start with."""
import pytest

from training.gateway.app import build_app
from training.gateway.registry import Registry

from .support import config


def test_auth_enabled_with_no_key_fails_at_startup():
    """A gateway that would answer 401 to everything never starts: the empty key set is missing config."""
    cfg = config("aiohttp")
    cfg.auth.api_key = " , ,"
    with pytest.raises(ValueError, match="names no key"):
        build_app(cfg, Registry())
