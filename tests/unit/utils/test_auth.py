"""utils.auth: the valid keys are AuthConfig.parsed_keys(), the one reading of the key setting that the
engine and the gateway share; no part of a key reaches the logs."""
import logging

import pytest
from fastapi import HTTPException

from tinkercloud.training.config import AuthConfig
from tinkercloud.training.utils.auth import APIKeyAuth


def test_the_key_setting_is_parsed_in_one_place():
    # the accepted keys, and the one an engine and the gateway send each other: the first in sorted order
    assert AuthConfig(api_key=" tml-b , tml-a ,").parsed_keys() == (frozenset({"tml-a", "tml-b"}), "tml-a")
    assert AuthConfig(api_key=" , ").parsed_keys() == (frozenset(), None)
    assert AuthConfig(api_key="tml-a", enabled=False).parsed_keys() == (None, None)   # auth off
    assert APIKeyAuth(AuthConfig(api_key=" tml-b , tml-a ,")).valid_keys == {"tml-a", "tml-b"}


def test_a_rejected_key_is_not_logged(caplog):
    auth = APIKeyAuth(AuthConfig(api_key="tml-real-key"))
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        auth.verify("tml-real-kez")
    assert "Invalid API key" in caplog.text and "tml-real" not in caplog.text
