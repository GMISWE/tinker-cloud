"""utils.auth: no part of a key reaches the logs; is_valid answers without logging."""
import logging

import pytest
from fastapi import HTTPException

from tinkercloud.training.utils.auth import APIKeyAuth


def test_is_valid_checks_without_logging(caplog):
    auth = APIKeyAuth(api_key="tml-secret-a, tml-secret-b")
    with caplog.at_level(logging.DEBUG):
        assert auth.is_valid("tml-secret-b")
        assert not auth.is_valid("tml-secret-c")
        assert not auth.is_valid(None)
        assert not auth.is_valid("")
    assert caplog.records == []
    assert APIKeyAuth(api_key="k", enabled=False).is_valid(None)


def test_a_rejected_key_is_not_logged(caplog):
    auth = APIKeyAuth(api_key="tml-real-key")
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException):
        auth.verify("tml-real-kez")
    assert "Invalid API key" in caplog.text and "tml-real" not in caplog.text
