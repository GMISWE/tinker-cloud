"""
Authentication utilities for the training API.

This module provides API key verification and authentication helpers.
"""
import logging
from typing import Optional

from fastapi import HTTPException

from ..config import AuthConfig

logger = logging.getLogger(__name__)


class APIKeyAuth:
    """
    API key authentication manager.

    Handles API key verification with support for multiple keys
    and optional authentication bypass for development.
    """

    def __init__(self, config: AuthConfig):
        """
        Initialize API key authentication.

        Args:
            config: The auth settings; the valid keys are config.parsed_keys()
        """
        self.enabled = config.enabled
        self.header_name = config.api_key_header
        keys, _ = config.parsed_keys()
        self.valid_keys = set(keys or ())

        if not self.valid_keys and self.enabled:
            logger.warning("No API keys configured but authentication is enabled")

    def verify(self, provided_key: Optional[str] = None) -> None:
        """
        Verify the provided API key.

        Args:
            provided_key: The API key to verify

        Raises:
            HTTPException: If authentication fails
        """
        if self.is_valid(provided_key):
            # Success - log at debug level only
            logger.debug("API key validated successfully")
            return

        if not provided_key:
            logger.warning("API key not provided in request")
            raise HTTPException(
                status_code=401,
                detail=f"Missing API key (provide via {self.header_name} header)"
            )

        logger.warning("Invalid API key provided")   # no part of it: a mistyped key is mostly a real one
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    def is_valid(self, provided_key: Optional[str]) -> bool:
        """The decision verify() makes, as a boolean that logs nothing (for routes that also serve keyless callers)."""
        return not self.enabled or (bool(provided_key) and provided_key in self.valid_keys)

    def add_key(self, api_key: str) -> None:
        """
        Add a new valid API key.

        Args:
            api_key: API key to add
        """
        self.valid_keys.add(api_key)
        logger.info("Added an API key")

    def remove_key(self, api_key: str) -> bool:
        """
        Remove an API key.

        Args:
            api_key: API key to remove

        Returns:
            True if key was removed, False if not found
        """
        if api_key in self.valid_keys:
            self.valid_keys.remove(api_key)
            logger.info("Removed an API key")
            return True
        return False

    def list_keys(self) -> list[str]:
        """
        List all configured API keys (masked).

        Returns:
            List of masked API keys
        """
        return [f"{key[:8]}..." for key in self.valid_keys]


# Helper function for backward compatibility
def verify_api_key(
    x_api_key: Optional[str] = None,
    valid_api_key: str = "tml-dev-key"
) -> None:
    """
    Simple API key verification function.

    This is a lightweight wrapper for backward compatibility with
    the existing api.py implementation.

    Args:
        x_api_key: Provided API key
        valid_api_key: Expected API key

    Raises:
        HTTPException: If API key is invalid
    """
    if not x_api_key or x_api_key != valid_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")