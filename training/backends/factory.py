"""
Backend factory — creates the appropriate TrainingBackend from configuration.
"""
import logging
from typing import Optional, Dict, Any

from .base import TrainingBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """Creates the appropriate backend based on configuration."""

    @staticmethod
    def create(
        backend_type: str,
        backend_overrides: Optional[Dict[str, Any]] = None,
    ) -> TrainingBackend:
        """
        Instantiate the training backend for the given type.

        Args:
            backend_type: "miles" or "nemo_rl"
            backend_overrides: Optional backend-specific config overrides.

        Returns:
            A TrainingBackend instance.

        Raises:
            ValueError: If backend_type is unknown.
        """
        overrides = backend_overrides or {}

        if backend_type == "miles":
            from .miles.backend import MilesBackend
            logger.info("Creating Miles backend")
            return MilesBackend(overrides)

        elif backend_type == "nemo_rl":
            from .nemo_rl.backend import NemoRLBackend
            logger.info("Creating NeMo RL backend")
            return NemoRLBackend(overrides)

        else:
            raise ValueError(
                f"Unknown backend: {backend_type!r}. "
                f"Supported backends: miles, nemo_rl"
            )
