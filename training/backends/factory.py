"""
Backend factory — creates the appropriate TrainingBackend from configuration.

`SUPPORTED_BACKENDS` is the one allow-list; the CLI flag, the config
validator and the signature-contract test all import it from here.
"""
import importlib
import logging
from typing import Any, Dict, Optional, Type

from .base import TrainingBackend

logger = logging.getLogger(__name__)

# backend_type -> (module relative to this package, class name)
_BACKEND_CLASSES = {
    "miles": (".miles.backend", "MilesBackend"),
    "nemo_rl": (".nemo_rl.backend", "NemoRLBackend"),
    "verl": (".verl.backend", "VerlBackend"),
    "megatron_bridge": (".megatron_bridge.backend", "MegatronBridgeBackend"),
    "automodel": (".automodel.backend", "AutomodelBackend"),
    "fake": (".fake.backend", "FakeBackend"),  # deterministic CPU backend for protocol tests
}
SUPPORTED_BACKENDS = tuple(_BACKEND_CLASSES)


class BackendFactory:
    """Creates the appropriate backend based on configuration."""

    @staticmethod
    def backend_class(backend_type: str) -> Type[TrainingBackend]:
        """Import and return the backend class without instantiating it.

        Raises ValueError for an unknown type; ImportError propagates when the
        backend's runtime dependencies are absent.
        """
        try:
            module_name, class_name = _BACKEND_CLASSES[backend_type]
        except KeyError:
            raise ValueError(
                f"Unknown backend: {backend_type!r}. "
                f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}"
            ) from None
        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, class_name)

    @staticmethod
    def create(
        backend_type: str,
        backend_overrides: Optional[Dict[str, Any]] = None,
    ) -> TrainingBackend:
        """Instantiate the training backend for `backend_type` (see SUPPORTED_BACKENDS)."""
        cls = BackendFactory.backend_class(backend_type)
        logger.info("Creating %s backend (%s)", backend_type, cls.__name__)
        return cls(backend_overrides or {})
