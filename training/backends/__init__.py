"""
Backend abstraction layer for TinkerCloud.

Provides a unified interface for different post-training backends
(Miles, NeMo RL) behind a common TrainingBackend ABC.
"""
from .base import (
    TrainingBackend,
    ArgumentBuilder,
    DataConverter,
    BackendHandle,
    BackendError,
)
from .factory import BackendFactory

__all__ = [
    "TrainingBackend",
    "ArgumentBuilder",
    "DataConverter",
    "BackendHandle",
    "BackendError",
    "BackendFactory",
]
