"""
Routers Layer - HTTP Handlers

Routers are thin FastAPI handlers that delegate to Services.
"""

from .training import router as training_router
from .health import router as health_router
from .futures import router as futures_router
from .models import router as models_router
from .checkpoints import router as checkpoints_router
from .sampling import router as sampling_router
from .session import router as session_router

__all__ = [
    "training_router",
    "health_router",
    "futures_router",
    "models_router",
    "checkpoints_router",
    "sampling_router",
    "session_router"
]
