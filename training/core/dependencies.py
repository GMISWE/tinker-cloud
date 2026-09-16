"""
Shared FastAPI dependencies.

Everything on app.state is set by create_app / the startup event before any
route is served, so each accessor reads the attribute directly: a missing
one is a wiring bug (AttributeError), not a request-time condition.
"""
from typing import Any, Dict, Optional
from fastapi import Depends, Header, Request

from .runtime_state import TrainingRuntimeState
from ..checkpoints import CheckpointStore
from ..config import TrainingConfig
from ..services.checkpoint_service import CheckpointService
from ..services.model_service import ModelService
from ..services.sampling_service import SamplingService
from ..services.session_service import SessionService
from ..services.training_service import TrainingService
from ..storage.futures import FuturesStorage
from ..storage.metadata import MetadataStorage
from ..utils.auth import APIKeyAuth


def get_config(request: Request) -> TrainingConfig:
    return request.app.state.config


def get_runtime(request: Request) -> TrainingRuntimeState:
    return request.app.state.runtime


def get_training_clients(request: Request) -> Dict[str, Dict[str, Any]]:
    return request.app.state.runtime.training_clients


def get_training_runs_metadata(request: Request) -> Dict[str, Dict[str, Any]]:
    return request.app.state.runtime.training_runs_metadata


def get_poll_tracking(request: Request) -> Dict[str, Dict[str, Any]]:
    return request.app.state.runtime.poll_tracking


def get_auth(request: Request) -> APIKeyAuth:
    return request.app.state.auth


def get_checkpoint_store(request: Request) -> CheckpointStore:
    return request.app.state.checkpoint_store


def get_futures_storage(request: Request) -> FuturesStorage:
    return request.app.state.futures_storage


def get_metadata_storage(request: Request) -> MetadataStorage:
    return request.app.state.metadata_storage


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_training_service(request: Request) -> TrainingService:
    return request.app.state.training_service


def get_checkpoint_service(request: Request) -> CheckpointService:
    return request.app.state.checkpoint_service


def get_sampling_service(request: Request) -> SamplingService:
    return request.app.state.sampling_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


async def verify_api_key_dep(
    x_api_key: Optional[str] = Header(None),
    auth: APIKeyAuth = Depends(get_auth),
):
    """Dependency to verify API key"""
    auth.verify(x_api_key)