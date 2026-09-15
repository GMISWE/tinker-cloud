"""
Health Router - System Health and Capabilities

Simple router for health checks and server capabilities.
No service layer needed - just queries system state.
"""
import logging
import ray
from datetime import datetime
from fastapi import APIRouter, Depends, Request

from ..models.responses import ClientConfigResponse, HealthResponse, ServerCapabilities, ModelInfo
from ..core.dependencies import verify_api_key_dep
from ..core.task_manager import TaskManager
from ..config import TrainingConfig
from ..proto.wire import zstd_available

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["health"]
)

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def _get_config(request: Request) -> TrainingConfig:
    config = getattr(request.app.state, "config", None)
    if config is None:
        raise RuntimeError("Training config not initialized")
    return config


@router.get("/health", response_model=HealthResponse)
async def health_simple(request: Request):
    """Health check for k8s probes (backward compatibility)"""
    runtime = _get_runtime(request)
    training_clients = runtime.training_clients

    return HealthResponse(
        status="healthy",
        version="3.1.0",
        timestamp=datetime.now().isoformat(),
        ray_initialized=ray.is_initialized(),
        active_training_clients=len(training_clients),
        model_ids=list(training_clients.keys()),
        futures_count=TaskManager.inflight_count()
    )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint - refactored with typed response"""
    runtime = _get_runtime(request)
    training_clients = runtime.training_clients

    return HealthResponse(
        status="healthy",
        version="3.1.0",
        timestamp=datetime.now().isoformat(),
        ray_initialized=ray.is_initialized(),
        active_training_clients=len(training_clients),
        model_ids=list(training_clients.keys()),
        futures_count=TaskManager.inflight_count()
    )


@router.get("/api/v1/healthz")
async def healthz():
    """Readiness probe in the SDK's shape (`HealthResponse.status` is the literal "ok")."""
    return {"status": "ok"}


@router.post("/api/v1/client/config", response_model=ClientConfigResponse)
async def client_config(request: Request):
    """Feature flags for the connecting SDK (>= 0.25 fetches this once at
    construction and fails without it). The body carries the SDK version;
    every client gets the same flags."""
    return ClientConfigResponse(proto_compress_fwdbwd=zstd_available())


@router.get("/api/v1/get_server_capabilities", response_model=ServerCapabilities)
async def get_server_capabilities(
    request: Request,
    _: None = Depends(verify_api_key_dep)
):
    """Get server capabilities - refactored with config and typed response"""
    config = _get_config(request)

    supported_models = [
        ModelInfo(
            model_name=model.model_name,
            max_context_length=model.max_context_length,
            supports_lora=model.supports_lora
        )
        for model in config.supported_models
    ]

    return ServerCapabilities(
        supported_models=supported_models,
        features=["gradient_accumulation", "lora", "checkpointing"],
        version="3.1.0"
    )


@router.post("/api/v1/telemetry")
async def send_telemetry(request: Request):
    """
    Telemetry endpoint - stub that accepts and discards data.

    No API key verification needed - telemetry is optional and non-critical.
    """
    return {"status": "accepted"}
