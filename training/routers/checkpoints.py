"""
Checkpoints Router - HTTP Layer for Checkpoint Management

Endpoints:
- POST /api/v1/save_weights - Save model weights to disk
- POST /api/v1/save_weights_for_sampler - Save weights for SGLang sampler
- POST /api/v1/load_weights - Deprecated endpoint (returns error message)
- POST /api/v1/weights_info - Get weights/checkpoint info from tinker path
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..services.checkpoint_service import CheckpointService
from ..services.session_service import SessionService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    SaveWeightsRequest,
    SaveWeightsForSamplerRequest,
    WeightsInfoRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    SaveWeightsForSamplerResult,
    DeprecatedEndpointError,
    WeightsInfoResponse,
)
from ..utils import generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_checkpoint_service(request: Request) -> CheckpointService:
    """Dependency injection for CheckpointService."""
    service = getattr(request.app.state, "checkpoint_service", None)
    if service is None:
        raise RuntimeError("CheckpointService not initialized on app state")
    return service


def get_metadata_storage(request: Request) -> MetadataStorage:
    """Dependency injection for MetadataStorage."""
    storage = getattr(request.app.state, "metadata_storage", None)
    if storage is None:
        raise RuntimeError("MetadataStorage not initialized on app state")
    return storage


def get_futures_storage(request: Request) -> FuturesStorage:
    """Dependency injection for FuturesStorage."""
    storage = getattr(request.app.state, "futures_storage", None)
    if storage is None:
        raise RuntimeError("FuturesStorage not initialized on app state")
    return storage


def get_training_clients(request: Request) -> Dict[str, Dict[str, Any]]:
    """Dependency injection for training_clients."""
    runtime = _get_runtime(request)
    return runtime.training_clients


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


def get_session_service(request: Request) -> SessionService:
    """Dependency injection for SessionService."""
    service = getattr(request.app.state, "session_service", None)
    if service is None:
        raise RuntimeError("SessionService not initialized on app state")
    return service


# ============================================================================
# Checkpoint Management Endpoints
# ============================================================================

@router.post("/api/v1/save_weights", response_model=AsyncOperationResponse)
async def save_weights(
    request: SaveWeightsRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Save model weights to disk.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    async def execute():
        return await service.save_weights(
            model_id=request.model_id,
            request_id=request_id,
            path=request.path,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="save_weights",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/save_weights_for_sampler", response_model=AsyncOperationResponse)
async def save_weights_for_sampler(
    request: SaveWeightsForSamplerRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Save weights for SGLang sampler.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    # Get base_model from training client (use None if missing, not empty string)
    client_info = training_clients[request.model_id]
    base_model = client_info.get("base_model") or None

    async def execute():
        result = await service.save_weights_for_sampler(
            model_id=request.model_id,
            request_id=request_id,
            name=request.name,
            training_clients=training_clients,
            metadata_storage=metadata_storage,
            path=request.path,
            sampling_session_seq_id=request.sampling_session_seq_id
        )

        # Register ephemeral sampler with session if sampling_session_id was created
        sampling_session_id = result.get("sampling_session_id")
        if sampling_session_id:
            checkpoint_path = result.get("checkpoint_path")
            session_service.register_ephemeral_sampler(
                sampler_id=sampling_session_id,
                model_id=request.model_id,
                base_model=base_model,
                model_path=checkpoint_path
            )

        return SaveWeightsForSamplerResult(**result)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="save_weights_for_sampler",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/load_weights")
async def load_weights(
    _: None = Depends(verify_api_key_dep)
):
    """Deprecated endpoint - use checkpoint_path in create_model instead."""
    return DeprecatedEndpointError(
        error="Endpoint deprecated",
        reason="load_weights is no longer supported as a separate operation",
        solution={
            "description": "Use checkpoint_path parameter in create_model request",
            "example": {
                "base_model": "meta-llama/Llama-3.1-8B",
                "checkpoint_path": "tinker://run_abc123/weights/checkpoint_001"
            }
        }
    )


@router.post("/api/v1/weights_info", response_model=WeightsInfoResponse)
async def weights_info(
    request: WeightsInfoRequest,
    _: None = Depends(verify_api_key_dep),
    training_clients: Dict = Depends(get_training_clients),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage)
):
    """
    Get weights/checkpoint info from tinker path.
    Used for loading checkpoints via create_training_client_from_state.

    Parses tinker:// URI and returns model metadata needed for checkpoint loading.
    Validates both the model exists AND the specific checkpoint is recorded.
    """
    tinker_path = request.tinker_path
    logger.info(f"weights_info request for: {tinker_path}")

    # Parse tinker:// path: tinker://model_xxx/weights/checkpoint_name
    if not tinker_path.startswith("tinker://"):
        raise HTTPException(status_code=400, detail=f"Invalid tinker path: {tinker_path}")

    # Extract model_id and checkpoint_name from path: tinker://model_xxx/weights/checkpoint_name
    path_parts = tinker_path[9:].split("/")  # Remove "tinker://"
    if len(path_parts) < 1:
        raise HTTPException(status_code=400, detail=f"Invalid tinker path format: {tinker_path}")

    model_id = path_parts[0]
    checkpoint_name = path_parts[2] if len(path_parts) >= 3 else None
    logger.info(f"Extracted model_id: {model_id}, checkpoint_name: {checkpoint_name}")

    # Try to find model in active training clients first
    if model_id in training_clients:
        client_info = training_clients[model_id]
        base_model = client_info.get("base_model", "")
        args = client_info.get("args", {})
        lora_rank = getattr(args, "lora_rank", None) if args else None

        # Determine if LoRA is enabled
        is_lora = lora_rank is not None and lora_rank > 0

        # Verify checkpoint exists if name was provided
        if checkpoint_name:
            checkpoint_meta = metadata_storage.load_checkpoint(model_id, checkpoint_name)
            if not checkpoint_meta:
                logger.warning(f"Checkpoint {checkpoint_name} not found for model {model_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Checkpoint not found: {checkpoint_name} for model {model_id}"
                )

        logger.info(f"Found active model: base_model={base_model}, is_lora={is_lora}, lora_rank={lora_rank}")
        return WeightsInfoResponse(
            base_model=base_model,
            is_lora=is_lora,
            lora_rank=lora_rank if is_lora else None
        )

    # If not in active clients, try metadata storage
    metadata = metadata_storage.load_training_run(model_id)
    if metadata:
        base_model = metadata.get("base_model", "")
        lora_config = metadata.get("lora_config", {})
        lora_rank = lora_config.get("rank", 0) if lora_config else 0
        is_lora = lora_rank > 0

        # Verify checkpoint exists if name was provided
        if checkpoint_name:
            checkpoint_meta = metadata_storage.load_checkpoint(model_id, checkpoint_name)
            if not checkpoint_meta:
                logger.warning(f"Checkpoint {checkpoint_name} not found for model {model_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Checkpoint not found: {checkpoint_name} for model {model_id}"
                )

        logger.info(f"Found stored metadata: base_model={base_model}, is_lora={is_lora}, lora_rank={lora_rank}")
        return WeightsInfoResponse(
            base_model=base_model,
            is_lora=is_lora,
            lora_rank=lora_rank if is_lora else None
        )

    # Model not found anywhere
    logger.warning(f"Model not found: {model_id}")
    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
