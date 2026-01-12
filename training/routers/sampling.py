"""
Sampling Router - HTTP Layer for Model Sampling

Endpoints:
- POST /api/v1/asample - Async sampling via SGLang
- POST /api/v1/sample - Sync sampling via SGLang
- POST /api/v1/create_sampling_client - Create SGLang sampling client
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..services.sampling_service import SamplingService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep
from ..storage import FuturesStorage
from ..models.requests import (
    ASampleRequest,
    SampleRequest,
    CreateSamplingClientRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    SampleResult,
    SamplingSequence,
    CreateSamplingClientResult,
)
from ..utils import generate_request_id
from ..utils.helpers import find_model_with_rollout_manager

logger = logging.getLogger(__name__)

router = APIRouter()

def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_sampling_service(request: Request) -> SamplingService:
    """Dependency injection for SamplingService."""
    service = getattr(request.app.state, "sampling_service", None)
    if service is None:
        raise RuntimeError("SamplingService not initialized on app state")
    return service


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


# ============================================================================
# Sampling Endpoints
# ============================================================================

@router.post("/api/v1/asample", response_model=AsyncOperationResponse)
async def asample(
    request: ASampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Async sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Find model with RolloutManager
    model_id = find_model_with_rollout_manager(training_clients)
    if not model_id:
        raise HTTPException(status_code=404, detail="No model with RolloutManager found")

    # Extract prompt tokens and images (for VLM models)
    prompt_tokens = request.prompt.get_tokens()
    image_data = request.prompt.get_images() or None  # Convert empty list to None

    async def execute():
        result_dict = await service.async_sample(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            prompt_logprobs=request.prompt_logprobs,
            training_clients=training_clients,
            image_data=image_data
        )

        # Convert to response model
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(
            sequences=sequences,
            prompt_logprobs=result_dict.get("prompt_logprobs")
        )

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="asample",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/sample", response_model=AsyncOperationResponse)
async def sample(
    request: SampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Synchronous sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Find model with rollout manager
    model_id = find_model_with_rollout_manager(training_clients)
    if not model_id:
        raise HTTPException(status_code=404, detail="No model with RolloutManager found")

    async def execute():
        result_dict = await service.sync_sample(
            request_id=request_id,
            prompts=request.prompts,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            training_clients=training_clients
        )

        # Convert to response model
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(sequences=sequences)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="sample",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/create_sampling_client", response_model=AsyncOperationResponse)
async def create_sampling_client(
    request: CreateSamplingClientRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
):
    """
    Create sampling client (SGLang).
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Determine model path
    model_path = request.model_path or request.base_model
    if not model_path:
        raise HTTPException(status_code=400, detail="Either model_path or base_model must be provided")

    # Find model with rollout manager
    model_id = find_model_with_rollout_manager(training_clients)
    if not model_id:
        raise HTTPException(status_code=404, detail="No model with RolloutManager found")

    async def execute():
        result_dict = await service.create_sampling_client(
            request_id=request_id,
            model_path=request.model_path,
            base_model=request.base_model,
            training_clients=training_clients
        )
        return CreateSamplingClientResult(**result_dict)

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="create_sampling_client",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )
