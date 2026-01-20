"""
Training Router - HTTP Endpoints for Training Operations

Thin HTTP layer that:
1. Validates HTTP request
2. Delegates to TrainingService for business logic
3. Uses TaskManager for async execution
4. Returns HTTP response

No business logic here - just HTTP concerns.
"""
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict

from ..models.requests import ForwardRequest, ForwardBackwardRequest, OptimStepRequest
from ..models.responses import AsyncOperationResponse
from ..services.training_service import TrainingService
from ..core.task_manager import TaskManager
from ..storage import FuturesStorage

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["training"],
    responses={404: {"description": "Model not found"}}
)


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Training runtime state not initialized")
    return runtime


def get_training_clients(request: Request) -> Dict:
    """Get training clients from runtime state."""
    runtime = _get_runtime(request)
    return runtime.training_clients


def get_futures_storage(request: Request) -> FuturesStorage:
    """Get futures storage instance from app state."""
    storage = getattr(request.app.state, "futures_storage", None)
    if storage is None:
        raise RuntimeError("Futures storage not initialized")
    return storage


def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:16]}"


def get_training_service() -> TrainingService:
    """Dependency: Get training service instance"""
    return TrainingService()


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Dependency: Get task manager instance"""
    return TaskManager(futures_storage)


@router.post("/api/v1/forward", response_model=AsyncOperationResponse)
async def forward(
    request: ForwardRequest,
    service: TrainingService = Depends(get_training_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
) -> AsyncOperationResponse:
    """
    Forward-only pass (no gradients).

    Used for DPO reference model inference.

    Returns:
        AsyncOperationResponse with request_id to poll for results
    """
    request_id = generate_request_id()

    # Check if model exists (HTTP concern)
    if request.model_id not in training_clients:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_id} not found"
        )

    # Get client info
    client_info = training_clients[request.model_id]
    train_group = client_info["train_group"]

    # Business logic wrapped in async task
    async def execute_forward():
        return await service.forward(
            model_id=request.model_id,
            train_group=train_group,
            data=request.forward_input.data,
            loss_fn=request.forward_input.loss_fn,
            client_info=client_info
        )

    # Create background task with automatic error handling
    task_manager.create_task(
        request_id=request_id,
        operation="forward",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute_forward
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/forward_backward", response_model=AsyncOperationResponse)
async def forward_backward(
    request: ForwardBackwardRequest,
    service: TrainingService = Depends(get_training_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
) -> AsyncOperationResponse:
    """
    Forward-backward pass (accumulate gradients, no optimizer step).

    Returns:
        AsyncOperationResponse with request_id to poll for results
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_id} not found"
        )

    # Get client info
    client_info = training_clients[request.model_id]
    train_group = client_info["train_group"]
    args = client_info["args"]

    # Business logic wrapped in async task
    async def execute_forward_backward():
        return await service.forward_backward(
            model_id=request.model_id,
            train_group=train_group,
            args=args,
            data=request.forward_backward_input.data,
            loss_fn=request.forward_backward_input.loss_fn,
            client_info=client_info
        )

    # Create background task
    task_manager.create_task(
        request_id=request_id,
        operation="forward_backward",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute_forward_backward
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/optim_step", response_model=AsyncOperationResponse)
async def optim_step(
    request: OptimStepRequest,
    service: TrainingService = Depends(get_training_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
) -> AsyncOperationResponse:
    """
    Apply optimizer step to update model weights.

    Returns:
        AsyncOperationResponse with request_id to poll for results
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model_id} not found"
        )

    # Get client info
    client_info = training_clients[request.model_id]
    train_group = client_info["train_group"]

    # Business logic wrapped in async task
    async def execute_optim_step():
        return await service.apply_optimizer_step(
            model_id=request.model_id,
            train_group=train_group,
            client_info=client_info,
            adam_params=request.adam_params
        )

    # Create background task
    task_manager.create_task(
        request_id=request_id,
        operation="optim_step",
        model_id=request.model_id,
        payload=request.dict(),
        task_func=execute_optim_step
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )
