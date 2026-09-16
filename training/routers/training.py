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
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Dict, Tuple

from ..models.requests import ForwardRequest, ForwardBackwardRequest, OptimStepRequest
from ..models.responses import AsyncOperationResponse
from ..services.training_service import TrainingService
from ..core.task_manager import TaskManager
from ..core.dependencies import verify_api_key_dep, get_training_clients, get_futures_storage, get_training_service
from ..core import loss_registry
from ..proto.wire import PROTO_CONTENT_TYPE, WireError, decompress_zstd, parse_forward_backward_request
from ..storage import FuturesStorage

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["training"],
    responses={404: {"description": "Model not found"}}
)


def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{uuid.uuid4().hex[:16]}"


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Dependency: Get task manager instance"""
    return TaskManager(futures_storage)


def _validate_loss(loss_fn: str, loss_fn_config) -> None:
    """Unknown loss names / keys are a client error, not a silent default."""
    try:
        loss_registry.validate(loss_fn, loss_fn_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _submit_pass(
    *, forward_only: bool, model_id: str, seq_id, batch, payload: Dict,
    service: TrainingService, task_manager: TaskManager, training_clients: Dict,
) -> AsyncOperationResponse:
    """Queue a forward or forward_backward pass over `batch` (a ForwardInput /
    ForwardBackwardInput) and return its future. Shared by both endpoints:
    which pass runs is decided here, not by the URL."""
    if model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    client_info = training_clients[model_id]
    _validate_loss(batch.loss_fn, batch.loss_fn_config)
    run = service.forward if forward_only else service.forward_backward

    async def execute():
        return await run(
            model_id=model_id,
            data=batch.data,
            loss_fn=batch.loss_fn,
            client_info=client_info,
            loss_fn_config=batch.loss_fn_config,
        )

    request_id = task_manager.create_task(
        request_id=generate_request_id(),
        operation="forward" if forward_only else "forward_backward",
        model_id=model_id,
        payload=payload,
        seq_id=seq_id,
        task_func=execute,
    )
    return AsyncOperationResponse(request_id=request_id, model_id=model_id)


async def _read_forward_backward_request(http_request: Request) -> Tuple[ForwardBackwardRequest, bool]:
    """The forward_backward body in either wire format -> (request, forward_only).

    SDK >= 0.25 sends protobuf (`Content-Type: application/x-protobuf`,
    possibly `Content-Encoding: zstd`) and routes forward-only passes here via
    the proto's `forward_only` flag; older clients send JSON, which is never
    forward-only. Both yield the same pydantic request. Malformed bodies are
    422 in FastAPI's own shape, like a body FastAPI parsed itself.
    """
    body = await http_request.body()
    try:
        if http_request.headers.get("content-encoding", "").strip().lower() == "zstd":
            body = decompress_zstd(body)
        if PROTO_CONTENT_TYPE in http_request.headers.get("content-type", "").lower():
            request_dict, forward_only = parse_forward_backward_request(body)
            return ForwardBackwardRequest.model_validate(request_dict), forward_only
        return ForwardBackwardRequest.model_validate_json(body), False
    except WireError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValidationError as e:
        raise RequestValidationError(e.errors())


@router.post("/api/v1/forward", response_model=AsyncOperationResponse)
async def forward(
    request: ForwardRequest,
    _: None = Depends(verify_api_key_dep),
    service: TrainingService = Depends(get_training_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
) -> AsyncOperationResponse:
    """Forward-only pass (no gradients), JSON body. SDK >= 0.25 does not call
    this: it sends forward-only passes to /api/v1/forward_backward as proto."""
    return _submit_pass(
        forward_only=True, model_id=request.model_id, seq_id=request.seq_id,
        batch=request.forward_input, payload=request.dict(),
        service=service, task_manager=task_manager, training_clients=training_clients,
    )


@router.post("/api/v1/forward_backward", response_model=AsyncOperationResponse)
async def forward_backward(
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    service: TrainingService = Depends(get_training_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients)
) -> AsyncOperationResponse:
    """Forward-backward pass (accumulate gradients, no optimizer step), or a
    forward-only pass when the proto body asks for it. Proto or JSON body."""
    request, forward_only = await _read_forward_backward_request(http_request)
    return _submit_pass(
        forward_only=forward_only, model_id=request.model_id, seq_id=request.seq_id,
        batch=request.forward_backward_input, payload=request.dict(),
        service=service, task_manager=task_manager, training_clients=training_clients,
    )


@router.post("/api/v1/optim_step", response_model=AsyncOperationResponse)
async def optim_step(
    request: OptimStepRequest,
    _: None = Depends(verify_api_key_dep),
    http_request: Request = None,
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

    # Business logic wrapped in async task
    async def execute_optim_step():
        return await service.apply_optimizer_step(
            model_id=request.model_id,
            client_info=client_info,
            adam_params=request.adam_params,
        )

    # Create background task
    request_id = task_manager.create_task(
        request_id=request_id,
        operation="optim_step",
        model_id=request.model_id,
        payload=request.dict(),
        seq_id=request.seq_id,
        task_func=execute_optim_step
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )
