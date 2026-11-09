"""
Models Router - HTTP Layer for Model Management

Endpoints:
- POST /api/v1/create_model - Create training model with Ray actors
- POST /api/v1/delete_model - Delete model and free GPU resources
- POST /api/v1/unload_model - Unload model (Tinker SDK compatible alias for delete_model)
- POST /api/v1/get_info - Get model metadata for tokenizer
- GET /api/v1/get_tokenizer - Get tokenizer information
- GET /api/v1/training_runs - List training runs (paginated)
- GET /api/v1/training_runs/{model_id} - Get training run metadata
"""
import logging
from typing import Dict, Literal, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

from ..services.checkpoint_service import CheckpointService
from ..services.model_service import ModelService
from ..core.task_manager import TaskManager
from ..core.dependencies import (
    verify_api_key_dep,
    get_checkpoint_service,
    get_model_service, 
    get_metadata_storage,
    get_futures_storage, 
    get_training_clients,
    get_training_runs_metadata, 
)
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    CreateModelRequest,
    DeleteModelRequest,
    UnloadModelRequest,
    GetInfoRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    DeleteModelResponse,
    UnloadModelResponse,
    GetInfoResponse,
    ModelData,
    Cursor,
    TrainingRunResponse,
    TrainingRunsResponse,
    TokenizerInfo,
    SpecialTokens,
)
from ..utils import generate_request_id
from ..utils.helpers import MODEL_ID_HEADER, engine_name, engine_name_of

logger = logging.getLogger(__name__)

router = APIRouter()

def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.post("/api/v1/create_model", response_model=AsyncOperationResponse)
async def create_model(
    request: CreateModelRequest,
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
    training_runs_metadata: Dict = Depends(get_training_runs_metadata),
    x_model_id: Optional[str] = Header(None, alias=MODEL_ID_HEADER),
):
    """
    Create a new training model with Ray actors and GPU resources.
    This operation is asynchronous - use retrieve_future to check status.

    The gateway mints the model id and records the model under its session before the
    request reaches here; the engine creates the model under that id.
    """
    request_id = generate_request_id()
    if not x_model_id:
        raise HTTPException(status_code=400, detail=f"{MODEL_ID_HEADER} is required: model ids are minted by the gateway")
    if engine_name_of(x_model_id) != engine_name():
        raise HTTPException(status_code=400, detail=f"{MODEL_ID_HEADER} {x_model_id!r} does not name this engine ({engine_name()})")
    model_id = x_model_id

    # The family is fixed for the model's life; a family the
    # backend does not serve fails here, not by training under another one.
    try:
        request.optimizer_config.check()
        service.require_optimizer(request.optimizer_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def execute():
        try:
            return await service.create_model(
                model_id=model_id,
                request_id=request_id,
                base_model=request.base_model,
                lora_config=request.lora_config.dict() if request.lora_config else None,
                parallelism_config=request.parallelism_config.dict() if request.parallelism_config else None,
                metadata_storage=metadata_storage,
                training_clients=training_clients,
                training_runs_metadata=training_runs_metadata,
                optimizer_config=request.optimizer_config,
            )
        except Exception:
            # the gateway recorded the model at submit; a create that fails is reported before the future is
            gateway = http_request.app.state.gateway_client
            if gateway is not None:   # None: an engine with no gateway
                try:
                    await gateway.model_failed(model_id)
                except httpx.HTTPError:   # the create's own error is the one to report; the row's fate is logged beside it
                    logger.exception("Gateway did not drop the failed model %s", model_id)
            raise

    # Create async task
    task_manager.create_task(
        request_id=request_id,
        operation="create_model",
        model_id=model_id,
        payload=request.dict(),
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/delete_model", response_model=DeleteModelResponse)
async def delete_model(
    request: DeleteModelRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
):
    """Delete training client and release GPU resources."""
    try:
        result = await service.delete_model(
            model_id=request.model_id,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )
        return DeleteModelResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/api/v1/unload_model", response_model=AsyncOperationResponse)
async def unload_model(
    request: UnloadModelRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
):
    """Unload model and release GPU resources (Tinker SDK compatible).

    This endpoint is the Tinker-standard way to release model resources.
    Returns an async operation that can be polled via retrieve_future.
    The final result is an UnloadModelResponse with model_id and type fields.
    """
    request_id = generate_request_id()
    model_id = request.model_id

    async def execute():
        await service.delete_model(
            model_id=model_id,
            training_clients=training_clients,
            metadata_storage=metadata_storage
        )
        # Return UnloadModelResponse format for retrieve_future
        return UnloadModelResponse(model_id=model_id).dict()

    # Create async task
    request_id = task_manager.create_task(
        request_id=request_id,
        operation="unload_model",
        model_id=model_id,
        payload=request.dict(),
        seq_id=request.seq_id,
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id
    )


@router.post("/api/v1/get_info", response_model=GetInfoResponse)
async def get_info(
    request: GetInfoRequest,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    training_clients: Dict = Depends(get_training_clients)
):
    """Get model info for tokenizer initialization."""
    try:
        result = service.get_model_info(
            model_id=request.model_id,
            training_clients=training_clients
        )
        return GetInfoResponse(
            model_id=result["model_id"],
            model_data=ModelData(**result["model_data"]),
            is_lora=result["is_lora"],
            lora_rank=result["lora_rank"],
            model_name=result["model_name"],
            optimizer_config=result["optimizer_config"],
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/v1/get_tokenizer")
async def get_tokenizer(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    training_clients: Dict = Depends(get_training_clients)
):
    """Get tokenizer information."""
    try:
        result = service.get_tokenizer_info(
            model_id=model_id,
            training_clients=training_clients
        )
        return TokenizerInfo(
            vocab_size=result["vocab_size"],
            model_max_length=result["model_max_length"],
            pad_token_id=result["pad_token_id"],
            eos_token_id=result["eos_token_id"],
            bos_token_id=result["bos_token_id"],
            special_tokens=SpecialTokens(**result["special_tokens"]),
            hf_checkpoint=result["hf_checkpoint"]
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tokenizer: {str(e)}") from e


def _training_run(metadata: Dict, checkpoints: CheckpointService) -> TrainingRunResponse:
    """A stored run as the SDK's TrainingRun; its last checkpoints come from the store, so a
    deleted or expired checkpoint is never reported as the one to resume from."""
    return TrainingRunResponse(**{**metadata, **checkpoints.latest_checkpoints(metadata["training_run_id"])})


@router.get("/api/v1/training_runs", response_model=TrainingRunsResponse)
async def list_training_runs(
    limit: int = Query(20, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    access_scope: Literal["owned", "accessible"] = "owned",
    project_id: Optional[str] = None,
    _: None = Depends(verify_api_key_dep),
    checkpoints: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """Every training run this server has recorded, live or deleted (a deleted run's
    checkpoints still resume), newest first by creation so offsets stay stable while runs
    train. Owners and projects are not tracked: every key sees every run, so access_scope
    and project_id do not filter (as on the billing usage route)."""
    runs = metadata_storage.list_training_runs(limit=10**9)
    runs.sort(key=lambda m: (m.get("created_at") or "", m["training_run_id"]), reverse=True)
    page = [_training_run(m, checkpoints) for m in runs[offset:offset + limit]]
    return TrainingRunsResponse(
        training_runs=page, cursor=Cursor(offset=offset, limit=limit, total_count=len(runs)),
    )


@router.get("/api/v1/training_runs/{model_id}", response_model=TrainingRunResponse)
async def get_training_run(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: ModelService = Depends(get_model_service),
    checkpoints: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage)
):
    """Get persistent training run metadata."""
    try:
        metadata = service.get_training_run_metadata(
            model_id=model_id,
            metadata_storage=metadata_storage
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _training_run(metadata, checkpoints)
