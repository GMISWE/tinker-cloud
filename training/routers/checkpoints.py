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
from ..core.dependencies import verify_api_key_dep, get_checkpoint_store
from ..checkpoints import CheckpointKind, CheckpointRef, CheckpointStore
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    LoadWeightsRequest,
    SaveWeightsRequest,
    SaveWeightsForSamplerRequest,
    WeightsInfoRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    SaveWeightsForSamplerResult,
    WeightsInfoResponse,
)
from fastapi import Response
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
        )

    # Create async task
    request_id = task_manager.create_task(
        request_id=request_id,
        operation="save_weights",
        model_id=request.model_id,
        payload=request.dict(),
        seq_id=request.seq_id,
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
            path=request.path,
            sampling_session_seq_id=request.sampling_session_seq_id
        )

        # Register ephemeral sampler with session if sampling_session_id was created
        sampling_session_id = result.get("sampling_session_id")
        if sampling_session_id:
            # BUG-015: pin the weight version at save time so pinned logprob
            # reads are not served from the live (refit-every-step) engine.
            pinned_version = client_info["backend_handle"].weight_version
            session_service.register_ephemeral_sampler(
                sampler_id=sampling_session_id,
                model_id=request.model_id,
                base_model=base_model,
                model_path=result.get("uri"),
                pinned_version=pinned_version,
            )

        return SaveWeightsForSamplerResult(**result)

    # Create async task
    request_id = task_manager.create_task(
        request_id=request_id,
        operation="save_weights_for_sampler",
        model_id=request.model_id,
        payload=request.dict(),
        seq_id=request.seq_id,
        task_func=execute
    )

    return AsyncOperationResponse(
        request_id=request_id,
        model_id=request.model_id
    )


@router.post("/api/v1/load_weights", response_model=AsyncOperationResponse)
async def load_weights(
    request: LoadWeightsRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """Load a saved training checkpoint into a model.

    Permitted only as the model's first request (before any forward /
    forward_backward / optim_step), matching the Tinker service; later loads
    belong in a fresh model created with checkpoint_path. `optimizer=false`
    restores weights only; `optimizer=true` also restores optimizer state and
    fails the future when the checkpoint or backend cannot provide it.
    """
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    if futures_storage.has_training_requests(request.model_id):
        raise HTTPException(
            status_code=400,
            detail=f"LoadWeights is not permitted with seq_id {request.seq_id}: the model has already "
                   "trained; create a new model with checkpoint_path instead",
        )
    ref = CheckpointRef.parse(request.path)           # 400 on a malformed path
    store.require(ref, kind=CheckpointKind.WEIGHTS)   # 404 / 425 / 500 / wrong kind, before the future exists
    request_id = generate_request_id()

    async def execute():
        return await service.load_weights(
            model_id=request.model_id, request_id=request_id, ref=ref,
            optimizer=request.optimizer,
            training_clients=training_clients, metadata_storage=metadata_storage,
        )

    request_id = task_manager.create_task(
        request_id=request_id, operation="load_weights", model_id=request.model_id,
        payload=request.dict(),
        seq_id=request.seq_id, task_func=execute,
    )
    return AsyncOperationResponse(request_id=request_id, model_id=request.model_id)


@router.get("/api/v1/training_runs/{model_id}/checkpoints")
async def list_checkpoints(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """Checkpoints of a training run, in the SDK's CheckpointsListResponse shape."""
    if metadata_storage.load_training_run(model_id) is None:
        raise HTTPException(status_code=404, detail=f"Training run not found: {model_id}")
    return {"checkpoints": service.list_checkpoints(model_id), "cursor": None}


def _delete(service, model_id, checkpoint_type, checkpoint_id):
    kinds = {"training": CheckpointKind.WEIGHTS, "sampler": CheckpointKind.SAMPLER_WEIGHTS}
    if checkpoint_type not in kinds:
        raise HTTPException(status_code=400, detail="checkpoint_type must be 'training' or 'sampler'")
    ref = CheckpointRef.make(model_id, kinds[checkpoint_type], checkpoint_id)
    if not service.delete_checkpoint(ref):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_type} {checkpoint_id} of {model_id}")
    return Response(status_code=204)


@router.delete("/api/v1/training_runs/{model_id}/checkpoints/{kind}/{checkpoint_id}")
async def delete_checkpoint_typed(
    model_id: str, kind: str, checkpoint_id: str,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """DELETE .../checkpoints/weights/<id> or .../checkpoints/sampler_weights/<id>."""
    kinds = {"weights": "training", "sampler_weights": "sampler"}
    if kind not in kinds:
        raise HTTPException(status_code=400, detail="checkpoint path must be weights/<id> or sampler_weights/<id>")
    return _delete(service, model_id, kinds[kind], checkpoint_id)


@router.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint_bare(
    model_id: str, checkpoint_id: str, checkpoint_type: str = None,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """A bare id needs ?checkpoint_type=training|sampler: the two kinds can share an id."""
    if not checkpoint_type:
        raise HTTPException(status_code=400, detail="specify the kind: .../checkpoints/weights/<id>, "
                            ".../checkpoints/sampler_weights/<id>, or ?checkpoint_type=training|sampler")
    return _delete(service, model_id, checkpoint_type, checkpoint_id)


@router.post("/api/v1/weights_info", response_model=WeightsInfoResponse)
async def weights_info(
    request: WeightsInfoRequest,
    _: None = Depends(verify_api_key_dep),
    training_clients: Dict = Depends(get_training_clients),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """
    Get weights/checkpoint info from tinker path.
    Used for loading checkpoints via create_training_client_from_state.

    The path must name a completed checkpoint of either kind (400 malformed,
    404 unknown, 425 still being written); the model info comes from the live
    client or the stored training run.
    """
    tinker_path = request.tinker_path
    logger.info(f"weights_info request for: {tinker_path}")
    ref = CheckpointRef.parse(tinker_path)
    store.require(ref)
    model_id = ref.model_id

    # Try to find model in active training clients first
    if model_id in training_clients:
        client_info = training_clients[model_id]
        base_model = client_info.get("base_model", "")
        lora_rank = int((client_info.get("lora_config") or {}).get("rank") or 0)
        is_lora = lora_rank > 0
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
        logger.info(f"Found stored metadata: base_model={base_model}, is_lora={is_lora}, lora_rank={lora_rank}")
        return WeightsInfoResponse(
            base_model=base_model,
            is_lora=is_lora,
            lora_rank=lora_rank if is_lora else None
        )

    # Model not found anywhere
    logger.warning(f"Model not found: {model_id}")
    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
