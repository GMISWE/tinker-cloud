"""
Checkpoints Router - HTTP Layer for Checkpoint Management

Endpoints:
- POST /api/v1/save_weights - Save model weights to disk
- POST /api/v1/save_weights_for_sampler - Save weights for SGLang sampler
- POST /api/v1/save_weights_external - Export weights in HF format (external_weights kind)
- POST /api/v1/load_weights - Load a training checkpoint as a model's first request
- GET  /api/v1/training_runs/{model_id}/checkpoints - List
- DELETE .../checkpoints/{kind}/{name} - Delete one checkpoint of any kind
- GET  .../checkpoints/external_weights/{name}/external_weights_urls - Signed per-file URLs
- GET  /api/v1/external_weights/{model_id}/{name}/{relpath} - Serve one signed file
- POST /api/v1/weights_info - Get weights/checkpoint info from tinker path
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from ..services.checkpoint_service import CheckpointService
from ..services.session_service import SessionService
from ..core.task_manager import TaskManager
from ..config import TrainingConfig
from ..core.dependencies import (
    verify_api_key_dep, 
    get_checkpoint_store,
    get_config,
    get_checkpoint_service, 
    get_metadata_storage, 
    get_futures_storage, 
    get_training_clients,
    get_session_service,
)
from ..checkpoints import CheckpointKind, CheckpointRef, CheckpointStore
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    SERVED_OPTIMIZER,
    LoadWeightsRequest,
    SaveWeightsExternalRequest,
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
from ..services.checkpoint_service import CHECKPOINT_TYPES
from ..utils import generate_request_id
from ..utils import signed_urls

logger = logging.getLogger(__name__)

router = APIRouter()

def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)

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


@router.post("/api/v1/save_weights_external", response_model=AsyncOperationResponse)
async def save_weights_external(
    request: SaveWeightsExternalRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
):
    """Export the model's weights in HF format as an external_weights checkpoint.

    Async like save_weights; the future's result is the SDK's
    SaveWeightsExternalResponse (path, size_bytes). Encryption is not offered
    on this server, so a non-empty age_encryption_recipients is refused here.
    ttl_seconds is recorded as the checkpoint's expires_at (listing shows it;
    reads refuse an expired checkpoint; nothing reaps the bytes yet).
    """
    if request.age_encryption_recipients:
        raise HTTPException(status_code=400, detail="encryption not supported on this server: "
                            "age_encryption_recipients must be empty")
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    request_id = generate_request_id()

    async def execute():
        return await service.save_weights_external(
            model_id=request.model_id, request_id=request_id, name=request.path,
            ttl_seconds=request.ttl_seconds, training_clients=training_clients,
        )

    request_id = task_manager.create_task(
        request_id=request_id, operation="save_weights_external", model_id=request.model_id,
        payload=request.dict(), seq_id=request.seq_id, task_func=execute,
    )
    return AsyncOperationResponse(request_id=request_id, model_id=request.model_id)


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
                   "trained; create a new model and load into it first",
        )
    if request.optimizer_config is not None and request.optimizer_config.type != SERVED_OPTIMIZER:
        raise HTTPException(
            status_code=400,
            detail=f"optimizer_config.type {request.optimizer_config.type!r} is not supported "
                   f"on this server; only {SERVED_OPTIMIZER!r} is",
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


# SDK CheckpointType -> store kind (the inverse of CHECKPOINT_TYPES)
_KIND_OF_TYPE = {t: k for k, t in CHECKPOINT_TYPES.items()}
_TYPES_HELP = "|".join(_KIND_OF_TYPE)
_KIND_VALUES = {k.value for k in CHECKPOINT_TYPES}
_KINDS_HELP = "|".join(sorted(_KIND_VALUES))


def _delete(service, model_id, checkpoint_type, checkpoint_id):
    if checkpoint_type not in _KIND_OF_TYPE:
        raise HTTPException(status_code=400, detail=f"checkpoint_type must be one of {_TYPES_HELP}")
    ref = CheckpointRef.make(model_id, _KIND_OF_TYPE[checkpoint_type], checkpoint_id)
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
    """DELETE .../checkpoints/<kind>/<id>, kind in weights|sampler_weights|external_weights."""
    if kind not in _KIND_VALUES:
        raise HTTPException(status_code=400, detail=f"checkpoint path must be <{_KINDS_HELP}>/<id>")
    return _delete(service, model_id, CHECKPOINT_TYPES[CheckpointKind(kind)], checkpoint_id)


@router.delete("/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}")
async def delete_checkpoint_bare(
    model_id: str, checkpoint_id: str, checkpoint_type: str = None,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """A bare id needs ?checkpoint_type=<type>: the kinds can share an id."""
    if not checkpoint_type:
        raise HTTPException(status_code=400, detail=f"specify the kind: .../checkpoints/<{_KINDS_HELP}>/<id> "
                            f"or ?checkpoint_type=<{_TYPES_HELP}>")
    return _delete(service, model_id, checkpoint_type, checkpoint_id)


# ============================================================================
# External weights: signed per-file download URLs, served by this server
# ============================================================================

def _public_base(request: Request, config: TrainingConfig) -> str:
    return config.external_weights.public_base_url or str(request.base_url).rstrip("/")


@router.get("/api/v1/training_runs/{model_id}/checkpoints/external_weights/{name}/external_weights_urls")
async def external_weights_urls(
    model_id: str, name: str, request: Request,
    _: None = Depends(verify_api_key_dep),
    store: CheckpointStore = Depends(get_checkpoint_store),
    config: TrainingConfig = Depends(get_config),
):
    """One signed URL per file of a completed external_weights checkpoint
    (SDK ExternalWeightsUrlsResponse). The SDK sends the checkpoint id
    `external_weights/<name>` unencoded, hence the literal segment above."""
    ref = CheckpointRef.make(model_id, CheckpointKind.EXTERNAL_WEIGHTS, name)
    files = store.files(ref, kind=CheckpointKind.EXTERNAL_WEIGHTS)  # 404 / 425 / 500 via CheckpointError
    exp = int(time.time()) + config.external_weights.url_ttl_s
    key = config.external_weights.url_signing_key
    base = _public_base(request, config)
    urls = {
        relpath: f"{base}/api/v1/external_weights/{model_id}/{name}/{relpath}"
                 f"?exp={exp}&sig={signed_urls.sign(key, model_id, name, relpath, exp)}"
        for relpath in files
    }
    return {"urls": urls, "expires": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()}


@router.get("/api/v1/external_weights/{model_id}/{name}/{relpath:path}")
async def download_external_weights_file(
    model_id: str, name: str, relpath: str, exp: int, sig: str,
    store: CheckpointStore = Depends(get_checkpoint_store),
    config: TrainingConfig = Depends(get_config),
):
    """Serve one file of an external_weights checkpoint. The signed URL is the
    credential (no API key); a bad or expired signature is 404, like an
    object store's presigned GET."""
    if not signed_urls.verify(config.external_weights.url_signing_key, model_id, name, relpath, exp, sig):
        raise HTTPException(status_code=404, detail="invalid or expired download URL")
    ref = CheckpointRef.make(model_id, CheckpointKind.EXTERNAL_WEIGHTS, name)
    files = store.files(ref, kind=CheckpointKind.EXTERNAL_WEIGHTS)
    if relpath not in files:
        raise HTTPException(status_code=404, detail=f"no file {relpath!r} in {ref.uri}")
    return FileResponse(files[relpath], filename=files[relpath].name, media_type="application/octet-stream")


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
