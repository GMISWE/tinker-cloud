"""
Checkpoints Router - HTTP Layer for Checkpoint Management

Endpoints:
- POST /api/v1/save_weights - Save model weights to disk
- POST /api/v1/save_weights_for_sampler - Save weights for SGLang sampler
- POST /api/v1/save_weights_external - Export weights in HF format (external_weights kind)
- POST /api/v1/load_weights - Load a training checkpoint as a model's first request
- GET  /api/v1/training_runs/{model_id}/checkpoints - List
- GET  /api/v1/checkpoints - List every run's checkpoints (paginated)
- DELETE .../checkpoints/{kind}/{name} - Delete one checkpoint of any kind
- PUT  .../checkpoints/{kind}/{name}/ttl - Set or clear a checkpoint's expiry
- GET  .../checkpoints/external_weights/{name}/external_weights_urls - Signed per-file URLs
- GET  /api/v1/external_weights/{model_id}/{name}/{relpath} - Serve one signed file
- POST /api/v1/weights_info - Get weights/checkpoint info from tinker path
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ..services.checkpoint_service import CheckpointService
from ..services.model_service import ModelService
from ..core.task_manager import TaskManager
from ..config import TrainingConfig
from ..core.dependencies import (
    verify_api_key_dep, 
    get_checkpoint_store,
    get_config,
    get_checkpoint_service, 
    get_model_service,
    get_training_runs_metadata,
    get_metadata_storage, 
    get_futures_storage, 
    get_training_clients,
)
from ..checkpoints import CheckpointKind, CheckpointRef, CheckpointStore
from ..storage import MetadataStorage, FuturesStorage
from ..models.requests import (
    LoadWeightsRequest,
    OptimizerConfig,
    SaveWeightsExternalRequest,
    SaveWeightsRequest,
    SaveWeightsForSamplerRequest,
    SetCheckpointTTLRequest,
    WeightsInfoRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    CheckpointsListResponse,
    Cursor,
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
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    task_manager: TaskManager = Depends(get_task_manager),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
):
    """
    Save weights for SGLang sampler.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Check if model exists
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

    if request.kind not in service.backend.SUPPORTED_SAMPLER_KINDS:
        raise HTTPException(status_code=400, detail=f"sampler kind {request.kind!r} is not served by "
                            f"{type(service.backend).__name__}; {', '.join(sorted(service.backend.SUPPORTED_SAMPLER_KINDS))} are")
    inflight = request.kind == "inflight"
    if inflight and (request.sampling_session_seq_id is None or request.path is not None or request.name is not None):
        raise HTTPException(status_code=400, detail="an inflight save names no checkpoint: it publishes the model's "
                            "shared weights to a new sampling session (sampling_session_seq_id, no path)")

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

        # An ephemeral sampler is the gateway's to route: it learns of it before the SDK does.
        # A snapshot's weight version is pinned at save time so pinned logprob reads are not
        # served from the live (refit-every-step) engine; an inflight one follows the live weights.
        sampling_session_id = result.get("sampling_session_id")
        gateway = http_request.app.state.gateway_client
        if sampling_session_id and gateway is not None:   # None: an engine with no gateway, whose samplers nothing routes to
            await gateway.sampler_created(
                sampling_session_id, request.model_id,
                None if inflight else client_info["backend_handle"].weight_version,
                result["uri"], base_model,
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
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    model_service: ModelService = Depends(get_model_service),
    task_manager: TaskManager = Depends(get_task_manager),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
    training_clients: Dict = Depends(get_training_clients),
    training_runs_metadata: Dict = Depends(get_training_runs_metadata),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """Load a saved training checkpoint into a model.

    Permitted only as the model's first request (before any forward /
    forward_backward / optim_step), matching the Tinker service; later loads
    belong in a fresh model created with checkpoint_path. `optimizer=false`
    restores weights only; `optimizer=true` also restores optimizer state and
    fails the future when the checkpoint or backend cannot provide it.
    Session addressing creates the model first (`_create_via_load`).
    """
    try:
        creates = request.creates()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if creates:
        return _create_via_load(request, http_request, service, model_service, task_manager, metadata_storage,
                                training_clients, training_runs_metadata, store)
    assert request.model_id is not None  # creates() is False only with model_id set
    if request.model_id not in training_clients:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    if futures_storage.has_training_requests(request.model_id):
        raise HTTPException(
            status_code=400,
            detail=f"LoadWeights is not permitted with seq_id {request.seq_id}: the model has already "
                   "trained; create a new model and load into it first",
        )
    ref = CheckpointRef.parse(request.path)           # 400 on a malformed path
    store.require(ref, kind=CheckpointKind.WEIGHTS)   # 404 / 425 / 500 / wrong kind, before the future exists
    try:   # one optimizer family per model, for life
        if request.optimizer_config is not None:
            request.optimizer_config.check()
        service.require_load_family(training_clients[request.model_id], ref, request.optimizer, request.optimizer_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
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


def _create_via_load(
    request: LoadWeightsRequest,
    http_request: Request,
    service: CheckpointService,
    model_service: ModelService,
    task_manager: TaskManager,
    metadata_storage: MetadataStorage,
    training_clients: Dict,
    training_runs_metadata: Dict,
    store: CheckpointStore,
) -> AsyncOperationResponse:
    """Create the model `{session_id}:train:{model_seq_id}` from the checkpoint and load
    it, as one future. The SDK derives that id itself and reads no id back. Base model
    and LoRA config are the checkpoint's run's; the optimizer family is the
    checkpoint's when its state is restored, else the declared one (AdamW by default)."""
    assert request.session_id is not None and request.model_seq_id is not None  # creates() checked both
    model_id = f"{request.session_id}:train:{request.model_seq_id}"
    if model_id in training_clients:
        raise HTTPException(status_code=400, detail=f"Model {model_id} already exists: create-via-load creates it")
    ref = CheckpointRef.parse(request.path)           # 400 on a malformed path
    store.require(ref, kind=CheckpointKind.WEIGHTS)   # 404 / 425 / 500 / wrong kind, before the future exists
    source = metadata_storage.load_training_run(ref.model_id)   # written at every create, kept past unload
    if source is None:
        raise HTTPException(status_code=404, detail=f"Training run {ref.model_id} of {ref.uri} not found")
    base_model = source["base_model"]
    if request.base_model is not None and request.base_model != base_model:
        raise HTTPException(status_code=400, detail=f"base_model {request.base_model!r} is not the checkpoint's ({base_model!r})")
    try:
        if request.optimizer:
            optimizer_config = OptimizerConfig(**store.optimizer_config(ref))
            if request.optimizer_config is not None and request.optimizer_config.type != optimizer_config.type:
                raise ValueError(f"load_weights declares optimizer {request.optimizer_config.type} but {ref.uri} "
                                 f"holds {optimizer_config.type} optimizer state; load it with optimizer=false")
        else:
            # undeclared: AdamW, as the SDK documents for a weights-only load (types/load_weights_request.py)
            optimizer_config = request.optimizer_config or OptimizerConfig()
        optimizer_config.check()
        model_service.require_optimizer(optimizer_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    request_id = generate_request_id()

    async def execute():
        try:
            await model_service.create_model(
                model_id=model_id, request_id=request_id, base_model=base_model,
                lora_config=source["lora_config"], parallelism_config=None,
                metadata_storage=metadata_storage, training_clients=training_clients,
                training_runs_metadata=training_runs_metadata, optimizer_config=optimizer_config,
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
        return await service.load_weights(
            model_id=model_id, request_id=request_id, ref=ref, optimizer=request.optimizer,
            training_clients=training_clients, metadata_storage=metadata_storage,
        )

    request_id = task_manager.create_task(
        request_id=request_id, operation="load_weights", model_id=model_id,
        payload=request.dict(), task_func=execute,
    )
    return AsyncOperationResponse(request_id=request_id, model_id=model_id)


@router.get("/api/v1/training_runs/{model_id}/checkpoints", response_model=CheckpointsListResponse)
async def list_checkpoints(
    model_id: str,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """Checkpoints of a training run, unpaginated."""
    if metadata_storage.load_training_run(model_id) is None:
        raise HTTPException(status_code=404, detail=f"Training run not found: {model_id}")
    return CheckpointsListResponse(checkpoints=service.list_checkpoints(model_id), cursor=None)


@router.get("/api/v1/checkpoints", response_model=CheckpointsListResponse)
async def list_user_checkpoints(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
):
    """Every training run's checkpoints, newest first, with a cursor (list_user_checkpoints;
    `tinker checkpoint list` with no run id). Owners are not tracked, so a key sees every
    run's checkpoints."""
    page, total = service.list_all_checkpoints(limit=limit, offset=offset)
    return CheckpointsListResponse(checkpoints=page, cursor=Cursor(offset=offset, limit=limit, total_count=total))


# SDK CheckpointType -> store kind (the inverse of CHECKPOINT_TYPES)
_KIND_OF_TYPE = {t: k for k, t in CHECKPOINT_TYPES.items()}
_TYPES_HELP = "|".join(_KIND_OF_TYPE)
_KIND_VALUES = {k.value for k in CHECKPOINT_TYPES}
_KINDS_HELP = "|".join(sorted(_KIND_VALUES))


def _typed_ref(model_id: str, kind: str, checkpoint_id: str) -> CheckpointRef:
    """The ref a `<kind>/<id>` path segment names; 400 for an unknown kind."""
    if kind not in _KIND_VALUES:
        raise HTTPException(status_code=400, detail=f"checkpoint path must be <{_KINDS_HELP}>/<id>")
    return CheckpointRef.make(model_id, CheckpointKind(kind), checkpoint_id)


def _delete(service: CheckpointService, ref: CheckpointRef) -> Response:
    if not service.delete_checkpoint(ref):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ref.uri}")
    return Response(status_code=204)


@router.delete("/api/v1/training_runs/{model_id}/checkpoints/{kind}/{checkpoint_id}")
async def delete_checkpoint_typed(
    model_id: str, kind: str, checkpoint_id: str,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
    metadata_storage: MetadataStorage = Depends(get_metadata_storage),
):
    """DELETE .../checkpoints/<kind>/<id>, kind in weights|sampler_weights|external_weights."""
    return _delete(service, _typed_ref(model_id, kind, checkpoint_id))


@router.put("/api/v1/training_runs/{model_id}/checkpoints/{kind}/{checkpoint_id}/ttl", status_code=204)
async def set_checkpoint_ttl(
    model_id: str, kind: str, checkpoint_id: str, request: SetCheckpointTTLRequest,
    _: None = Depends(verify_api_key_dep),
    service: CheckpointService = Depends(get_checkpoint_service),
):
    """Expire a checkpoint ttl_seconds from now, or never (null); the background sweep
    deletes it once expired. The SDK sends the checkpoint id `<kind>/<name>` unencoded."""
    ref = _typed_ref(model_id, kind, checkpoint_id)
    if not service.set_ttl(ref, request.ttl_seconds):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ref.uri}")
    return Response(status_code=204)


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
    if checkpoint_type not in _KIND_OF_TYPE:
        raise HTTPException(status_code=400, detail=f"checkpoint_type must be one of {_TYPES_HELP}")
    return _delete(service, CheckpointRef.make(model_id, _KIND_OF_TYPE[checkpoint_type], checkpoint_id))


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
            lora_rank=lora_rank if is_lora else None,
            optimizer_config=store.optimizer_config(ref),
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
            lora_rank=lora_rank if is_lora else None,
            optimizer_config=store.optimizer_config(ref),
        )

    # Model not found anywhere
    logger.warning(f"Model not found: {model_id}")
    raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
