"""
Sampling Router - HTTP Layer for Model Sampling

Endpoints:
- POST /api/v1/asample - Async sampling via SGLang
- POST /api/v1/sample - Sync sampling via SGLang
- POST /api/v1/create_sampling_client - Create SGLang sampling client
"""
import logging
from typing import Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException

from ..services.sampling_service import SamplingService
from ..core.task_manager import TaskManager
from ..core.dependencies import (
    verify_api_key_dep,
    get_checkpoint_store,
    get_sampling_service,
    get_futures_storage,
    get_training_clients,
    get_session_service
)
from ..checkpoints import CheckpointRef, CheckpointStore
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

logger = logging.getLogger(__name__)

router = APIRouter()

def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage)
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Sampling Endpoints
# ============================================================================


BASE_MODEL_SAMPLING_UNSUPPORTED = (
    "base-model sampling is not supported by this deployment; create a model "
    "and a sampler from it (save_weights_and_get_sampling_client) instead"
)


def resolve_target_model(
    training_clients: Dict,
    session_service,
    sampling_session_id: Optional[str] = None,
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
    store: Optional[CheckpointStore] = None,
) -> tuple:
    """Return (model_id, pinned_version) for a sampling request, or raise.

    A sampler names its owning model (registered at save_weights_for_sampler /
    create_sampling_session); a checkpoint model_path names it through the
    store. There is no fallback to "some model": under a multi-tenant pool
    that serves a co-tenant's adapter, and a bare base_model has no engine
    behind it.
    """
    if sampling_session_id:
        info = session_service.get_sampler(sampling_session_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Unknown sampling_session_id: {sampling_session_id}")
        if not info.model_id:
            raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
        if info.model_id not in training_clients:
            raise HTTPException(status_code=404, detail=f"Sampler {sampling_session_id}'s model {info.model_id} no longer exists")
        return info.model_id, info.pinned_version
    if model_path:
        # A checkpoint path names its model AND the weight version it was
        # saved at: the sampler is pinned there, not served the live weights.
        ref = CheckpointRef.parse(model_path)
        if store is None:
            raise RuntimeError("resolve_target_model needs the checkpoint store to resolve a model_path")
        store.require(ref)
        if ref.model_id not in training_clients:
            raise HTTPException(status_code=404, detail=f"Model {ref.model_id!r} from model_path {model_path!r} not found")
        rec = store.get(ref) or {}
        return ref.model_id, rec.get("weight_version")
    if base_model:
        raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
    raise HTTPException(status_code=400, detail="Provide sampling_session_id or a tinker:// model_path")


def _sequence_ids(request_id: str, n: int) -> List[str]:
    """Identity of each sequence a sampling request will return, fixed at
    submission (SDK >= 0.25 requires one per sequence on the promise)."""
    return [f"{request_id}-{i}" for i in range(n)]


@router.post("/api/v1/asample", response_model=AsyncOperationResponse)
async def asample(
    request: ASampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """
    Async sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Extract prompt tokens
    prompt_tokens = request.prompt.get_tokens()

    # BUG-015: resolve the sampler's pinned weight version (snapshot samplers,
    # e.g. DPO's frozen reference) so pinned logprob reads aren't served from
    # the live refit-every-step engine. The sampler also names its OWNING
    # model — required routing under a multi-tenant pool, where find-first
    # would serve a co-tenant's adapter.
    model_id, pinned_version = resolve_target_model(
        training_clients, session_service,
        sampling_session_id=request.sampling_session_id,
        model_path=request.model_path, base_model=request.base_model, store=store,
    )
    target_model_id = model_id

    async def execute():
        result_dict = await service.async_sample(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            prompt_logprobs=request.prompt_logprobs,
            training_clients=training_clients,
            pinned_version=pinned_version,
            model_id=target_model_id,
        )

        # Convert to response model
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(
            sequences=sequences,
            prompt_logprobs=result_dict.get("prompt_logprobs"),
            weight_version=result_dict.get("weight_version"),
            latest_weight_version=result_dict.get("latest_weight_version"),
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
        model_id=model_id,
        sample_sequence_ids=_sequence_ids(request_id, request.num_samples),
    )


@router.post("/api/v1/sample", response_model=AsyncOperationResponse)
async def sample(
    request: SampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """
    Synchronous sampling via SGLang.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    model_id, _ = resolve_target_model(
        training_clients, session_service,
        sampling_session_id=request.sampling_session_id,
        model_path=request.model_path, base_model=request.base_model, store=store,
    )

    async def execute():
        result_dict = await service.sync_sample(
            request_id=request_id,
            prompts=request.prompts,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict() if request.sampling_params else None,
            training_clients=training_clients,
            model_id=model_id,
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
        model_id=model_id,
        sample_sequence_ids=_sequence_ids(request_id, len(request.prompts) * request.num_samples),
    )


@router.post("/api/v1/create_sampling_client", response_model=AsyncOperationResponse)
async def create_sampling_client(
    request: CreateSamplingClientRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    session_service=Depends(get_session_service),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """
    Create sampling client bound to the model named by model_path.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    model_id, _ = resolve_target_model(
        training_clients, session_service,
        model_path=request.model_path, base_model=request.base_model, store=store,
    )

    async def execute():
        result_dict = await service.create_sampling_client(
            request_id=request_id,
            model_path=request.model_path,
            training_clients=training_clients,
            model_id=model_id,
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
