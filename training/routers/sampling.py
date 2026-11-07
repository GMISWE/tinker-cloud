"""
Sampling Router - HTTP Layer for Model Sampling

Endpoints:
- POST /api/v1/asample - Async sampling
- POST /api/v1/sample - Sync sampling
- POST /api/v1/create_sampling_client - Create sampling client
"""

import logging
from typing import Dict, Optional, List, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException

from ..services.sampling_service import SamplingService
from ..core.sample_tasks import start_sample_task
from ..core.task_manager import TaskManager
from ..core.dependencies import (
    verify_api_key_dep,
    get_checkpoint_store,
    get_sampling_service,
    get_futures_storage,
    get_sample_futures,
    get_training_clients,
)
from ..checkpoints import CheckpointRef, CheckpointStore
from ..storage import FuturesStorage, SampleFutureStore
from ..models.requests import (
    ASampleRequest,
    SampleRequest,
    CreateSamplingClientRequest,
    ResolveSamplerTargetRequest,
)
from ..models.responses import (
    AsyncOperationResponse,
    ResolveSamplerTargetResponse,
    SampleResult,
    SamplingSequence,
    CreateSamplingClientResult,
)
from ..utils import generate_request_id
from ..utils.helpers import BASE_MODEL_HEADER, BASE_MODEL_SAMPLING_UNSUPPORTED, MODEL_ID_HEADER, PINNED_VERSION_HEADER

logger = logging.getLogger(__name__)

router = APIRouter()


def get_task_manager(
    futures_storage: FuturesStorage = Depends(get_futures_storage),
) -> TaskManager:
    """Create TaskManager with FuturesStorage dependency."""
    return TaskManager(futures_storage)


# ============================================================================
# Sampling Endpoints
# ============================================================================




def resolve_target_model(
    training_clients: Dict,
    sampling_session_id: Optional[str] = None,
    model_path: Optional[str] = None,
    base_model: Optional[str] = None,
    store: Optional[CheckpointStore] = None,
    model_id_header: Optional[str] = None,
    pinned_version_header: Optional[str] = None,
) -> tuple:
    """Return (model_id, pinned_version) for a sampling request, or raise.

    A sampler names its owning model and the weight version it pinned; the gateway, which
    holds the samplers, puts both in the request headers (`X-Model-Id`, `X-Pinned-Version`).
    A checkpoint model_path names its model through the store. There is no fallback to
    "some model": under a multi-tenant pool that serves a co-tenant's adapter, and a bare
    base_model has no engine behind it.
    """
    if sampling_session_id:
        if not model_id_header:
            raise HTTPException(
                status_code=400,
                detail=f"{MODEL_ID_HEADER} is required with a sampling_session_id: samplers are resolved by the gateway",
            )
        if model_id_header not in training_clients:
            raise HTTPException(
                status_code=404,
                detail=f"Sampler {sampling_session_id}'s model {model_id_header} no longer exists",
            )
        if pinned_version_header is None:
            return model_id_header, None
        try:
            return model_id_header, int(pinned_version_header)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{PINNED_VERSION_HEADER} {pinned_version_header!r} is not an integer") from None
    if model_path:
        # A checkpoint path names its model AND the weight version it was
        # saved at: the sampler is pinned there, not served the live weights.
        ref = CheckpointRef.parse(model_path)
        if store is None:
            raise RuntimeError(
                "resolve_target_model needs the checkpoint store to resolve a model_path"
            )
        store.require(ref)
        if ref.model_id not in training_clients:
            raise HTTPException(
                status_code=404,
                detail=f"Model {ref.model_id!r} from model_path {model_path!r} not found",
            )
        rec = store.get(ref) or {}
        return ref.model_id, rec.get("weight_version")
    if base_model:
        raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
    raise HTTPException(
        status_code=400, detail="Provide sampling_session_id or a tinker:// model_path"
    )


def resolve_sample_target(
    training_clients: Dict,
    sampling_session_id: Optional[str],
    model_path: Optional[str],
    base_model: Optional[str],
    store: CheckpointStore,
    model_id_header: Optional[str],
    pinned_version_header: Optional[str],
    base_model_header: Optional[str],
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """(model_id, pinned_version, None) for a model's sampler or path (resolve_target_model);
    (None, None, base model) for a base-model sampler, which the gateway names in X-Base-Model
    after the engine accepted it at create_sampling_session."""
    if base_model_header is not None:
        if not sampling_session_id or model_id_header is not None:
            raise HTTPException(status_code=400,
                                detail=f"{BASE_MODEL_HEADER} comes with a sampling_session_id and no {MODEL_ID_HEADER}")
        return None, None, base_model_header
    model_id, pinned_version = resolve_target_model(
        training_clients, sampling_session_id=sampling_session_id, model_path=model_path, base_model=base_model,
        store=store, model_id_header=model_id_header, pinned_version_header=pinned_version_header,
    )
    return model_id, pinned_version, None


def check_sample_extensions(request: ASampleRequest, prompt_len: int) -> None:
    """Boundary check of the SDK 0.33 sampling fields; each refusal names the field.
    target_prompt_logprobs and prompt_alt_tokens_k have no backend here, so a
    request that carries them fails now rather than returning a response the
    SDK then rejects for lacking them."""
    if request.target_prompt_logprobs is not None:
        raise HTTPException(status_code=400, detail="target_prompt_logprobs is unsupported on this server")
    if request.prompt_alt_tokens_k > 0:
        raise HTTPException(status_code=400, detail="prompt_alt_tokens_k is unsupported on this server")
    last_n = request.prompt_logprobs_last_n
    if last_n is not None:
        if not request.prompt_logprobs:
            raise HTTPException(status_code=400, detail="prompt_logprobs_last_n requires prompt_logprobs")
        if not 1 <= last_n <= prompt_len - 1:
            raise HTTPException(
                status_code=400,
                detail=f"prompt_logprobs_last_n must be between 1 and len(prompt) - 1 = {prompt_len - 1}, got {last_n}",
            )


def _sequence_ids(request_id: str, n: int) -> List[str]:
    """Identity of each sequence a sampling request will return, fixed at
    submission (SDK >= 0.25 requires one per sequence on the promise)."""
    return [f"{request_id}-{i}" for i in range(n)]


@router.post("/api/v1/asample", response_model=AsyncOperationResponse)
async def asample(
    request: ASampleRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
    x_model_id: Optional[str] = Header(None, alias=MODEL_ID_HEADER),
    x_pinned_version: Optional[str] = Header(None, alias=PINNED_VERSION_HEADER),
    x_base_model: Optional[str] = Header(None, alias=BASE_MODEL_HEADER),
):
    """
    Async sampling.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    # Extract prompt tokens
    prompt_tokens = request.prompt.get_tokens()
    check_sample_extensions(request, len(prompt_tokens))

    # BUG-015: resolve the sampler's pinned weight version (snapshot samplers,
    # e.g. DPO's frozen reference) so pinned logprob reads aren't served from
    # the live refit-every-step engine. The sampler also names its OWNING
    # model — required routing under a multi-tenant pool, where find-first
    # would serve a co-tenant's adapter.
    model_id, pinned_version, target_base = resolve_sample_target(
        training_clients, request.sampling_session_id, request.model_path, request.base_model, store,
        x_model_id, x_pinned_version, x_base_model,
    )

    owner = sample_futures.register(
        request_id,
        "asample",
        model_id,
        request.sampling_session_id,
        request.seq_id,
    )
    if owner != request_id:  # SDK retry of a seq_id it already submitted
        return AsyncOperationResponse(
            request_id=owner,
            model_id=model_id,
            sample_sequence_ids=_sequence_ids(owner, request.num_samples),
        )

    async def execute():
        result_dict = await service.async_sample(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict()
            if request.sampling_params
            else None,
            prompt_logprobs=request.prompt_logprobs,
            training_clients=training_clients,
            pinned_version=pinned_version,
            model_id=model_id,
            base_model=target_base,
            topk_sample_logprobs=request.topk_sample_logprobs,
            topk_prompt_logprobs=request.topk_prompt_logprobs,
            prompt_logprobs_last_n=request.prompt_logprobs_last_n,
        )
        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(
            sequences=sequences,
            prompt_logprobs=result_dict.get("prompt_logprobs"),
            topk_prompt_logprobs=result_dict.get("topk_prompt_logprobs"),
            weight_version=result_dict.get("weight_version"),
            latest_weight_version=result_dict.get("latest_weight_version"),
        )

    start_sample_task(sample_futures, request_id, "asample", execute)
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
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
    x_model_id: Optional[str] = Header(None, alias=MODEL_ID_HEADER),
    x_pinned_version: Optional[str] = Header(None, alias=PINNED_VERSION_HEADER),
    x_base_model: Optional[str] = Header(None, alias=BASE_MODEL_HEADER),
):
    """
    Synchronous sampling.
    Sampling multiple prompts, in memory future store
    """
    request_id = generate_request_id()

    model_id, _, target_base = resolve_sample_target(
        training_clients, request.sampling_session_id, request.model_path, request.base_model, store,
        x_model_id, x_pinned_version, x_base_model,
    )
    n_seq = len(request.prompts) * request.num_samples

    owner = sample_futures.register(
        request_id,
        "sample",
        model_id,
        request.sampling_session_id,
        request.seq_id,
    )
    if owner != request_id:
        return AsyncOperationResponse(
            request_id=owner,
            model_id=model_id,
            sample_sequence_ids=_sequence_ids(owner, n_seq),
        )

    async def execute():
        result_dict = await service.sync_sample(
            request_id=request_id,
            prompts=request.prompts,
            num_samples=request.num_samples,
            sampling_params=request.sampling_params.dict()
            if request.sampling_params
            else None,
            training_clients=training_clients,
            model_id=model_id,
            base_model=target_base,
        )

        sequences = [SamplingSequence(**seq) for seq in result_dict["sequences"]]
        return SampleResult(sequences=sequences)

    start_sample_task(sample_futures, request_id, "sample", execute)
    return AsyncOperationResponse(
        request_id=request_id,
        model_id=model_id,
        sample_sequence_ids=_sequence_ids(request_id, n_seq),
    )


@router.post("/api/v1/resolve_sampler_target", response_model=ResolveSamplerTargetResponse)
async def resolve_sampler_target(
    request: ResolveSamplerTargetRequest,
    _: None = Depends(verify_api_key_dep),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
    service: SamplingService = Depends(get_sampling_service),
):
    """The gateway's question at create_sampling_session: which model does this path name, and at
    which weight version. The store's verdict on the path (404, 425, 400) is the answer. A bare
    base_model is a base-model sampler on a backend that serves that base model; 400 otherwise."""
    if not request.model_path and request.base_model:
        backend = service.backend
        if not backend.serves_base_model_sampling:
            raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
        try:
            await backend.check_base_model(request.base_model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return ResolveSamplerTargetResponse(model_id=None, pinned_version=None, base_model=request.base_model)
    model_id, pinned_version = resolve_target_model(
        training_clients, model_path=request.model_path, base_model=request.base_model, store=store,
    )
    return ResolveSamplerTargetResponse(
        model_id=model_id, pinned_version=pinned_version, base_model=training_clients[model_id]["base_model"],
    )


@router.post("/api/v1/create_sampling_client", response_model=AsyncOperationResponse)
async def create_sampling_client(
    request: CreateSamplingClientRequest,
    _: None = Depends(verify_api_key_dep),
    service: SamplingService = Depends(get_sampling_service),
    task_manager: TaskManager = Depends(get_task_manager),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
    x_model_id: Optional[str] = Header(None, alias=MODEL_ID_HEADER),
    x_pinned_version: Optional[str] = Header(None, alias=PINNED_VERSION_HEADER),
):
    """
    Create sampling client bound to the model named by model_path, or by the sampler's headers.
    This operation is asynchronous - use retrieve_future to check status.
    """
    request_id = generate_request_id()

    model_id, _ = resolve_target_model(
        training_clients,
        sampling_session_id=request.sampling_session_id,
        model_path=request.model_path,
        base_model=request.base_model,
        store=store,
        model_id_header=x_model_id,
        pinned_version_header=x_pinned_version,
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
        task_func=execute,
    )

    return AsyncOperationResponse(request_id=request_id, model_id=model_id)
