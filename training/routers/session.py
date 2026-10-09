"""
Session Router - Client Session Management

Endpoints:
- POST /api/v1/create_session - Create client session with metadata
- POST /api/v1/session_heartbeat - Keep session alive
- POST /api/v1/create_sampling_session - Create a sampling session
- POST /api/v1/join_sampling_session - Allocate a client id for a cloned SamplingClient
- POST /api/v1/sessions/{session_id}/finish - Mark a session terminal (first-wins)
- GET /api/v1/sessions - List sessions with pagination
- GET /api/v1/sessions/{session_id} - Get session details
- GET /api/v1/samplers/{sampler_id} - Get sampler details
"""
import json
import logging
import uuid
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException

from ..models.requests import (
    CreateSessionRequest,
    FinishSessionRequest,
    JoinSamplingSessionRequest,
    SessionHeartbeatRequest,
    CreateSamplingSessionRequest,
)
from ..models.responses import (
    CreateSessionResponse,
    FinishSessionResponse,
    JoinSamplingSessionResponse,
    SessionHeartbeatResponse,
    CreateSamplingSessionResponse,
    GetSessionResponse,
    ListSessionsResponse,
    GetSamplerResponse,
)
from ..core.dependencies import (
    verify_api_key_dep, 
    get_checkpoint_store,
    get_training_clients, 
    get_session_service,
)
from ..checkpoints import CheckpointStore
from ..services.session_service import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["session"])

@router.post("/api/v1/create_session", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Create a client session - returns session_id for tracking.

    This endpoint is called by the tinker client on initialization
    to establish a session for tracking and metadata purposes.
    """
    session_id = str(uuid.uuid4())

    # Track session in SessionService
    session_service.create_session(
        session_id=session_id,
        tags=request.tags,
        user_metadata=request.user_metadata or {},
        sdk_version=request.sdk_version
    )

    return CreateSessionResponse(
        session_id=session_id,
        info_message=None,
        warning_message=None,
        error_message=None
    )


@router.post("/api/v1/session_heartbeat", response_model=SessionHeartbeatResponse)
async def session_heartbeat(
    request: SessionHeartbeatRequest,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Send a heartbeat to keep the session alive.

    This endpoint is called periodically by the tinker client
    to signal that the session is still active.
    """
    session = session_service.get_session(request.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    # The SDK stops its heartbeat loop on 410 ("session has finished").
    if session.finished_at is not None:
        raise HTTPException(
            status_code=410,
            detail=f"Session finished ({session.finish_reason}): {request.session_id}"
        )
    session_service.heartbeat(request.session_id)

    return SessionHeartbeatResponse()


@router.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
async def create_sampling_session(
    request: CreateSamplingSessionRequest,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service),
    training_clients: Dict = Depends(get_training_clients),
    store: CheckpointStore = Depends(get_checkpoint_store),
):
    """
    Create a sampling session for inference.

    This endpoint is called when the client wants to create a sampler
    that can be reused across multiple sample requests.
    """
    # Check session exists
    if not session_service.session_exists(request.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    # A sampler must name its owning model: a tinker:// model_path does; a bare
    # base_model has no engine behind it.
    from .sampling import resolve_target_model
    model_id, pinned_version = resolve_target_model(
        training_clients, session_service,
        model_path=request.model_path, base_model=request.base_model, store=store,
    )

    # Generate sampling_session_id
    sampling_session_id = (
        f"{request.session_id}_{request.sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
    )

    # Track in SessionService with metadata
    session_service.add_sampling_session(
        session_id=request.session_id,
        sampling_session_id=sampling_session_id,
        base_model=request.base_model or training_clients[model_id].get("base_model"),
        model_path=request.model_path,
        model_id=model_id,
        pinned_version=pinned_version,   # a checkpoint path pins the sampler at its saved version
    )

    logger.info(
        f"Created sampling session {sampling_session_id} "
        f"(parent={request.session_id}, base_model={request.base_model}, "
        f"model_path={request.model_path})"
    )

    return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)


@router.post("/api/v1/join_sampling_session", response_model=JoinSamplingSessionResponse)
async def join_sampling_session(
    request: JoinSamplingSessionRequest,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service),
):
    """Allocate a client id for a new client (unpickled SamplingClient) of an
    existing sampling session. The creator is 0; joiners get 1, 2, ... and
    put their seq_ids in the block 1e9 * client_counter."""
    client_counter = session_service.join_sampling_session(request.sampling_session_id)
    if client_counter is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sampling session not found: {request.sampling_session_id}"
        )
    return JoinSamplingSessionResponse(client_counter=client_counter)


@router.post("/api/v1/sessions/{session_id}/finish", response_model=FinishSessionResponse)
async def finish_session(
    session_id: str,
    request: FinishSessionRequest,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service),
):
    """Mark the session terminal. Every ServiceClient exit sends this.
    First-wins: a second finish is accepted and returns the recorded reason.
    Live models and samplers are untouched; the reaper frees them once
    heartbeats stop (the SDK stops heartbeating after finish)."""
    session = session_service.finish_session(session_id, request.reason.type, request.detail)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    assert session.finished_at is not None and session.finish_reason is not None
    return FinishSessionResponse(
        session_id=session_id,
        reason=session.finish_reason,
        detail=session.finish_detail,
        finished_at=session.finished_at.isoformat(),
    )


# ============================================================================
# Session Query Endpoints
# ============================================================================

@router.get("/api/v1/sessions", response_model=ListSessionsResponse)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service)
):
    """
    List sessions with pagination.

    Args:
        limit: Maximum number of sessions to return (default 20)
        offset: Number of sessions to skip (default 0)
    """
    sessions = session_service.list_sessions(limit=limit, offset=offset)
    return ListSessionsResponse(sessions=sessions)


@router.get("/api/v1/sessions/{session_id}", response_model=GetSessionResponse)
async def get_session(
    session_id: str,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get session details including training runs and samplers.

    Args:
        session_id: The session ID to get information for
    """
    session = session_service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    # The SDK validates dict[str, str] strictly; create_session accepted any JSON value.
    return GetSessionResponse(
        training_run_ids=session.model_ids,
        sampler_ids=session.sampling_session_ids,
        user_metadata={k: _metadata_str(v) for k, v in session.user_metadata.items()},
    )


def _metadata_str(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


@router.get("/api/v1/samplers/{sampler_id}", response_model=GetSamplerResponse)
async def get_sampler(
    sampler_id: str,
    _: None = Depends(verify_api_key_dep),
    session_service: SessionService = Depends(get_session_service)
):
    """
    Get sampler details.

    Args:
        sampler_id: The sampler ID (sampling_session_id) to get information for
    """
    sampler = session_service.get_sampler(sampler_id)
    if sampler is None:
        raise HTTPException(
            status_code=404,
            detail=f"Sampler not found: {sampler_id}"
        )

    return GetSamplerResponse(
        sampler_id=sampler.sampler_id,
        base_model=sampler.base_model,
        model_path=sampler.model_path
    )
