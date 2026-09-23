"""
Futures Router - Async Operation Result Retrieval

Thin HTTP layer for:
1. Retrieving async operation results
2. Cleaning up old futures
3. No business logic - just storage access
"""
import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from typing import Dict, Any

from ..models.requests import (
    CancelFutureRequest,
    CleanupFuturesRequest,
    SessionFuturesPollRequest,
    RetrieveFutureRequest,

)
from ..models.responses import CleanupResult, SessionFutureCompletion, SessionFuturesPollResponse
from ..storage import FuturesStorage, SampleFutureStore
from ..storage.sample_futures import SampleFuture
from ..core.dependencies import (
    verify_api_key_dep,
    get_futures_storage, 
    get_poll_tracking,
    get_sample_futures,
    get_session_service,
)
from ..core.task_manager import TaskManager
from ..proto.wire import PROTO_CONTENT_TYPE, PROTO_RESULT_OPERATIONS, serialize_result

# Hold a pending retrieve on the operation's task instead of answering 408
# immediately. The SDK's retrieve cycle costs ~450 ms client-side, so an
# immediate 408 turns completion discovery into a half-cycle latency tax
# (~0.2-0.3 s per training step, measured 2026-08-20); the SDK's HTTP timeout
# for retrieve is 300 s, so a held response is transparent to it, and 408
# after the hold window keeps the protocol unchanged.
LONG_POLL_HOLD_S = 30.0

# /retrieve_furtures hold: below the SDK poller's 45 s HTTP timeout, so a held
# reqponse is alwarys delivered rather than timed out and reissued.
RETRIEVE_FUTURES_HOLD_S = 30.0
RETRIEVE_FUTURES_MAX_HOLD_S = 40.0

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    tags=["futures"]
)


async def _completed_response(fut: Dict[str, Any], accept: str, futures_storage: FuturesStorage) -> Response:
    """A completed future's result: proto bytes when the client accepts proto
    and the operation has that view (SDK >= 0.25 rejects JSON for sample and
    forward/forward_backward results), else the JSON of record."""
    if fut["operation"] in PROTO_RESULT_OPERATIONS and PROTO_CONTENT_TYPE in accept.lower():
        blob = fut.get("result_proto")
        if blob is None:
            # Not built at completion (see TaskManager); build it now, once.
            try:
                blob = await asyncio.to_thread(serialize_result, fut["operation"], fut.get("result") or {})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"result has no proto encoding: {e}")
            futures_storage.set_result_proto(fut["request_id"], blob)
        return Response(content=blob, media_type=PROTO_CONTENT_TYPE)
    return JSONResponse(content=fut.get("result", {}))


async def _sample_response(fut: SampleFuture, accept: str, sample_futures: SampleFutureStore) -> Response:
    """Terminal in-memory sample future -> 200 (proto when accepted) or 400 """
    if fut.status == "failed":
        raise HTTPException(status_code=400, detail=fut.error or "Operation failed")
    sample_futures.mark_retrieved(fut.request_id)
    if PROTO_CONTENT_TYPE in accept.lower() and fut.result_proto is not None:
        return Response(content = fut.result_proto, media_type = PROTO_CONTENT_TYPE)
    return JSONResponse(content = fut.result)

@router.post("/api/v1/retrieve_future/{request_id}")
async def retrieve_future(
    request_id: str,
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    poll_tracking: Dict[str, Dict[str, Any]] = Depends(get_poll_tracking),
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
):
    """
    Retrieve async operation result.

    Returns:
    - 408 (Request Timeout) if operation is still running
    - 200 with result if completed successfully (proto when `Accept:
      application/x-protobuf` and the operation has a proto view, else JSON)
    - 400 if the operation terminally failed (the SDK retries 408 and every
      5xx indefinitely, so a failed future MUST return 4xx or clients hang)
    """
    # Smart logging for polling operations
    if request_id not in poll_tracking:
        poll_tracking[request_id] = {
            "start_time": time.time(),
            "count": 0
        }
        logger.info(f"[retrieve_future] Started polling for {request_id}")

    poll_tracking[request_id]["count"] += 1
    poll_count = poll_tracking[request_id]["count"]

    # Log every 10th poll at INFO, others at DEBUG
    if poll_count % 10 == 0:
        logger.info(f"[retrieve_future] Still polling {request_id} (#{poll_count})")
    else:
        logger.debug(f"[retrieve_future] Poll #{poll_count} for {request_id}")

    accept = http_request.headers.get("accept", "")

    # Sample futures live in memory (storage.sample_futures); 
    # a pending one is held on its completion event, never on the task,
    # so a cancelled task cannot surface here as CancelledError.
    sf = sample_futures.get(request_id)
    if sf is not None:
        if sf.status == "pending":
            try:
                await asyncio.wait_for(sf.done.wait(), timeout=LONG_POLL_HOLD_S)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Operation still in progress")
        poll_tracking.pop(request_id, None)
        return await _sample_response(sf, accept, sample_futures)

    # Get future from storage
    future = futures_storage.get_future(request_id)

    if not future:
        raise HTTPException(status_code=404, detail=f"Future {request_id} not found")

    async def _respond(fut: Dict[str, Any]):
        """Terminal-status dispatch (completed -> 200, failed -> 400)."""
        if fut["status"] == "completed":
            if request_id in poll_tracking:
                stats = poll_tracking.pop(request_id)
                duration = time.time() - stats["start_time"]
                logger.info(
                    f"[retrieve_future] {request_id} completed: "
                    f"{stats['count']} polls over {duration:.2f}s"
                )
            return await _completed_response(fut, accept, futures_storage)

        if request_id in poll_tracking:
            stats = poll_tracking.pop(request_id)
            duration = time.time() - stats["start_time"]
            logger.info(
                f"[retrieve_future] {request_id} failed: "
                f"{stats['count']} polls over {duration:.2f}s"
            )
        # Extract error message
        error = None
        result = fut.get("result")
        if result and isinstance(result, dict) and "error" in result:
            error = result["error"]
        elif "error" in fut:
            error = fut["error"]
        # Terminal failure -> 4xx: the SDK treats 408 and all 5xx as
        # retryable, so 500 here turns a dead op into an infinite client
        # poll loop (observed 2026-07-31, G6 blocker probe).
        raise HTTPException(status_code=400, detail=error or "Operation failed")

    if future["status"] != "pending":
        return await _respond(future)

    # Pending: long-poll on the operation's task, then re-check once. The
    # registry is class-level, so a fresh TaskManager sees tasks created by
    # the training router. get_task returning None means the task finished
    # (or predates a restart) — the re-read below covers that race.
    task = TaskManager(futures_storage).get_task(request_id)
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=LONG_POLL_HOLD_S)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass  # task errors are recorded in storage as status=failed
    refreshed = futures_storage.get_future(request_id)
    if refreshed and refreshed["status"] != "pending":
        return await _respond(refreshed)
    raise HTTPException(status_code=408, detail="Operation still in progress")


@router.post("/api/v1/retrieve_future")
async def retrieve_future_body(
    request: RetrieveFutureRequest,
    http_request: Request,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
    poll_tracking: Dict[str, Dict[str, Any]] = Depends(get_poll_tracking),
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
):
    """Retrieve future by request body (the form the SDK uses); same contract as the path form."""
    return await retrieve_future(
        request.request_id,
        http_request,
        _,
        futures_storage,
        poll_tracking,
        sample_futures,
    )

@router.post("/api/v1/retrieve_futures", response_model=SessionFuturesPollResponse, response_model_exclude_none=True)
async def retrieve_futures(
    request: SessionFuturesPollRequest,
    _: None = Depends(verify_api_key_dep),
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
    session_service = Depends(get_session_service),
):
    """Per session completion poll (SDK >= 0.25 with sample_use_retrieve_futures).
    
    Returns every sample of the (sampling session, cloned sampler) that finished
    or failed at or after `prev_cursor`, holding until the first one arrives or the
    hold expires. Entries below `prev_cursor` are treated as acked and dropped. 
    The result payload is fetched seperately with retrieve_future.
    """
    target = request.target 
    if session_service.get_sampler(target.sampling_session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown sampling_session_id: {target.sampling_session_id}")
    hold = RETRIEVE_FUTURES_HOLD_S if request.timeout is None else request.timeout
    entries, cursor = await sample_futures.poll(
        (target.sampling_session_id, target.cloned_sampler_id),
        request.prev_cursor,
        min(hold, RETRIEVE_FUTURES_MAX_HOLD_S),
    )
    return SessionFuturesPollResponse(
        completions=[
            SessionFutureCompletion(state=state, request_id=rid, response_payload_uncompressed_size=size)
            for rid, state, size in entries
        ],
        cursor=cursor,
    )
    
@router.post("/api/v1/cancel_future", status_code=204)
async def cancel_future(
    request: CancelFutureRequest,
    _: None = Depends(verify_api_key_dep),
    sample_futures: SampleFutureStore = Depends(get_sample_futures),
):
    """ Cancel an in-flight sample the SDK has abandoned. Only sample futures
    are cancellable; an already-terminal one is a no op, an unknown id 404. """
    try:
        cancelled = sample_futures.cancel(request.request_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Sample future {request.request_id} is not found") from None
    logger.info("[%s] cancel_future: %s", request.request_id, "cancelled" if cancelled else "already terminal")
    return Response(status_code=204)



@router.post("/api/v1/cleanup_futures", response_model=CleanupResult)
async def cleanup_futures(
    request: CleanupFuturesRequest,
    _: None = Depends(verify_api_key_dep),
    futures_storage: FuturesStorage = Depends(get_futures_storage),
):
    """
    Cleanup old futures - refactored with storage abstraction
    """
    try:
        total_removed = futures_storage.cleanup_old_futures(
            max_age_hours=request.max_age_hours
        )
        logger.info(f"Cleaned up {total_removed} old futures")

        return CleanupResult(
            futures_cleaned=total_removed,
            message=f"Successfully cleaned up {total_removed} futures older than {request.max_age_hours} hours"
        )

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
