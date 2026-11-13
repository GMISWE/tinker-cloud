"""
Task Manager - Eliminates Duplicated Async Pattern

This module provides a reusable pattern for managing background async tasks
with futures storage integration. Every endpoint in api_refactored.py follows
the same pattern:

1. Save future to storage
2. Create async task
3. Try/catch execution
4. Update future status on completion/failure

This manager encapsulates that pattern (DRY principle).
"""
import asyncio
import logging
import time
from typing import Callable, Dict, Any, Optional, Awaitable
from ..storage import FuturesStorage
from .request_log import RequestLine, RequestSize
from ..services import ordering
from ..models.responses import validate_result
from ..proto.response_conv import (
    FORWARD_BACKWARD_OPERATIONS, PROTO_RESULT_OPERATIONS, serialize_result, stored_forward_backward_result,
)

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Manages background async tasks with automatic futures storage integration.

    Eliminates the duplicated try/except/update_status pattern found in every
    endpoint of api_refactored.py.

    Example usage:
        task_manager = TaskManager(futures_storage)

        async def my_business_logic():
            # Do work
            return {"result": "success"}

        request_id = task_manager.create_task(
            operation="train",
            model_id="model_123",
            payload=request.dict(),
            task_func=my_business_logic
        )
    """

    # Process-global task registry. TaskManager is constructed per-request
    # via Depends, so an instance-level dict would make get_task/cancel_task
    # from any OTHER router a silent no-op (the futures router's long-poll
    # needs to find tasks created by the training router).
    _active_tasks: Dict[str, asyncio.Task] = {}

    def __init__(self, futures_storage: FuturesStorage):
        """
        Initialize task manager.

        Args:
            futures_storage: Storage backend for task status tracking
        """
        self.futures_storage = futures_storage

    def create_task(
        self,
        request_id: str,
        operation: str,
        model_id: str,
        payload: Dict[str, Any],
        task_func: Callable[[], Awaitable[Dict[str, Any]]],
        seq_id: Optional[int] = None,
        size: Optional[RequestSize] = None,
    ) -> str:
        """
        Create and track a background async task with automatic error handling.

        This method:
        1. Saves future to storage with "pending" status
        2. Wraps task_func with error handling
        3. Updates storage on completion/failure
        4. Tracks task lifecycle

        Args:
            request_id: Unique identifier for this request
            operation: Operation name (e.g., "forward_backward", "optim_step")
            model_id: Associated model ID
            payload: The request; a resend under the same seq_id is compared by its digest
            task_func: Async function that performs the actual work.
                      Must return Dict[str, Any] on success.
            size: What a pass request carried. With it the finished request gets its
                  log line (core/request_log.py); None for an operation that gets none.

        Example:
            async def do_training():
                result = await service.train(data)
                return {"loss": result.loss, "grad_norm": result.grad_norm}

            task_manager.create_task(
                request_id="req_123",
                operation="train",
                model_id="model_abc",
                payload=request.dict(),
                task_func=do_training
            )
        """
        # Save future with pending status; a retried seq_id returns the earlier request
        submitted = time.monotonic()
        owner = self.futures_storage.save_future(
            request_id=request_id,
            operation=operation,
            payload=payload,
            model_id=model_id,
            seq_id=seq_id,
        )
        if owner != request_id:
            return owner
        kind = ordering.KINDS.get(operation)
        stored = time.monotonic()
        # monotonic readings of the request's work: when it left the queue, when it returned
        work: Dict[str, float] = {}

        async def timed_task():
            work["started"] = time.monotonic()
            try:
                return await task_func()
            finally:
                work["ended"] = time.monotonic()

        def log_request(outcome: str) -> None:
            if size is None:
                return
            now = time.monotonic()
            started, ended = work["started"], work["ended"]
            logger.info(RequestLine(
                request_id=request_id, operation=operation, model_id=model_id, outcome=outcome, size=size,
                queued_s=started - stored, backend_s=ended - started, store_s=(stored - submitted) + (now - ended),
            ).format())

        async def wrapped_task():
            """Wrapper that handles success/failure and storage updates"""
            try:
                logger.info(f"[{request_id}] Starting {operation} for {model_id}")

                # Execute the actual business logic, in the model's program order
                if kind is not None:
                    result = await ordering.queues.run(model_id, kind, timed_task)
                else:
                    result = await timed_task()

                # A result that the operation's response model rejects fails
                # the future here, with the offending field named, instead of
                # failing inside the SDK's parser on the client.
                result = validate_result(operation, result)
                # Sample and forward/forward_backward results have a proto view
                # (SDK >= 0.25 retrieves them only as proto). Build it once here,
                # off the event loop; never rebuild it per poll. A forward or
                # forward_backward result is stored in the form that keeps an
                # array-valued tensor in that view only. A sample view that
                # fails to build is logged and left to the retrieve path, which
                # then answers the proto client with 500 while JSON clients
                # still get the result.
                result_proto = None
                if operation in FORWARD_BACKWARD_OPERATIONS:
                    result, result_proto = await asyncio.to_thread(stored_forward_backward_result, result)
                elif operation in PROTO_RESULT_OPERATIONS:
                    try:
                        result_proto = await asyncio.to_thread(serialize_result, operation, result)
                    except Exception as e:
                        logger.error(f"[{request_id}] proto view of {operation} result not built: {e}")
                self.futures_storage.update_status(
                    request_id=request_id,
                    status="completed",
                    result=result,
                    result_proto=result_proto,
                )

                logger.info(f"[{request_id}] {operation} completed successfully")
                log_request("completed")

            except Exception as e:
                # Log full traceback for debugging
                import traceback
                logger.error(f"[{request_id}] {operation} failed: {e}")
                logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")

                # Update storage with error
                self.futures_storage.update_status(
                    request_id=request_id,
                    status="failed",
                    result={"error": str(e)}
                )
                log_request("failed")

            finally:
                # Cleanup task reference
                if request_id in self._active_tasks:
                    del self._active_tasks[request_id]

        # Create and track the task
        task = asyncio.create_task(wrapped_task())
        self._active_tasks[request_id] = task

        # Prevent garbage collection
        task.add_done_callback(lambda t: None)
        return request_id

    def get_task(self, request_id: str) -> Optional[asyncio.Task]:
        """
        Get active task by request ID.

        Args:
            request_id: Request identifier

        Returns:
            Task if active, None if not found or completed
        """
        return self._active_tasks.get(request_id)

    def cancel_task(self, request_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            request_id: Request identifier

        Returns:
            True if task was cancelled, False if not found
        """
        task = self._active_tasks.get(request_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"[{request_id}] Task cancelled")
            return True
        return False

    async def wait_all(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all active tasks to complete.

        Useful for graceful shutdown.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self._active_tasks:
            return

        tasks = list(self._active_tasks.values())
        logger.info(f"Waiting for {len(tasks)} active tasks to complete...")

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for tasks, {len(self._active_tasks)} still active")

    @classmethod
    def inflight_count(cls) -> int:
        """Number of async operations not yet completed (process-wide)."""
        return len(cls._active_tasks)
