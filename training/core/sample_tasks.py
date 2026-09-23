
"""Run one sample as a background task and record it in the SampleFutureStore."""
import asyncio
import logging
import traceback
from typing import Any, Awaitable, Callable, Dict

from ..models.responses import validate_result
from ..proto.wire import serialize_result
from ..storage.sample_futures import SampleFutureStore

logger = logging.getLogger(__name__)

def start_sample_task(
    store: SampleFutureStore,
    request_id: str,
    operation: str,
    task_func: Callable[[], Awaitable[Dict[str, Any]]],
) -> asyncio.Task:
    async def run() -> None:
        try:
            result = validate_result(operation, await task_func())
            # SDK >= 0.25 fetches sample results only as proto; a result with no
            # proto encoding is a server bug and fails the future, not the poll.
            proto = await asyncio.to_thread(serialize_result, operation, result)
            store.complete(request_id, result, proto)
        except asyncio.CancelledError:
            store.fail(request_id, "cancelled")   # no-op when cancel() already recorded it
            raise
        except Exception as e:
            logger.error("[%s] %s failed: %s\n%s", request_id, operation, e, traceback.format_exc())
            store.fail(request_id, str(e))

    task = asyncio.create_task(run())
    store.attach_task(request_id, task)
    return task