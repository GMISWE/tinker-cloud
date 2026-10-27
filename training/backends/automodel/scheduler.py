"""
The engine scheduler of one training group (R2.2): one FIFO for every tenant
that shares the group's ranks, run by one worker task.

The service's per-model ordering (services/ordering.py) delivers a model's
passes concurrently and in program order, and a barrier only after its earlier
passes completed. Here the head of the queue decides what runs next:

- a pass takes with it the consecutive passes behind it that have the same
  PassKey (model, kind, loss, loss config), and they run as one coalesced pass.
  The loss is a per-token sum, so the coalesced gradient is the sum of the
  separate ones. A run ends at the first op of another model or key: tenants
  keep arrival order, so adapter swaps follow it.
- a barrier (optim_step, save, load, delete) runs alone.

There is no batching window: whatever queued while the previous op ran is the
next run. A failure inside a run fails every request in it (one pass, one
fault); validation happens before submit, so a bad request fails alone.
"""
import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Generic, List, Optional, Tuple, TypeVar, Union

from ..base import BackendError

logger = logging.getLogger(__name__)

P = TypeVar("P")  # one request's pass payload
R = TypeVar("R")  # one request's pass result


@dataclass(frozen=True)
class PassKey:
    """Passes coalesce only when all of these are equal."""
    model_id: str
    kind: str                                   # "forward" | "forward_backward"
    loss_fn: str
    loss_fn_config: Tuple[Tuple[str, float], ...]


@dataclass
class _Pass(Generic[P]):
    key: PassKey
    payload: P
    future: asyncio.Future


@dataclass
class _Barrier:
    model_id: str
    fn: Callable[[], Awaitable[Any]]
    future: asyncio.Future


RunPass = Callable[[PassKey, List[P]], Awaitable[List[R]]]


class GroupScheduler(Generic[P, R]):
    """`run_pass(key, payloads)` executes one coalesced pass and returns one
    result per payload, in order."""

    def __init__(self, name: str, run_pass: RunPass):
        self.name = name
        self._run_pass = run_pass
        self._queue: Deque[Union[_Pass[P], _Barrier]] = deque()
        self._wakeup = asyncio.Event()
        self._closed: Optional[str] = None
        self._worker = asyncio.create_task(self._loop(), name=f"scheduler:{name}")
        self._worker.add_done_callback(self._on_worker_exit)

    async def submit_pass(self, key: PassKey, payload: P) -> R:
        return await self._enqueue(_Pass(key, payload, self._new_future()))

    async def submit_barrier(self, model_id: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        return await self._enqueue(_Barrier(model_id, fn, self._new_future()))

    def queued(self) -> int:
        return len(self._queue)

    def abandon(self, reason: str) -> None:
        """Fail every queued op with `reason` and refuse new ones, at once;
        the op in progress finishes. Callable from inside that op."""
        self._closed = reason
        while self._queue:
            self._fail(self._queue.popleft().future, reason)
        self._wakeup.set()

    async def close(self, reason: str) -> None:
        """abandon(reason), then wait for the op in progress."""
        self.abandon(reason)
        await self._worker

    def _new_future(self) -> asyncio.Future:
        if self._closed is not None:
            raise BackendError(f"training group {self.name} is closed: {self._closed}",
                               backend="automodel", operation="schedule")
        return asyncio.get_running_loop().create_future()

    async def _enqueue(self, op: Union[_Pass[P], _Barrier]) -> Any:
        self._queue.append(op)
        self._wakeup.set()
        return await op.future

    async def _loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while self._queue:
                head = self._queue.popleft()
                if head.future.cancelled():   # the submitter gave up before it ran
                    continue
                if isinstance(head, _Barrier):
                    await self._run_barrier(head)
                else:
                    await self._run_passes(self._take_run(head))
            if self._closed is not None:
                return

    def _take_run(self, head: _Pass[P]) -> List[_Pass[P]]:
        run = [head]
        while self._queue:
            nxt = self._queue[0]
            if isinstance(nxt, _Barrier) or nxt.key != head.key:
                break
            self._queue.popleft()
            if not nxt.future.cancelled():
                run.append(nxt)
        return run

    async def _run_passes(self, run: List[_Pass[P]]) -> None:
        if len(run) > 1:
            logger.info("[%s] coalesced %d %s passes of %s", self.name, len(run), run[0].key.kind, run[0].key.model_id)
        try:
            results = await self._run_pass(run[0].key, [op.payload for op in run])
            if len(results) != len(run):
                raise BackendError(f"run_pass returned {len(results)} results for {len(run)} requests",
                                   backend="automodel", operation=run[0].key.kind)
        except Exception as e:  # one pass, one fault: every request in it fails with the cause
            for op in run:
                self._fail(op.future, e)
            return
        for op, result in zip(run, results):
            if not op.future.done():
                op.future.set_result(result)

    async def _run_barrier(self, op: _Barrier) -> None:
        try:
            result = await op.fn()
        except Exception as e:  # the barrier's own failure, delivered to its submitter
            self._fail(op.future, e)
            return
        if not op.future.done():
            op.future.set_result(result)

    def _on_worker_exit(self, task: asyncio.Task) -> None:
        """A worker that dies (a bug, or cancellation at shutdown) must not
        leave submitters waiting: close the group and fail what is queued."""
        if task.cancelled():
            reason = "scheduler cancelled"
        elif task.exception() is not None:
            reason = f"scheduler died: {task.exception()!r}"
            logger.error("[%s] %s", self.name, reason)
        else:
            return
        self._closed = reason
        while self._queue:
            self._fail(self._queue.popleft().future, reason)

    @staticmethod
    def _fail(future: asyncio.Future, cause: Union[str, Exception]) -> None:
        if future.done():
            return
        if isinstance(cause, str):
            cause = BackendError(cause, backend="automodel", operation="schedule")
        future.set_exception(cause)


def pass_key(model_id: str, kind: str, loss_fn: str, loss_fn_config: Optional[dict]) -> PassKey:
    return PassKey(model_id, kind, loss_fn, tuple(sorted((loss_fn_config or {}).items())))
