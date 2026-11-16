"""
A scheduler for the operations that share one set of GPU workers: one FIFO,
run by one worker task. A backend makes one per group of workers and submits
two kinds of operation, each with a key the backend chooses:

- a pass. The head pass opens a run; the backend's `admit(run, next)` rules on
  each operation queued behind it, in order: JOIN (a pass enters the run), END
  (the run is complete) or DEFER (`next` waits, and the ruling goes on with the
  operation behind it). `run_pass` receives the run's payloads as one coalesced
  pass.
- a barrier runs alone. It may carry a key, so `admit` can tell whose it is.

The operations a run deferred and the one that ended it wait, in arrival
order, and each runs before the queue is read for a new head. A waiting
operation is not offered to `admit` again: a barrier runs alone, and a pass
heads its own run, which may take operations from the queue ahead of what
still waits behind it.
`same_key` is the admission of a backend that only coalesces equal keys and
keeps strict arrival order. An `admit` that raises fails the run and the
operation it was ruling on, as a failed pass does.

The service's per-model ordering (services/ordering.py) delivers a model's
passes concurrently and in program order, and a barrier only after its earlier
passes completed; with `same_key` this queue keeps that order. A backend whose
`admit` defers answers for the order it gives up.

There is no batching window: whatever queued while the previous operation ran
is the next run. A failure inside a run fails every request in it (one pass,
one fault), so a backend refuses a bad request before it submits it.

The backend answers for what coalescing computes: `run_pass` over several
payloads must return what the separate passes would. Errors raised here carry
the backend's name, given at construction.
"""
import asyncio
import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Generic, Hashable, List, Optional, Tuple, TypeVar, Union

from ..base import BackendError

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)  # an operation's key
P = TypeVar("P")  # one request's pass payload
R = TypeVar("R")  # one request's pass result


class Admit(enum.Enum):
    JOIN = "join"
    END = "end"
    DEFER = "defer"


@dataclass(frozen=True)
class PassKey:
    """The key of a backend that coalesces passes only when all of these are equal."""
    model_id: str
    kind: str                                   # "forward" | "forward_backward"
    loss_fn: str
    loss_fn_config: Tuple[Tuple[str, float], ...]


@dataclass
class Pass(Generic[K, P]):
    key: K
    payload: P
    future: asyncio.Future = field(repr=False)


@dataclass
class Barrier(Generic[K]):
    key: Optional[K]
    fn: Callable[[], Awaitable[Any]] = field(repr=False)
    future: asyncio.Future = field(repr=False)


Op = Union[Pass[K, P], Barrier[K]]


@dataclass(frozen=True)
class Run(Generic[K, P]):
    """What `admit` sees: the passes taken so far, head first, and the operations that already wait
    behind this run (deferred by it, or left waiting by an earlier one)."""
    passes: Tuple[Pass[K, P], ...]
    waiting: Tuple[Op, ...]


AdmitFn = Callable[[Run[K, P], Op], Admit]
RunPass = Callable[[K, List[P]], Awaitable[List[R]]]


def same_key(run: Run, nxt: Op) -> Admit:
    """Consecutive passes of one key coalesce; anything else ends the run."""
    return Admit.JOIN if isinstance(nxt, Pass) and nxt.key == run.passes[0].key else Admit.END


class GroupScheduler(Generic[K, P, R]):
    """`run_pass(key, payloads)` executes one coalesced pass, under its head's key, and returns one
    result per payload, in order."""

    def __init__(self, name: str, admit: AdmitFn, run_pass: RunPass, backend: str):
        self.name = name
        self.backend = backend
        self._admit = admit
        self._run_pass = run_pass
        self._queue: Deque[Op] = deque()
        self._waiting: Deque[Op] = deque()
        self._wakeup = asyncio.Event()
        self._closed: Optional[str] = None
        self._worker = asyncio.create_task(self._loop(), name=f"scheduler:{name}")
        self._worker.add_done_callback(self._on_worker_exit)

    async def submit_pass(self, key: K, payload: P) -> R:
        return await self._enqueue(Pass(key, payload, self._new_future()))

    async def submit_barrier(self, fn: Callable[[], Awaitable[Any]], key: Optional[K] = None) -> Any:
        return await self._enqueue(Barrier(key, fn, self._new_future()))

    def queued(self) -> int:
        return len(self._waiting) + len(self._queue)

    def abandon(self, reason: str) -> None:
        """Fail every queued op with `reason` and refuse new ones, at once;
        the op in progress finishes. Callable from inside that op."""
        self._closed = reason
        self._fail_queued(reason)
        self._wakeup.set()

    async def close(self, reason: str) -> None:
        """abandon(reason), then wait for the op in progress. A worker that
        already ended (cancelled through an earlier close, or dead) is not
        awaited again: its CancelledError is not this caller's."""
        self.abandon(reason)
        if not self._worker.done():
            await self._worker

    def _new_future(self) -> asyncio.Future:
        if self._closed is not None:
            raise BackendError(f"training group {self.name} is closed: {self._closed}",
                               backend=self.backend, operation="schedule")
        return asyncio.get_running_loop().create_future()

    async def _enqueue(self, op: Op) -> Any:
        self._queue.append(op)
        self._wakeup.set()
        return await op.future

    async def _loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            while self._waiting or self._queue:
                head = self._waiting.popleft() if self._waiting else self._queue.popleft()
                if head.future.cancelled():   # the submitter gave up before it ran
                    continue
                if isinstance(head, Barrier):
                    await self._run_barrier(head)
                else:
                    await self._run_passes(head)
            if self._closed is not None:
                return

    async def _run_passes(self, head: Pass[K, P]) -> None:
        run: List[Pass[K, P]] = [head]
        ruled_on: Optional[Op] = None   # left the queue and waits nowhere while `admit` rules on it
        deferred = 0
        try:
            while self._queue:
                nxt = self._queue.popleft()
                if nxt.future.cancelled():
                    continue
                ruled_on = nxt
                verdict = self._admit(Run(tuple(run), tuple(self._waiting)), nxt)
                if verdict is Admit.JOIN:
                    if not isinstance(nxt, Pass):
                        raise BackendError(f"admit joined a barrier to a pass ({nxt.key!r})",
                                           backend=self.backend, operation="schedule")
                    run.append(nxt)
                    ruled_on = None
                    continue
                ruled_on = None
                self._waiting.append(nxt)   # behind what waits already: arrival order holds among them
                if verdict is Admit.END:
                    break
                deferred += 1
            if deferred:
                logger.info("[%s] deferred %d op(s) past a run of %d", self.name, deferred, len(run))
            if len(run) > 1:
                logger.info("[%s] coalesced %d passes under %r", self.name, len(run), head.key)
            results = await self._run_pass(head.key, [op.payload for op in run])
            if len(results) != len(run):
                raise BackendError(f"run_pass returned {len(results)} results for {len(run)} requests",
                                   backend=self.backend, operation="schedule")
        except Exception as e:  # one pass, one fault: every request in it fails with the cause  # noqa: BLE001  (#191)
            for op in run if ruled_on is None else [*run, ruled_on]:
                self._fail(op.future, e)
            return
        for op, result in zip(run, results):
            if not op.future.done():
                op.future.set_result(result)

    async def _run_barrier(self, op: Barrier[K]) -> None:
        try:
            result = await op.fn()
        except Exception as e:  # the barrier's own failure, delivered to its submitter  # noqa: BLE001  (#191)
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
        self._fail_queued(reason)

    def _fail_queued(self, reason: str) -> None:
        for ops in (self._waiting, self._queue):
            while ops:
                self._fail(ops.popleft().future, reason)

    def _fail(self, future: asyncio.Future, cause: Union[str, Exception]) -> None:
        if future.done():
            return
        if isinstance(cause, str):
            cause = BackendError(cause, backend=self.backend, operation="schedule")
        future.set_exception(cause)


def pass_key(model_id: str, kind: str, loss_fn: str, loss_fn_config: Optional[dict]) -> PassKey:
    return PassKey(model_id, kind, loss_fn, tuple(sorted((loss_fn_config or {}).items())))
