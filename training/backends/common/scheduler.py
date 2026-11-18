"""
One FIFO of operations for one group of GPU workers, run by one worker task.

A backend makes one `GroupScheduler` per group and submits two kinds of operation:

- a pass, `submit_pass(key, payload)`. Passes coalesce into a run, and the backend's
  `run_pass(key, payloads)` executes the run as one pass.
- a barrier, `submit_barrier(fn, key)`. It runs alone. Its optional key tells `admit` whose it is.

Forming a run. The first pass taken from the queue heads a run. The backend's `admit(run, next)`
then rules on each operation behind it, in queue order:

- JOIN: `next` is a pass and enters the run.
- DEFER: `next` waits; the ruling goes on with the operation behind it.
- END: `next` waits; the run is complete.

A waiting operation is not offered to `admit` again. The waiting operations run in arrival order
before the queue is read for a new head. A waiting pass that becomes a head heads its own run, which
may take passes from the queue ahead of the operations that still wait.

`same_key` coalesces equal keys only and keeps strict arrival order; `one_at_a_time` coalesces nothing.

Order. The service's per-model ordering (services/ordering.py) delivers a model's passes concurrently
and in program order, and a barrier only after its earlier passes complete. With `same_key` this
queue keeps that order. A backend whose `admit` defers answers for the order it gives up.

Failure. One run, one fault: a `run_pass` that raises fails every request in the run. An `admit` that
raises fails the run and the operation it was ruling on. So a backend refuses a bad request before it
submits it. Errors raised here carry the backend's name, given at construction.

Contract on the backend. `run_pass` over several payloads returns what the separate passes would.

There is no batching window: whatever queued while the previous operation ran forms the next run.
"""
import asyncio
import enum
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Generic, Hashable, List, Optional, Tuple, TypeVar, Union, cast

from ..base import BackendError

logger = logging.getLogger(__name__)

K = TypeVar("K", bound=Hashable)  # an operation's key
P = TypeVar("P")  # one request's pass payload
R = TypeVar("R")  # one request's pass result


# --- Operations -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PassKey:
    """The key of a backend that coalesces passes only when all of these are equal."""
    model_id: str
    kind: str                                   # "forward" | "forward_backward"
    loss_fn: str
    loss_fn_config: Tuple[Tuple[str, float], ...]


def pass_key(model_id: str, kind: str, loss_fn: str, loss_fn_config: Optional[dict]) -> PassKey:
    return PassKey(model_id, kind, loss_fn, tuple(sorted((loss_fn_config or {}).items())))


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


# --- Admission --------------------------------------------------------------------------------------

class Admit(enum.Enum):
    JOIN = "join"
    END = "end"
    DEFER = "defer"


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


def one_at_a_time(run: Run, nxt: Op) -> Admit:
    """Admission that joins nothing: every operation runs as its own run, in arrival order."""
    return Admit.END


# --- Scheduler --------------------------------------------------------------------------------------

class GroupScheduler(Generic[K, P, R]):
    """The FIFO of one group of workers (module docstring). `run_pass(key, payloads)` executes one
    coalesced pass under its head's key and returns one result per payload, in order."""

    def __init__(self, name: str, admit: AdmitFn, run_pass: RunPass, backend: str):
        self.name = name
        self.backend = backend
        self._admit = admit
        self._run_pass = run_pass
        self._queue: Deque[Op] = deque()     # submitted, not yet ruled on
        self._waiting: Deque[Op] = deque()   # ruled on (DEFER or END), not run; runs before `_queue`
        self._wakeup = asyncio.Event()
        self._closed: Optional[str] = None   # the close reason
        self._in_progress: Tuple[Op, ...] = ()   # what the worker awaits; a worker that dies fails it too
        self._worker = asyncio.create_task(self._loop(), name=f"scheduler:{name}")
        self._worker.add_done_callback(self._on_worker_exit)

    # Submission and close.

    async def submit_pass(self, key: K, payload: P) -> R:
        return await self._enqueue(Pass(key, payload, self._new_future()))

    async def submit_barrier(self, fn: Callable[[], Awaitable[Any]], key: Optional[K] = None) -> Any:
        return await self._enqueue(Barrier(key, fn, self._new_future()))

    def queued(self) -> int:
        return len(self._waiting) + len(self._queue)

    def abandon(self, reason: str) -> None:
        """Close the group at once: every queued op and every later submission
        fail with the closed error, which carries `reason`; the op in progress
        finishes. Callable from inside that op."""
        self._closed = reason
        for ops in (self._waiting, self._queue):
            while ops:
                self._fail(ops.popleft().future, self._closed_error())
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
            raise self._closed_error()
        return asyncio.get_running_loop().create_future()

    def _closed_error(self) -> BackendError:
        """One message for a closed group: what the close failed and what is submitted after it."""
        return BackendError(f"training group {self.name} is closed: {self._closed}",
                            backend=self.backend, operation="schedule")

    async def _enqueue(self, op: Op) -> Any:
        self._queue.append(op)
        self._wakeup.set()
        return await op.future

    # The worker.

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
                # Cleared here, not in a `finally`: a worker that dies inside an op leaves it set
                # for `_on_worker_exit` to fail.
                self._in_progress = ()
            if self._closed is not None:
                return

    async def _run_passes(self, head: Pass[K, P]) -> None:
        passes: List[Pass[K, P]] = [head]
        try:
            passes = self._form_run(head)
            if len(passes) > 1:
                logger.info("[%s] coalesced %d passes under %r", self.name, len(passes), head.key)
            self._in_progress = tuple(passes)
            results = await self._run_pass(head.key, [op.payload for op in passes])
            if len(results) != len(passes):
                raise BackendError(f"run_pass returned {len(results)} results for {len(passes)} requests",
                                   backend=self.backend, operation="schedule")
        except Exception as e:  # one pass, one fault: every request in it fails with the cause  # noqa: BLE001  (#191)
            for op in passes:   # an `admit` failure: `_form_run` already failed its run, and `passes` is [head]
                self._fail(op.future, e)
            return
        for op, result in zip(passes, results):
            self._resolve(op.future, result)

    def _form_run(self, head: Pass[K, P]) -> List[Pass[K, P]]:
        """Rule on the queue behind `head`, in order, and return the run: a JOIN enters it; DEFER and
        END move the op to `_waiting`; END stops. An `admit` that raises fails the run so far and the
        op it was ruling on, then re-raises."""
        passes: List[Pass[K, P]] = [head]
        deferred = 0
        while self._queue:
            nxt = self._queue.popleft()
            if nxt.future.cancelled():
                continue
            try:
                verdict = self._admit(Run(tuple(passes), tuple(self._waiting)), nxt)
                if verdict is Admit.JOIN and not isinstance(nxt, Pass):
                    raise BackendError(f"admit joined a barrier to a pass ({nxt.key!r})",
                                       backend=self.backend, operation="schedule")
            except Exception as e:   # `nxt` left the queue and waits nowhere: it fails with the run
                for op in (*passes, nxt):
                    self._fail(op.future, e)
                raise
            if verdict is Admit.JOIN:
                passes.append(cast(Pass[K, P], nxt))   # a barrier's JOIN raised above
                continue
            self._waiting.append(nxt)   # behind what waits already: arrival order holds among them
            if verdict is Admit.END:
                break
            deferred += 1
        if deferred:
            logger.info("[%s] deferred %d op(s) past a run of %d", self.name, deferred, len(passes))
        return passes

    async def _run_barrier(self, op: Barrier[K]) -> None:
        self._in_progress = (op,)
        try:
            result = await op.fn()
        except Exception as e:  # the barrier's own failure, delivered to its submitter  # noqa: BLE001  (#191)
            self._fail(op.future, e)
            return
        self._resolve(op.future, result)

    def _on_worker_exit(self, task: asyncio.Task) -> None:
        """A worker that dies (a bug, or cancellation at shutdown) must not
        leave submitters waiting: close the group and fail what is queued and
        what was in progress."""
        if task.cancelled():
            reason = "scheduler cancelled"
        elif task.exception() is not None:
            reason = f"scheduler died: {task.exception()!r}"
            logger.error("[%s] %s", self.name, reason)
        else:
            return
        self.abandon(reason)
        for op in self._in_progress:
            self._fail(op.future, self._closed_error())
        self._in_progress = ()

    # Delivery: a future its submitter cancelled, or one already failed, keeps its outcome.

    def _resolve(self, future: asyncio.Future, result: Any) -> None:
        if not future.done():
            future.set_result(result)

    def _fail(self, future: asyncio.Future, cause: Exception) -> None:
        if not future.done():
            future.set_exception(cause)
