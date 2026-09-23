
"""
Sample futures: the in-memory store for asample / sample results.

Rollout results are transient: a restart frees every model, so a persisted
sample result would never be read back. Sample futures therefore never touch
the SQLite futures table (training and control operations still do). The
store also serves the SDK's per-session poll (/api/v1/retrieve_futures): one
completion log per (sampling_session_id, cloned_sampler_id) with a monotonic
cursor, and per-request results kept until fetched. Single event loop, no
locks.
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# SDK: a SamplingClient's seq_ids live in one block of 1e9 per cloned client.
CLONE_BLOCK = 1_000_000_000

# Retention after each state, seconds.
RETRIEVED_TTL_S = 120.0   # delivered; kept so an SDK retry after a lost response still finds it
COMPLETED_TTL_S = 600.0   # terminal, never fetched
PENDING_TTL_S = 3600.0    # task never reported: a bug or a hung engine
SWEEP_INTERVAL_S = 30.0


SessionKey = Tuple[str, int]                       # (sampling_session_id, cloned_sampler_id)
Completion = Tuple[str, str, Optional[int]]        # (request_id, "finished"|"failed", payload bytes)


def clone_of(seq_id: Optional[int]) -> int:
    return 0 if seq_id is None else seq_id // CLONE_BLOCK


@dataclass
class SampleFuture:
    request_id: str
    operation: str                     # "asample" | "sample"
    model_id: str
    session: Optional[SessionKey]
    status: str = "pending"            # pending | completed | failed
    result: Optional[Dict[str, Any]] = None
    result_proto: Optional[bytes] = None
    error: Optional[str] = None
    task: Optional[asyncio.Task] = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None
    retrieved_at: Optional[float] = None
    
    @property
    def payload_size(self) -> Optional[int]:
        return None if self.result_proto is None else len(self.result_proto)


@dataclass
class _SessionLog:
    """Completion log of one (session, clone). Absolute cursor = base + len(entries)."""

    entries: Deque[Completion] = field(default_factory=deque)
    base: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def cursor(self) -> int:
        return self.base + len(self.entries)

    def append(self, entry: Completion) -> None:
        self.entries.append(entry)
        self.changed.set()

    def ack(self, prev_cursor: int) -> None:
        """The client presenting `prev_cursor` has seen every entry below it."""
        if prev_cursor > self.cursor:
            # Client is ahead of this process (server restarted mid-session):
            # renumber so what we have appended since sits above its cursor.
            self.base = prev_cursor
            return
        while self.entries and self.base < prev_cursor:
            self.entries.popleft()
            self.base += 1

    def since(self, prev_cursor: int) -> List[Completion]:
        return list(self.entries)[max(0, prev_cursor - self.base):]

class SampleFutureStore:
    def __init__(self) -> None:
        self._futures: Dict[str, SampleFuture] = {}
        self._logs: Dict[SessionKey, _SessionLog] = {}
        # Cursor a session had when its log was dropped: a recreated log starts
        # there, so a poller's old cursor never acks entries it has not seen.
        self._cursors: Dict[SessionKey, int] = {}
        self._by_seq: Dict[Tuple[str, int], str] = {}    # (sampling_session_id, seq_id) -> request_id

    def _log(self, key: SessionKey) -> _SessionLog:
        log = self._logs.get(key)
        if log is None:
            log = self._logs[key] = _SessionLog(base=self._cursors.get(key, 0))
        return log

    # -- registration --------------------------------------------------------

    def register(
        self,
        request_id: str,
        operation: str,
        model_id: str,
        sampling_session_id: Optional[str],
        seq_id: Optional[int],
    ) -> str:
        """Register a pending sample; returns the request_id that owns it.

        An SDK retry carries the same (sampling_session_id, seq_id): it gets
        the earlier request_id back and the caller starts no second task.
        """
        if sampling_session_id is not None and seq_id is not None:
            owner = self._by_seq.get((sampling_session_id, seq_id))
            if owner is not None and owner in self._futures:
                logger.info("[%s] retry of sample seq_id %s in %s -> %s", request_id, seq_id, sampling_session_id, owner)
                return owner
            self._by_seq[(sampling_session_id, seq_id)] = request_id
        session = None if sampling_session_id is None else (sampling_session_id, clone_of(seq_id))
        self._futures[request_id] = SampleFuture(
            request_id=request_id, operation=operation, model_id=model_id, session=session,
        )
        if session is not None:
            self._log(session)
        return request_id

    def attach_task(self, request_id: str, task: asyncio.Task) -> None:
        self._futures[request_id].task = task

    def get(self, request_id: str) -> Optional[SampleFuture]:
        return self._futures.get(request_id)

    # -- completion ----------------------------------------------------------

    def complete(self, request_id: str, result: Dict[str, Any], result_proto: bytes) -> None:
        fut = self._futures[request_id]
        if fut.status != "pending":       # cancelled while the result was in flight
            return
        fut.status, fut.result, fut.result_proto = "completed", result, result_proto
        self._finish(fut, "finished")

    def fail(self, request_id: str, error: str) -> None:
        fut = self._futures[request_id]
        if fut.status != "pending":
            return
        fut.status, fut.error = "failed", error
        self._finish(fut, "failed")

    def _finish(self, fut: SampleFuture, state: str) -> None:
        fut.completed_at = time.monotonic()
        fut.task = None
        fut.done.set()
        if fut.session is not None:
            self._logs[fut.session].append((fut.request_id, state, fut.payload_size))

    def mark_retrieved(self, request_id: str) -> None:
        fut = self._futures[request_id]
        if fut.retrieved_at is None:
            fut.retrieved_at = time.monotonic()
    def cancel(self, request_id: str, reason: str = "cancelled by client") -> bool:
        """Cancel a pending sample; False if it had already finished. KeyError if unknown."""
        fut = self._futures[request_id]
        if fut.status != "pending":
            return False
        task = fut.task
        self.fail(request_id, reason)
        if task is not None:
            task.cancel()
        return True

    def cancel_model(self, model_id: str) -> int:
        """Cancel every pending sample of a model that was deleted."""
        pending = [f.request_id for f in self._futures.values() if f.model_id == model_id and f.status == "pending"]
        for rid in pending:
            self.cancel(rid, f"model {model_id} was deleted")
        return len(pending)

    # -- per-session poll (SDK /retrieve_futures) ----------------------------
    async def poll(self, key: SessionKey, prev_cursor: int, timeout_s: float) -> Tuple[List[Completion], int]:
        """Entries at or after `prev_cursor`, waiting up to `timeout_s` for the first."""
        log = self._log(key)
        log.ack(prev_cursor)
        if not log.since(prev_cursor):
            log.changed.clear()          # no await between since() and clear(): no lost wakeup
            try:
                await asyncio.wait_for(log.changed.wait(), timeout_s)
            except asyncio.TimeoutError:
                pass
        return log.since(prev_cursor), log.cursor

    # -- retention -----------------------------------------------------------

    def sweep(self, now: Optional[float] = None) -> int:
        now = time.monotonic() if now is None else now
        # A sample pending past PENDING_TTL_S is failed like any other outcome
        # (logged to its session, done set) and then ages out as completed.
        for rid in [r for r, f in self._futures.items() if f.status == "pending" and now - f.created_at > PENDING_TTL_S]:
            logger.error("[%s] sample pending for %.0fs; failing it", rid, PENDING_TTL_S)
            self.cancel(rid, f"sample pending longer than {PENDING_TTL_S:.0f}s")
        expired: List[str] = []
        for rid, fut in self._futures.items():
            if fut.status == "pending":
                continue
            assert fut.completed_at is not None  # _finish set it with the terminal status
            if fut.retrieved_at is not None:
                if now - fut.retrieved_at > RETRIEVED_TTL_S:
                    expired.append(rid)
            elif now - fut.completed_at > COMPLETED_TTL_S:
                expired.append(rid)
        for rid in expired:
            del self._futures[rid]
        # Log entries whose future is gone can no longer be fetched: trim them
        # from the left (completion order ~ expiry order) so a session that is
        # polled only through retrieve_future does not grow without bound.
        for log in self._logs.values():
            while log.entries and log.entries[0][0] not in self._futures:
                log.entries.popleft()
                log.base += 1
        # Drop a log once every future of its session is gone, remembering its
        # cursor so a recreated log continues the numbering.
        live_sessions = {f.session for f in self._futures.values() if f.session is not None}
        for key in [k for k in self._logs if k not in live_sessions]:
            self._cursors[key] = self._logs.pop(key).cursor
        self._by_seq = {k: v for k, v in self._by_seq.items() if v in self._futures}
        return len(expired)
        
    async def sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            n = self.sweep()
            if n:
                logger.debug("swept %d sample future(s)", n)

    def stats(self) -> Dict[str, int]:
        by_status: Dict[str, int] = {}
        for f in self._futures.values():
            by_status[f.status] = by_status.get(f.status, 0) + 1
        return {"total": len(self._futures), "sessions": len(self._logs), **by_status}
