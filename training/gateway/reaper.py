"""
One task per model being freed.

The session reaper's tick and the server's shutdown schedule frees; neither
waits on one. A model whose backend delete is stuck (a hung op holding the
group) holds only its own task: the other expired models are still freed on
schedule, and shutdown cancels the stuck free, then frees every live model at
once. A model already being freed is not freed a second time.
"""
import asyncio
import logging
from typing import Awaitable, Callable, Collection, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

FreeFn = Callable[[str, str], Awaitable[None]]      # (model_id, reason) -> None; never raises
Expired = Iterable[Tuple[str, List[str]]]            # (session_id, its model ids)


class ModelReaper:
    def __init__(self, free: FreeFn) -> None:
        self._free = free
        self._frees: Dict[str, asyncio.Task] = {}   # model_id -> the task freeing it

    def free(self, model_id: str, reason: str) -> asyncio.Task:
        """Free `model_id` in its own task. A model whose free is in flight
        keeps that task; the new reason is dropped."""
        task = self._frees.get(model_id)
        if task is not None and not task.done():
            return task
        task = asyncio.create_task(self._free(model_id, reason), name=f"free:{model_id}")
        self._frees[model_id] = task
        task.add_done_callback(lambda t: self._forget(model_id, t))
        return task

    def _forget(self, model_id: str, task: asyncio.Task) -> None:
        if self._frees.get(model_id) is task:
            del self._frees[model_id]

    def inflight(self) -> int:
        return sum(1 for t in self._frees.values() if not t.done())

    def tick(self, expired: Expired, live: Collection[str]) -> int:
        """Schedule a free for every live model of an expired session and
        return at once. Returns how many frees were scheduled."""
        n = 0
        for session_id, model_ids in expired:
            logger.warning("Session %s expired: freeing %d model(s)", session_id, len(model_ids))
            for model_id in model_ids:
                if model_id in live:
                    self.free(model_id, f"session {session_id} expired")
                    n += 1
        return n

    async def run_forever(self, interval_s: float, expired: Callable[[], Expired],
                          live: Callable[[], Collection[str]]) -> None:
        """A tick every `interval_s`; a tick that raises is logged, not fatal."""
        while True:
            await asyncio.sleep(interval_s)
            try:
                self.tick(expired(), live())
            except Exception:  # pylint: disable=broad-except
                logger.exception("Session reaper tick failed")

    async def shutdown(self, model_ids: Iterable[str], reason: str = "shutdown") -> None:
        """Cancel every free in flight, wait for them to end, then free every
        model in `model_ids` concurrently and wait for all of them. A free that
        raised, in either phase, is reported after every free has run."""
        inflight = [t for t in self._frees.values() if not t.done()]
        for task in inflight:
            task.cancel()
        if inflight:
            await asyncio.wait(inflight)
        self._frees.clear()
        frees = [self.free(model_id, reason) for model_id in model_ids]
        if frees:
            await asyncio.wait(frees)   # every model gets its free, whatever the others do
        for task in inflight + frees:
            if not task.cancelled():    # the cancel was ours; anything else a free raised broke FreeFn's contract
                task.result()
