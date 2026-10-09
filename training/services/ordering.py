"""
Per-model program order for training operations.

The SDK issues a model's requests in program order and the routers turn each
into an asyncio task, so without this layer forward_backward(N+1) could run
before optim_step(N) had even started, and a save could observe a half-applied
step. Two kinds of operation:

- "pass"    (forward, forward_backward): may overlap with earlier work. A pass
            starts only after every earlier operation on the model has STARTED,
            so it can never be folded into an optimizer step that was submitted
            before it.
- "barrier" (optim_step, save_*, load_weights, unload_model): starts only after
            every earlier operation on the model has COMPLETED, and nothing
            submitted after it starts before it does.

Backend-level locks remain the second line of defence; this layer makes the
protocol's ordering explicit and backend-independent.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Set

logger = logging.getLogger(__name__)

PASS = "pass"
BARRIER = "barrier"

# operation name -> kind; anything else is not ordered (e.g. sampling)
KINDS = {
    "forward": PASS,
    "forward_backward": PASS,
    "optim_step": BARRIER,
    "save_weights": BARRIER,
    "save_weights_for_sampler": BARRIER,
    "save_weights_external": BARRIER,
    "load_weights": BARRIER,
    "unload_model": BARRIER,
}


@dataclass
class _ModelState:
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight: Set[asyncio.Task] = field(default_factory=set)


class ModelQueues:
    """Process-wide ordering state, one entry per model."""

    def __init__(self) -> None:
        self._models: Dict[str, _ModelState] = {}

    def _state(self, model_id: str) -> _ModelState:
        st = self._models.get(model_id)
        if st is None:
            st = self._models[model_id] = _ModelState()
        return st

    async def run(self, model_id: str, kind: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Run `fn` on `model_id` honouring the ordering rules for `kind`."""
        st = self._state(model_id)
        async with st.start_lock:  # starts happen strictly in submission order
            if kind == BARRIER and st.inflight:
                earlier = list(st.inflight)
                logger.debug("%s barrier waits for %d in-flight op(s)", model_id, len(earlier))
                await asyncio.gather(*earlier, return_exceptions=True)
            task = asyncio.ensure_future(fn())
            st.inflight.add(task)
            task.add_done_callback(st.inflight.discard)
        return await task

    def forget(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def inflight_count(self, model_id: str) -> int:
        st = self._models.get(model_id)
        return len(st.inflight) if st else 0


queues = ModelQueues()
