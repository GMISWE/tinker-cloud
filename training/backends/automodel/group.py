"""
One training group's Ray side (R1): a placement group of {GPU:1, CPU:1}
bundles, one AutomodelWorker per bundle, and the calls that reach every rank
at once. The backend sees the ranks only through this class, so a CPU test
can stand in a fake with the same nine coroutines. Every op names its tenant:
the ranks hold one adapter per tenant of the group (worker.py).

Names: the placement group is `am-<key>`, rank r's actor `am-<key>-r<r>`.
Neither is detached (they die with the server's Ray job); the names exist so
a server that starts after a crash can sweep what the job left (R5).

A call fails as a whole: after the first exception the other ranks get
RANK_GRACE_S to answer, so a refused optimizer step (the same verdict on every
rank, the norm being all-reduced) surfaces once, as the instance the rank sent
(see _all): the backend catches NonFiniteGradNorm by name and reports its
message. A rank that does not answer within the grace is stuck in a collective
its failed peer left; the call raises RanksStranded, and the group cannot be
used again.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.exceptions import RayTaskError
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ...models.requests import OptimParams
from ..base import BackendError
from .config import AutomodelConfig
from .converter import Batch
from .lora import LoraSpec

logger = logging.getLogger(__name__)

NAME_PREFIX = "am-"
# The cluster is static: bundles not placed within this time will not be
# placed later, so the create fails instead of hanging.
PG_READY_TIMEOUT_S = 60.0
# Expected, not measured: peers reaching the same verdict (an all-reduced norm)
# answer within seconds of each other, so a rank still running after this is
# taken to wait in a collective that will not complete.
RANK_GRACE_S = 30.0


class RanksStranded(RuntimeError):
    """A rank raised while other ranks of its group were still running; those
    ranks wait in a collective that will not complete. The rank's exception is
    the cause."""


class TrainingGroup:
    def __init__(self, key: str, num_ranks: int):
        self.key = key
        self.num_ranks = num_ranks
        self.name = f"{NAME_PREFIX}{key}"
        self.pg: Optional[Any] = None
        self.ranks: List[Any] = []

    async def start(self, base_model: str, lora: LoraSpec, config: AutomodelConfig) -> int:
        """Place the bundles, start the ranks, build the model on each (no
        tenant yet); returns the LoRA element count of the full model. A
        failure stops what was started, then re-raises."""
        # The worker needs the training env (Automodel); importing it here
        # keeps this module importable where that env is absent.
        from .worker import AutomodelWorker

        try:
            pg = placement_group([{"GPU": 1, "CPU": 1}] * self.num_ranks, strategy="PACK", name=self.name)
            self.pg = pg
            ready, _ = await asyncio.to_thread(ray.wait, [pg.ready()], timeout=PG_READY_TIMEOUT_S)
            if not ready:
                raise BackendError(
                    f"{self.num_ranks} free GPUs for training group {self.key} were not placed within "
                    f"{PG_READY_TIMEOUT_S:.0f}s", backend="automodel", operation="create_model")
            worker = ray.remote(num_gpus=1, num_cpus=1)(AutomodelWorker)
            self.ranks = [
                worker.options(
                    name=f"{self.name}-r{r}",
                    scheduling_strategy=PlacementGroupSchedulingStrategy(self.pg, placement_group_bundle_index=r),
                ).remote(r, self.num_ranks, base_model, lora, config)
                for r in range(self.num_ranks)
            ]
            init_method = await self.ranks[0].rendezvous_address.remote()
            lora_numel = await self._all([w.setup.remote(init_method) for w in self.ranks])
            # DTensor.numel() is the global count, so every rank reports the same number
            assert len(set(lora_numel)) == 1, f"ranks disagree on the LoRA size: {lora_numel}"
            return lora_numel[0]
        except BaseException:
            await self.stop()
            raise

    async def add_tenant(self, tenant: str, lora: LoraSpec, seed: int) -> None:
        """Every rank initialises the tenant's adapter from `seed` and makes it live, with the tenant's dropout."""
        await self._all([w.add_tenant.remote(tenant, lora, seed) for w in self.ranks])

    async def remove_tenant(self, tenant: str) -> None:
        """Every rank drops the tenant's state; the group stays up."""
        await self._all([w.remove_tenant.remote(tenant) for w in self.ranks])

    async def forward_backward(self, tenant: str, shards: Sequence[Batch],
                               clip: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """One ray.put per rank (R3); results in rank order."""
        return await self._all([w.forward_backward.remote(tenant, ray.put(s), clip) for w, s in zip(self.ranks, shards)])

    async def forward(self, tenant: str, shards: Sequence[Batch], clip: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        return await self._all([w.forward.remote(tenant, ray.put(s), clip) for w, s in zip(self.ranks, shards)])

    async def optim_step(self, tenant: str, params: OptimParams) -> List[Dict[str, float]]:
        return await self._all([w.optim_step.remote(tenant, params) for w in self.ranks])

    async def export_adapter(self, tenant: str, dest: Path) -> None:
        """Rank 0 writes the tenant's HF PEFT adapter at `dest`; every rank joins the gather."""
        await self._all([w.export_adapter.remote(tenant, str(dest)) for w in self.ranks])

    async def save_state(self, tenant: str, root: Path) -> None:
        """Rank 0 writes the tenant's adapter and AdamW state under `root`; every rank joins the gathers."""
        await self._all([w.save_state.remote(tenant, str(root)) for w in self.ranks])

    async def load_state(self, tenant: str, adapter_dir: Path, optimizer_dir: Optional[Path]) -> None:
        """Every rank copies its shard of the adapter and, given optimizer_dir, of the AdamW state into the tenant."""
        opt = None if optimizer_dir is None else str(optimizer_dir)
        await self._all([w.load_state.remote(tenant, str(adapter_dir), opt) for w in self.ranks])

    async def stop(self) -> None:
        """Kill the ranks and free the bundles. Safe to call twice."""
        for w in self.ranks:
            ray.kill(w, no_restart=True)
        self.ranks = []
        if self.pg is not None:
            remove_placement_group(self.pg)
            self.pg = None

    async def _all(self, refs: Sequence[Any]) -> List[Any]:
        """Every rank's answer in rank order, or the first rank's exception.

        The first exception is logged when it arrives, with the remote
        traceback. If every rank then answers within RANK_GRACE_S, the first
        exception in rank order is raised: a rank's own exception as itself
        (its message is what the caller reports), with the RayTaskError that
        carried it attached as the cause; an actor death as Ray reports it.
        Otherwise RanksStranded names the ranks still running."""
        futures = [asyncio.ensure_future(r) for r in refs]
        done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_EXCEPTION)
        if pending:   # a rank raised while its peers run
            first = next(r for r, f in enumerate(futures) if f in done and f.exception() is not None)
            err = futures[first].exception()
            logger.error("training group %s: rank %d raised %s; its peers get %.0fs to answer",
                         self.key, first, type(_unwrap(err)).__name__, RANK_GRACE_S, exc_info=err)
            _, pending = await asyncio.wait(pending, timeout=RANK_GRACE_S)
            if pending:
                stranded = [r for r, f in enumerate(futures) if f in pending]
                for f in pending:
                    f.cancel()
                raise RanksStranded(
                    f"training group {self.key}: rank {first} raised {_unwrap(err)!r}; ranks {stranded} "
                    f"did not answer within {RANK_GRACE_S:.0f}s and wait in a collective that will not "
                    f"complete") from _unwrap(err)
        for f in futures:
            err = f.exception()
            if err is not None and _unwrap(err) is not err:
                raise _unwrap(err) from err
            if err is not None:
                raise err
        return [f.result() for f in futures]


def _unwrap(err: BaseException) -> BaseException:
    """The exception a rank raised, out of the RayTaskError that carried it."""
    # Any: ray is not installed where the type check runs, so RayTaskError
    # is Unknown there and isinstance cannot narrow to it
    e: Any = err
    return e.cause if isinstance(e, RayTaskError) else err


def sweep_leftovers() -> int:
    """Kill `am-` actors and remove `am-` placement groups left in this
    namespace by an earlier server (R5). Returns how many were found."""
    found = 0
    for name in ray.util.list_named_actors():
        if name.startswith(NAME_PREFIX):
            ray.kill(ray.get_actor(name), no_restart=True)
            found += 1
    for info in placement_group_table().values():
        if info["name"].startswith(NAME_PREFIX) and info["state"] != "REMOVED":
            remove_placement_group(ray.util.get_placement_group(info["name"]))
            found += 1
    return found
