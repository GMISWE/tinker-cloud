"""
One training group's Ray side (R1): a placement group of {GPU:1, CPU:1}
bundles, one AutomodelWorker per bundle, and the calls that reach every rank
at once. The backend sees the ranks only through this class, so a CPU test
can stand in a fake with the same seven coroutines.

Names: the placement group is `am-<key>`, rank r's actor `am-<key>-r<r>`.
Neither is detached (they die with the server's Ray job); the names exist so
a server that starts after a crash can sweep what the job left (R5).

A call fails as a whole: every rank's answer is collected before the first
exception is raised, so a refused optimizer step (the same verdict on every
rank, the norm being all-reduced) surfaces once, with no reference left
pending. The rank's exception is raised as the instance it sent (see _all),
so the backend catches NonFiniteGradNorm by name and reports its message.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ray
from ray.exceptions import RayTaskError
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ...models.requests import AdamParams
from ..base import BackendError
from .converter import Batch
from .lora import LoraSpec

logger = logging.getLogger(__name__)

NAME_PREFIX = "am-"
# The cluster is static: bundles not placed within this time will not be
# placed later, so the create fails instead of hanging.
PG_READY_TIMEOUT_S = 60.0


class TrainingGroup:
    def __init__(self, key: str, num_ranks: int):
        self.key = key
        self.num_ranks = num_ranks
        self.name = f"{NAME_PREFIX}{key}"
        self.pg: Optional[Any] = None
        self.ranks: List[Any] = []

    async def start(self, base_model: str, lora: LoraSpec, seed: int, micro_batch_rows: int) -> int:
        """Place the bundles, start the ranks, build the model on each; returns
        the LoRA element count of the full model. A failure stops what was
        started, then re-raises (R5)."""
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
                ).remote(r, self.num_ranks, base_model, lora, seed, micro_batch_rows)
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

    async def forward_backward(self, shards: Sequence[Batch], clip: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """One ray.put per rank (R3); results in rank order."""
        return await self._all([w.forward_backward.remote(ray.put(s), clip) for w, s in zip(self.ranks, shards)])

    async def forward(self, shards: Sequence[Batch], clip: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        return await self._all([w.forward.remote(ray.put(s), clip) for w, s in zip(self.ranks, shards)])

    async def optim_step(self, adam: AdamParams) -> List[Dict[str, float]]:
        return await self._all([w.optim_step.remote(adam) for w in self.ranks])

    async def export_adapter(self, dest: Path) -> None:
        """Rank 0 writes the HF PEFT adapter at `dest`; every rank joins the gather."""
        await self._all([w.export_adapter.remote(str(dest)) for w in self.ranks])

    async def save_state(self, root: Path) -> None:
        """Rank 0 writes the adapter and the AdamW state under `root`; every rank joins the gathers."""
        await self._all([w.save_state.remote(str(root)) for w in self.ranks])

    async def load_state(self, adapter_dir: Path, optimizer_dir: Optional[Path]) -> None:
        """Every rank copies its shard of the adapter and, given optimizer_dir, of the AdamW state."""
        opt = None if optimizer_dir is None else str(optimizer_dir)
        await self._all([w.load_state.remote(str(adapter_dir), opt) for w in self.ranks])

    async def stop(self) -> None:
        """Kill the ranks and free the bundles. Safe to call twice."""
        for w in self.ranks:
            ray.kill(w, no_restart=True)
        self.ranks = []
        if self.pg is not None:
            remove_placement_group(self.pg)
            self.pg = None

    @staticmethod
    async def _all(refs: Sequence[Any]) -> List[Any]:
        """Every rank's answer in rank order; the first exception, once all have answered.

        A rank's own exception is raised as itself (its message is what the
        caller reports); the RayTaskError that carried it, with the remote
        traceback, stays attached as the cause. An actor death is not a task
        error and is raised as Ray reports it."""
        # Any: ray is not installed where the type check runs, so RayTaskError
        # is Unknown there and isinstance cannot narrow to it
        results: List[Any] = await asyncio.gather(*refs, return_exceptions=True)
        for r in results:
            if isinstance(r, RayTaskError):
                raise r.cause from r
            if isinstance(r, BaseException):
                raise r
        return list(results)


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
