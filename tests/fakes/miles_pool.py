"""A miles pool and its queue without a cluster: the train call is a stand-in that records the
requests it was given; the E0 guard is the converter's own."""
from argparse import Namespace

from tests.fakes.held_queue import queue_behind_barrier, run_behind_barrier
from tinkercloud.training.backends.miles.backend import MilesBackend, MilesPool, _FbRequest

CE = "cross_entropy"


class PoolRig:
    """One pool and its queue. `log` holds what ran, in order: a tuple of request tags for a train
    call, a string for any other operation."""

    def __init__(self, max_samples=0, reorder=False, e0_tokens=0):
        self.log = []
        self.failing = set()
        self.backend = MilesBackend()
        self.backend._execute_fb_batch = self._train
        self.pool = MilesPool(
            train_group=None, rollout_manager=None, placement_group=None, controller=None,
            args=Namespace(data_parallel_size=1), hf_path="", base_model="",
            cobatch_max_samples=max_samples, cobatch_reorder=reorder, cobatch_e0_tokens=e0_tokens,
        )
        self.pool.scheduler = self.backend._pool_scheduler(self.pool)

    async def _train(self, pool, batch):
        tags = tuple(op.rollout_data["tag"] for op in batch)
        self.log.append(tags)
        if self.failing & set(tags):
            raise RuntimeError("CUDA OOM")
        return list(tags)

    def fb(self, tenant, tag, rows=1, loss=CE, tokens=8):
        request = _FbRequest(tenant=tenant, loss_fn=loss, num_samples=rows,
                             rollout_data={"tag": tag, "tokens": [[0] * tokens for _ in range(rows)]})
        return self.backend._pool_fb(self.pool, request)

    def other(self, tenant, tag):
        """A non-train operation of `tenant` (an optimizer step, a save); tenant None is a barrier."""
        async def fn():
            self.log.append(tag)
            if tag in self.failing:
                raise RuntimeError("disk full")
            return tag
        return self.pool.scheduler.submit_barrier(fn, tenant)

    def stop(self):
        return self.backend._pool_stop(self.pool)

    async def queue(self, *submissions):
        return await queue_behind_barrier(self.pool.scheduler.submit_barrier, submissions)

    async def run(self, *submissions):
        return await run_behind_barrier(self.pool.scheduler.submit_barrier, submissions)
