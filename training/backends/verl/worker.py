"""The verl worker a model runs on: upstream's Tinker worker, with the gradient clip as a step argument, a
call that returns cached GPU memory to the device, and a reading of a pass's GPU memory."""
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

from ..common import gpu_memory


class ModelWorker(TinkerActorRolloutRefWorker):
    """What a Tinker model needs beyond upstream's worker. Upstream reads the clip from the engine's config,
    fixed at build, where a Tinker step carries its own; and upstream runs one model per GPU, where ours
    share one from separate processes."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, optim_step_params: dict, clip_grad: float) -> dict:
        """One optimizer step: the gradient is clipped at norm `clip_grad` (`math.inf`: no clip), then
        `optim_step_params` override every param group. Returns the pre-clip `grad_norm`."""
        self.actor.engine.optimizer_config.clip_grad = clip_grad
        return super().optimizer_step(optim_step_params=optim_step_params)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_cached_memory(self) -> None:
        """Return the allocator's cached GPU memory to the device: a process keeps what a pass reserved
        until it is told to let go, and the GPU's other models cannot use it
        (`tests/gpu/test_verl_fit.py` bounds what a pass leaves)."""
        aggressive_empty_cache(force_sync=True)

    # The two passes are upstream's, with the allocator's peaks restarted first, so that `pass_memory`
    # reads the pass in hand: upstream reports peaks too (`perf/max_memory_*`) and never restarts them.
    # Each override repeats upstream's dispatch mode, as `optimizer_step` above does.
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"), blocking=False)
    def forward_backward(self, data):
        gpu_memory.restart_peaks()
        return super().forward_backward(data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data):
        gpu_memory.restart_peaks()
        return super().compute_log_prob(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pass_memory(self) -> dict:
        """This rank's GPU memory for the pass that has just returned (`common/gpu_memory.py`). The
        device reading counts the GPU's other models too."""
        return gpu_memory.reading()
