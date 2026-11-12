"""The verl worker a model runs on: upstream's Tinker worker, with the gradient clip as a step argument and
a call that returns cached GPU memory to the device."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker


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
