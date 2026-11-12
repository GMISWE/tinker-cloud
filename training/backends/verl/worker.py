"""The verl worker a model runs on: upstream's Tinker worker, with the gradient clip as a step argument."""
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker


class StepClipWorker(TinkerActorRolloutRefWorker):
    """Upstream reads the clip from the engine's config, fixed at build; a Tinker step carries its own."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, optim_step_params: dict, clip_grad: float) -> dict:
        """One optimizer step: the gradient is clipped at norm `clip_grad` (`math.inf`: no clip), then
        `optim_step_params` override every param group. Returns the pre-clip `grad_norm`."""
        self.actor.engine.optimizer_config.clip_grad = clip_grad
        return super().optimizer_step(optim_step_params=optim_step_params)
