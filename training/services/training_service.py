"""
Training Service - Business Logic for Training Operations

Thin orchestration layer that delegates to the TrainingBackend abstraction.
All backend-specific logic (data conversion, validation, actor calls) lives
in the concrete backend implementation (e.g. MilesBackend, NemoRLBackend).
"""
import logging
from typing import Dict, List, Any, Optional

from ..backends.base import TrainingBackend, UnsupportedFeatureError

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Handles training operations: forward, forward_backward, optimizer step.

    Delegates all backend-specific work to the injected TrainingBackend.
    """

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    def _check_loss_supported(self, loss_fn: str) -> None:
        supported = self.backend.SUPPORTED_LOSS_FNS
        if loss_fn not in supported:
            raise UnsupportedFeatureError(
                f"loss_fn {loss_fn!r}", backend=type(self.backend).__name__,
                suggestion=f"this backend supports: {', '.join(sorted(supported))}",
            )

    async def forward(
        self,
        model_id: str,
        data: List[Any],
        loss_fn: str,
        client_info: Dict[str, Any],
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute forward-only pass (no gradients).

        Used for DPO reference model inference — computes logprobs without
        computing gradients.
        """
        logger.info("Forward pass for %s", model_id)
        self._check_loss_supported(loss_fn)

        handle = client_info["backend_handle"]
        result = await self.backend.forward(handle, data, loss_fn, loss_fn_config=loss_fn_config)

        logger.info("Forward pass completed for %s", model_id)
        return result

    async def forward_backward(
        self,
        model_id: str,
        data: List[Any],
        loss_fn: str,
        client_info: Dict[str, Any],
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Execute forward-backward pass (accumulate gradients, no optimizer step).
        """
        logger.info("Forward-backward pass for %s", model_id)
        self._check_loss_supported(loss_fn)

        handle = client_info["backend_handle"]
        result = await self.backend.forward_backward(handle, data, loss_fn, loss_fn_config=loss_fn_config)

        logger.info("Forward-backward completed for %s", model_id)
        return result

    async def apply_optimizer_step(
        self,
        model_id: str,
        client_info: Dict[str, Any],
        adam_params: Any = None,
    ) -> Dict[str, Any]:
        """
        Apply optimizer step to update model weights.
        """
        logger.info("Optimizer step for %s", model_id)

        handle = client_info["backend_handle"]

        # Extract learning rate from adam_params (Tinker API pattern)
        learning_rate = None
        adam_params_dict = None
        if adam_params is not None:
            learning_rate = adam_params.learning_rate
            logger.info("Setting learning rate to %s for %s", learning_rate, model_id)
            # P4: forward the FULL AdamParams so backends can honor or at least
            # detect-and-warn on beta/eps/clip values they cannot apply.
            adam_params_dict = {
                "beta1": adam_params.beta1,
                "beta2": adam_params.beta2,
                "eps": adam_params.eps,
                "weight_decay": adam_params.weight_decay,
                "grad_clip_norm": adam_params.grad_clip_norm,
            }

        result = await self.backend.apply_optimizer_step(
            handle, learning_rate=learning_rate, adam_params=adam_params_dict,
        )

        # Mirror top-level grad_norm into the metrics dict: the SDK's
        # OptimStepResponse carries only `metrics` (upstream shape), so a
        # value left at top level is silently dropped at the client seam.
        if result.get("grad_norm") is not None:
            metrics = dict(result.get("metrics") or {})
            metrics.setdefault("grad_norm", float(result["grad_norm"]))
            result["metrics"] = metrics

        logger.info(
            "Optimizer step completed for %s: grad_norm=%s, success=%s",
            model_id,
            result.get("grad_norm", "N/A"),
            result.get("success", "N/A"),
        )
        return result
