"""
Training Service - Business Logic for Training Operations

Thin orchestration layer that delegates to the TrainingBackend abstraction.
All backend-specific logic (data conversion, validation, actor calls) lives
in the concrete backend implementation (e.g. MilesBackend, NemoRLBackend).
"""
import logging
from typing import Dict, List, Any, Optional

from ..backends.base import TrainingBackend, UnsupportedFeatureError
from ..models.requests import OptimParams

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

    def check_target_rank(self, data: List[Any]) -> None:
        """Raises UnsupportedFeatureError for `target_tokens` of a rank the backend
        does not read. The router calls it before it queues the pass."""
        limit = self.backend.TARGET_TOKENS_MAX_RANK
        for i, datum in enumerate(data):
            target = datum.loss_fn_inputs.get("target_tokens")   # RL datums may carry none
            # no shape on the wire is a flat list: rank 1
            rank = 1 if target is None or target.shape is None else len(target.shape)
            if rank > limit:
                raise UnsupportedFeatureError(
                    f"target_tokens of rank {rank} (datum {i}, shape {target.shape})",
                    backend=type(self.backend).__name__,
                    suggestion=f"this backend reads target_tokens up to rank {limit}",
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

    @staticmethod
    def require_optimizer_family(client_info: Dict[str, Any], optim_params: OptimParams) -> None:
        """Raises ValueError (a 400) naming both families when the step's
        parameters are not the model's family."""
        family = client_info["optimizer_config"]["type"]
        if optim_params.type != family:
            raise ValueError(f"optim_step carries {optim_params.type} parameters but the model's optimizer is {family}")

    async def apply_optimizer_step(
        self,
        model_id: str,
        client_info: Dict[str, Any],
        optim_params: OptimParams,
    ) -> Dict[str, Any]:
        """
        Apply optimizer step to update model weights. `optim_params` passed
        `require_optimizer_family`.
        """
        logger.info("Optimizer step for %s", model_id)

        handle = client_info["backend_handle"]
        logger.info("Setting learning rate to %s for %s", optim_params.learning_rate, model_id)

        result = await self.backend.apply_optimizer_step(handle, optim_params)

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
