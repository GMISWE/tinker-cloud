"""
Training Service - Business Logic for Training Operations

Thin orchestration layer that delegates to the TrainingBackend abstraction.
All backend-specific logic (data conversion, validation, actor calls) lives
in the concrete backend implementation (e.g. MilesBackend, NemoRLBackend).
"""
import logging
from typing import Dict, List, Any, Optional

from ..backends.base import TrainingBackend

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Handles training operations: forward, forward_backward, optimizer step.

    Delegates all backend-specific work to the injected TrainingBackend.
    """

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    async def forward(
        self,
        model_id: str,
        data: List[Any],
        loss_fn: str,
        client_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute forward-only pass (no gradients).

        Used for DPO reference model inference — computes logprobs without
        computing gradients.
        """
        logger.info("Forward pass for %s", model_id)

        handle = client_info["backend_handle"]
        result = await self.backend.forward(handle, data, loss_fn)

        logger.info("Forward pass completed for %s", model_id)
        return result

    async def forward_backward(
        self,
        model_id: str,
        data: List[Any],
        loss_fn: str,
        client_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute forward-backward pass (accumulate gradients, no optimizer step).
        """
        logger.info("Forward-backward pass for %s", model_id)

        handle = client_info["backend_handle"]
        result = await self.backend.forward_backward(handle, data, loss_fn)

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
        if adam_params is not None and hasattr(adam_params, "learning_rate"):
            learning_rate = adam_params.learning_rate
            logger.info("Setting learning rate to %s for %s", learning_rate, model_id)

        result = await self.backend.apply_optimizer_step(handle, learning_rate=learning_rate)

        logger.info(
            "Optimizer step completed for %s: grad_norm=%s, success=%s",
            model_id,
            result.get("grad_norm", "N/A"),
            result.get("success", "N/A"),
        )
        return result
