"""
Sampling Service - Business Logic for Model Sampling

Backend-agnostic orchestration only: resolves the target model and delegates
to TrainingBackend.sample() / prepare_for_generation(). All backend specifics
(SGLang HTTP for Miles, batched Policy.generate() for NeMo RL) live in the
backend implementations (001 review P2).
"""
import logging
import uuid
from typing import Dict, Any, List, Optional

from ..backends.base import BackendHandle, TrainingBackend
from ..utils.helpers import find_model_with_rollout_manager

logger = logging.getLogger(__name__)


class SamplingService:
    """Service for model sampling via the backend's inference engine."""

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    def _resolve_handle(
        self, training_clients: Dict[str, Dict[str, Any]]
    ) -> BackendHandle:
        """Find the model serving rollouts and return its backend handle."""
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")
        handle = training_clients[model_id].get("backend_handle")
        if handle is None:
            raise RuntimeError(f"No backend handle for model {model_id}")
        return handle

    async def async_sample(
        self,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        prompt_logprobs: bool,
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Async sampling for a single prompt.

        Returns:
            Dict with sequences and optional prompt_logprobs

        Raises:
            RuntimeError: If no model with RolloutManager found
            BackendError: If the inference engine is unavailable
        """
        handle = self._resolve_handle(training_clients)
        logger.info(f"[{request_id}] Async sampling for {handle.model_id}")
        return await self.backend.sample(
            handle=handle,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            prompt_logprobs=prompt_logprobs,
        )

    async def sync_sample(
        self,
        request_id: str,
        prompts: List[List[int]],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synchronous sampling over multiple prompts.

        Returns:
            Dict with sequences list (num_samples per prompt, flattened)

        Raises:
            RuntimeError: If no model with RolloutManager found
            BackendError: If the inference engine is unavailable
        """
        handle = self._resolve_handle(training_clients)
        logger.info(f"[{request_id}] Sampling {num_samples} sequences")

        all_sequences = []
        for prompt_tokens in prompts:
            result = await self.backend.sample(
                handle=handle,
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                num_samples=num_samples,
                sampling_params=sampling_params,
                prompt_logprobs=False,
            )
            all_sequences.extend(result["sequences"])

        logger.info(f"[{request_id}] Sampling completed")
        return {"sequences": all_sequences}

    async def create_sampling_client(
        self,
        request_id: str,
        model_path: Optional[str],
        base_model: Optional[str],
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a sampling client bound to the serving model.

        Returns:
            Dict with sampling_client_id, model_path, status

        Raises:
            ValueError: If neither model_path nor base_model provided
            RuntimeError: If no model with RolloutManager found
            BackendError: If the inference engine cannot be made ready
        """
        resolved_model_path = model_path or base_model
        if not resolved_model_path:
            raise ValueError("Either model_path or base_model must be provided")

        handle = self._resolve_handle(training_clients)
        logger.info(f"[{request_id}] Creating sampling client for {resolved_model_path}")

        await self.backend.prepare_for_generation(handle)

        sampling_client_id = f"sampler_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{request_id}] Sampling client created: {sampling_client_id}")
        return {
            "sampling_client_id": sampling_client_id,
            "model_path": resolved_model_path,
            "status": "ready",
        }
