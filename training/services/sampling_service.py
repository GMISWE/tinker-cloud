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

logger = logging.getLogger(__name__)



def _check_sample_request(handle, prompt_tokens, sampling_params) -> None:
    """Typed rejection before the engine sees the request.

    `max_tokens` must be stated (pydantic already requires it when
    sampling_params is present; this covers an absent sampling_params), and
    prompt + max_tokens must fit the engine context the backend recorded on
    its handle at create_model. Unknown context (None) is not checked.
    """
    from ..backends.base import SampleRequestError

    backend = handle.backend_type
    max_tokens = (sampling_params or {}).get("max_tokens")
    if max_tokens is None:
        raise SampleRequestError("sampling_params.max_tokens is required", backend=backend)
    ctx = handle.context_length
    if ctx is not None and len(prompt_tokens) + int(max_tokens) > ctx:
        raise SampleRequestError(
            f"prompt_len {len(prompt_tokens)} + max_tokens {int(max_tokens)} = "
            f"{len(prompt_tokens) + int(max_tokens)} exceeds the model context {ctx}",
            backend=backend,
        )

class SamplingService:
    """Service for model sampling via the backend's inference engine."""

    def __init__(self, backend: TrainingBackend):
        self.backend = backend

    def _resolve_handle(
        self,
        training_clients: Dict[str, Dict[str, Any]],
        model_id: Optional[str],
    ) -> BackendHandle:
        """Return the target model's backend handle. The router resolves the
        model (sampler -> owning model, or tinker:// path); there is no
        fallback to an arbitrary model."""
        if not model_id:
            raise RuntimeError("sampling request did not resolve to a model")
        if model_id not in training_clients:
            raise RuntimeError(f"Sampler's model {model_id} no longer exists")
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
        training_clients: Dict[str, Dict[str, Any]],
        pinned_version: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async sampling for a single prompt.

        `pinned_version` (BUG-015): weight version the requesting sampler was
        created at; backends route pinned logprob reads off the live engine.
        `model_id`: the sampler's owning model (multi-tenant routing).

        Returns:
            Dict with sequences and optional prompt_logprobs

        Raises:
            RuntimeError: If no model with RolloutManager found
            BackendError: If the inference engine is unavailable
        """
        handle = self._resolve_handle(training_clients, model_id)
        _check_sample_request(handle, prompt_tokens, sampling_params)
        logger.info(f"[{request_id}] Async sampling for {handle.model_id}")
        return await self.backend.sample(
            handle=handle,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            prompt_logprobs=prompt_logprobs,
            pinned_version=pinned_version,
        )

    async def sync_sample(
        self,
        request_id: str,
        prompts: List[List[int]],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        training_clients: Dict[str, Dict[str, Any]],
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous sampling over multiple prompts on `model_id`.

        Returns:
            Dict with sequences list (num_samples per prompt, flattened)
        """
        handle = self._resolve_handle(training_clients, model_id)
        logger.info(f"[{request_id}] Sampling {num_samples} sequences")

        all_sequences = []
        for prompt_tokens in prompts:
            _check_sample_request(handle, prompt_tokens, sampling_params)
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
        training_clients: Dict[str, Dict[str, Any]],
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a sampling client bound to `model_id` (named by model_path).

        Returns:
            Dict with sampling_client_id, model_path, status
        """
        resolved_model_path = model_path or f"tinker://{model_id}"
        handle = self._resolve_handle(training_clients, model_id)
        logger.info(f"[{request_id}] Creating sampling client for {resolved_model_path}")

        await self.backend.prepare_for_generation(handle)

        sampling_client_id = f"sampler_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{request_id}] Sampling client created: {sampling_client_id}")
        return {
            "sampling_client_id": sampling_client_id,
            "model_path": resolved_model_path,
            "status": "ready",
        }
