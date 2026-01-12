"""
Sampling Service - Business Logic for Model Sampling

Handles:
- Async sampling via SGLang
- Sync sampling via SGLang
- SGLang sampling client creation
"""
import logging
import uuid
from typing import Dict, Any, List, Optional

from ..utils.sglang_client import SGLangClient
from ..utils.helpers import find_model_with_rollout_manager

logger = logging.getLogger(__name__)


class SamplingService:
    """Service for managing model sampling via SGLang."""

    def __init__(self):
        """Initialize SamplingService."""
        pass

    async def async_sample(
        self,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        prompt_logprobs: bool,
        training_clients: Dict[str, Dict[str, Any]],
        image_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Async sampling via SGLang.

        Args:
            request_id: Request identifier for logging
            prompt_tokens: Input prompt tokens
            num_samples: Number of samples to generate
            sampling_params: Optional sampling parameters dict
            prompt_logprobs: Whether to return prompt logprobs
            training_clients: Global training clients dict
            image_data: Optional list of base64-encoded images or URLs for VLM models

        Returns:
            Dict with sequences and optional prompt_logprobs

        Raises:
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Find model with RolloutManager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Async sampling for {model_id}")
        if image_data:
            logger.info(f"[{request_id}] With {len(image_data)} images")

        # Get SGLang router URL
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")
        if not router_ip or not router_port:
            raise RuntimeError(f"Router address not available for model {model_id}")

        sglang_url = f"http://{router_ip}:{router_port}"

        # Use SGLang client
        sglang_client = SGLangClient(base_url=sglang_url)
        sequences = []
        prompt_logprobs_result = None

        for i in range(num_samples):
            result = await sglang_client.generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params or {},
                prompt_logprobs=prompt_logprobs,
                image_data=image_data
            )

            sequences.append({
                "tokens": result["tokens"],
                "logprobs": result["logprobs"],
                "text": result.get("text"),
                "stop_reason": result.get("stop_reason", "length")
            })

            if prompt_logprobs and prompt_logprobs_result is None:
                prompt_logprobs_result = result.get("prompt_logprobs")

        return {
            "sequences": sequences,
            "prompt_logprobs": prompt_logprobs_result
        }

    async def sync_sample(
        self,
        request_id: str,
        prompts: List[List[int]],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        training_clients: Dict[str, Dict[str, Any]],
        image_data_list: Optional[List[Optional[List[str]]]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous sampling via SGLang.

        Args:
            request_id: Request identifier for logging
            prompts: List of prompt token sequences
            num_samples: Number of samples per prompt
            sampling_params: Optional sampling parameters dict
            training_clients: Global training clients dict
            image_data_list: Optional list of image data lists (one per prompt)

        Returns:
            Dict with sequences list

        Raises:
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Find model with rollout manager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Sampling {num_samples} sequences")
        if image_data_list:
            logger.info(f"[{request_id}] With images for {sum(1 for x in image_data_list if x)} prompts")

        # Get SGLang router URL
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")
        if not router_ip or not router_port:
            raise RuntimeError("SGLang router not available")

        sglang_url = f"http://{router_ip}:{router_port}"
        sglang_client = SGLangClient(base_url=sglang_url)

        # Sample from all prompts
        all_sequences = []
        for i, prompt_tokens in enumerate(prompts):
            # Get image data for this prompt if provided
            image_data = None
            if image_data_list and i < len(image_data_list):
                image_data = image_data_list[i]

            for _ in range(num_samples):
                result = await sglang_client.generate(
                    input_ids=prompt_tokens,
                    sampling_params=sampling_params or {},
                    prompt_logprobs=False,
                    image_data=image_data
                )

                all_sequences.append({
                    "tokens": result["tokens"],
                    "logprobs": result["logprobs"],
                    "text": result.get("text"),
                    "stop_reason": result.get("stop_reason", "length")
                })

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
        Create sampling client (SGLang).

        Args:
            request_id: Request identifier for logging
            model_path: Optional model path
            base_model: Optional base model
            training_clients: Global training clients dict

        Returns:
            Dict with sampling_client_id, model_path, status

        Raises:
            ValueError: If neither model_path nor base_model provided
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Determine model path
        resolved_model_path = model_path or base_model
        if not resolved_model_path:
            raise ValueError("Either model_path or base_model must be provided")

        # Find model with rollout manager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Creating sampling client for {resolved_model_path}")

        # Get SGLang router info
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")

        if not router_ip or not router_port:
            raise RuntimeError("SGLang router not available")

        # Generate client ID
        sampling_client_id = f"sampler_{uuid.uuid4().hex[:8]}"

        logger.info(f"[{request_id}] Sampling client created: {sampling_client_id}")

        return {
            "sampling_client_id": sampling_client_id,
            "model_path": resolved_model_path,
            "status": "ready"
        }
