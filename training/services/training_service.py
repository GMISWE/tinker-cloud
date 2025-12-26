"""
Training Service - Business Logic for Training Operations

Extracted from api_refactored.py to separate concerns:
- HTTP layer (routers) handles request/response
- Service layer (this file) handles business logic
- Domain layer handles data transformations

This service is pure Python with no FastAPI dependencies.
"""
import asyncio
import logging
import ray
from typing import Dict, List, Any, Optional
from miles.utils.ray_utils import Box

from ..core.data_converter import TinkerDataConverter
from ..core.validators import RequestValidator
from ..utils.helpers import extract_learning_rates
from ..config import get_config

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Handles training operations: forward, forward_backward, optimizer step.

    Pure business logic with no HTTP concerns. All methods are async and
    return Dict[str, Any] that can be serialized to JSON.

    Example usage:
        service = TrainingService()
        result = await service.forward_backward(
            model_id="model_123",
            train_group=train_group,
            args=args,
            data=[...],
            loss_fn="cross_entropy"
        )
    """

    def __init__(self):
        """Initialize training service with converter"""
        self.converter = TinkerDataConverter()
        config = get_config()
        self.allow_partial_batches = getattr(config, "allow_partial_batches", False)

    async def forward(
        self,
        model_id: str,
        train_group: Any,
        data: List[Any],
        loss_fn: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute forward-only pass (no gradients).

        Used for DPO reference model inference - computes logprobs without
        computing gradients.

        Args:
            model_id: Model identifier (for logging)
            train_group: Slime RayTrainGroup instance
            data: List of forward data samples
            loss_fn: Loss function type
            client_info: Client metadata (contains rollout_manager for offload)

        Returns:
            Dict with forward results in Tinker format

        Raises:
            Exception: If forward pass fails
        """
        logger.info(f"Forward pass for {model_id}")

        # Offload SGLang before forward pass to free GPU memory (only if offload_rollout enabled)
        rollout_manager = client_info.get("rollout_manager") if client_info else None
        args = client_info.get("args") if client_info else None
        offload_rollout = args.offload_rollout if args else True
        if rollout_manager is not None and offload_rollout:
            logger.info(f"Offloading SGLang for {model_id} before forward pass")
            await asyncio.to_thread(lambda: ray.get(rollout_manager.offload.remote()))
            logger.info(f"SGLang offloaded for {model_id}")

        # Convert Tinker data to Slime rollout format
        rollout_data = self.converter.forward_to_rollout(data)

        # Call Slime forward_only (no gradients)
        results = await asyncio.to_thread(
            train_group.forward_only,
            rollout_id=0,
            rollout_data_ref=Box(ray.put(rollout_data))
        )

        # Convert results back to Tinker format
        result = self.converter.rollout_to_forward_result(
            results,
            loss_fn=loss_fn,
            rollout_data=rollout_data,
            original_data=data
        )

        logger.info(f"Forward pass completed for {model_id}")
        return result

    async def forward_backward(
        self,
        model_id: str,
        train_group: Any,
        args: Any,
        data: List[Any],
        loss_fn: str,
        client_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute forward-backward pass (accumulate gradients, no optimizer step).

        Args:
            model_id: Model identifier (for logging)
            train_group: Slime RayTrainGroup instance
            args: Megatron args from Slime
            data: List of training data samples
            loss_fn: Loss function type
            client_info: Client metadata (contains rollout_manager for offload)

        Returns:
            Dict with loss and gradient norm

        Raises:
            ValueError: If validation fails
            Exception: If training fails
        """
        print(f"[DEBUG] Forward-backward pass for {model_id}", flush=True)
        logger.info(f"Forward-backward pass for {model_id}")

        # Offload SGLang before training to free GPU memory (only if offload_rollout enabled)
        rollout_manager = client_info.get("rollout_manager") if client_info else None
        offload_rollout = args.offload_rollout if args else True
        print(f"[DEBUG] rollout_manager = {rollout_manager}, offload_rollout = {offload_rollout}", flush=True)
        if rollout_manager is not None and offload_rollout:
            print(f"[DEBUG] Offloading SGLang for {model_id} before training...", flush=True)
            logger.info(f"Offloading SGLang for {model_id} before training")
            await asyncio.to_thread(lambda: ray.get(rollout_manager.offload.remote()))
            print(f"[DEBUG] SGLang offloaded for {model_id}", flush=True)
            logger.info(f"SGLang offloaded for {model_id}")
        elif rollout_manager is None:
            print(f"[DEBUG] No rollout_manager found for {model_id}, skipping SGLang offload", flush=True)
            logger.warning(f"No rollout_manager found for {model_id}, skipping SGLang offload")

        # Determine training mode (RL vs SFT)
        is_rl = not args.debug_train_only

        # Detect legacy test format that needs fake data generation
        # Original logic: if not data or (len(data) == 1 and isinstance(data[0], dict) and "input" in data[0])
        # After Pydantic validation, we need to check if data is insufficient for DP requirements
        needs_fake_data = (
            not data or
            len(data) == 0 or
            len(data) < args.data_parallel_size  # Insufficient samples for DP
        )

        # Handle legacy HTTP test format - generate fake test data like original api.py did
        if needs_fake_data:
            logger.info(f"[forward_backward] Insufficient data ({len(data) if data else 0} samples, need {args.data_parallel_size}) - generating fake test data for backward compatibility")
            import torch

            seq_length = args.seq_length
            vocab_size = args.vocab_size
            batch_size = args.global_batch_size
            prompt_length = seq_length // 2
            response_length = seq_length - prompt_length
            device = torch.device("cpu")

            tokens_list = []
            loss_masks_list = []
            response_lengths_list = []
            advantages_list = []
            log_probs_list = []
            ref_log_probs_list = []
            returns_list = []
            values_list = []

            for i in range(batch_size):
                sample_tokens = torch.randint(0, vocab_size, (seq_length,), dtype=torch.long, device=device)
                tokens_list.append(sample_tokens)
                sample_loss_mask = torch.ones(response_length, dtype=torch.float32, device=device)
                loss_masks_list.append(sample_loss_mask)
                response_lengths_list.append(response_length)
                sample_advantages = torch.randn(response_length, dtype=torch.float32, device=device) * 0.1
                advantages_list.append(sample_advantages)
                sample_log_probs = torch.randn(response_length, dtype=torch.float32, device=device) * 0.5 - 5.0
                log_probs_list.append(sample_log_probs)
                sample_ref_log_probs = torch.randn(response_length, dtype=torch.float32, device=device) * 0.5 - 5.0
                ref_log_probs_list.append(sample_ref_log_probs)
                sample_returns = torch.randn(response_length, dtype=torch.float32, device=device) * 0.5
                returns_list.append(sample_returns)
                sample_values = torch.randn(response_length, dtype=torch.float32, device=device) * 0.5
                values_list.append(sample_values)

            # Build rollout_data directly (skip converter)
            rollout_data = {
                "tokens": tokens_list,
                "loss_masks": loss_masks_list,
                "response_lengths": response_lengths_list,
                "advantages": advantages_list,
                "log_probs": log_probs_list,
                "ref_log_probs": ref_log_probs_list,
                "values": values_list,
                "returns": returns_list
            }
        else:
            # Normal path: validate and convert data
            validator = RequestValidator(
                args,
                allow_partial_batches=self.allow_partial_batches
            )
            validation_error = validator.validate_forward_backward_request(
                data,
                is_rl=is_rl
            )
            if validation_error:
                raise ValueError(
                    f"Request validation failed:\n{validation_error}\n\n"
                    f"{validator.get_config_summary()}"
                )

            # Convert data
            rollout_data = self.converter.forward_backward_to_rollout(
                data,
                is_rl=is_rl
            )

        # Debug: Log sample count
        num_samples = len(rollout_data.get("tokens", []))
        logger.info(f"Forward-backward with {num_samples} samples")

        # NOTE: Megatron log_probs computation now happens INSIDE Miles' forward_backward_step_only().
        # When use_rollout_logprobs=False (default), Miles computes Megatron log_probs internally
        # BEFORE training, matching the behavior of train_actor(). This is necessary for:
        # 1. Correct PPO ratio computation (comparing Megatron old vs new log_probs)
        # 2. Proper TIS (Truncated Importance Sampling) which compares Megatron vs SGLang log_probs
        #
        # The previous approach of calling forward_only externally had aggregation issues:
        # - Each DP rank processes a different subset of samples with internal reordering
        # - Gathered results don't maintain proper global sample order
        # - This caused log_probs vs rollout_log_probs size mismatch in TIS
        #
        # With the internal computation, each actor:
        # 1. Gets its local subset of samples via get_data_iterator()
        # 2. Computes Megatron log_probs via compute_log_prob() for its local samples
        # 3. Updates its local rollout_data["log_probs"] in-place
        # 4. Runs forward_backward with consistent ordering
        #
        # This matches how native Miles train_actor() works.

        # Call Slime forward_backward_only
        results = await asyncio.to_thread(
            train_group.forward_backward_only,
            rollout_id=0,
            rollout_data_ref=Box(ray.put(rollout_data))
        )

        # Convert results (pass ORIGINAL data from Tinker for correct weights lengths)
        result = self.converter.rollout_to_forward_backward_result(
            results,
            loss_fn=loss_fn,
            rollout_data=rollout_data,
            original_data=data  # Pass original Tinker request for weight lengths
        )

        logger.info(f"Forward-backward completed for {model_id}")
        return result

    async def apply_optimizer_step(
        self,
        model_id: str,
        train_group: Any,
        client_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply optimizer step to update model weights.

        Args:
            model_id: Model identifier (for logging)
            train_group: Slime RayTrainGroup instance
            client_info: Client metadata (contains optimizer, rollout_manager)

        Returns:
            Dict with success status, grad_norm, and learning_rates

        Raises:
            Exception: If optimizer step fails
        """
        logger.info(f"Optimizer step for {model_id}")

        args = client_info.get("args")
        offload_train = args.offload_train if args else True
        offload_rollout = args.offload_rollout if args else True
        rollout_manager = client_info.get("rollout_manager")

        if not offload_train and not offload_rollout:
            # Simple path: no offload, use combined method
            logger.info(f"Applying optimizer step and syncing weights for {model_id}")
            results = await asyncio.to_thread(train_group.apply_optimizer_step_and_sync)
            logger.info(f"Weights synced to SGLang for {model_id}")
        else:
            # Offload path: need interleaved offload/onload calls
            results = await asyncio.to_thread(train_group.apply_optimizer_step)

            if rollout_manager is not None:
                from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
                try:
                    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
                except ImportError:
                    GPU_MEMORY_TYPE_CUDA_GRAPH = None

                # Step 1: Offload Megatron model to free GPU memory for SGLang
                if offload_train:
                    logger.info(f"Offloading Megatron model for {model_id}")
                    await asyncio.to_thread(train_group.offload)

                # Step 2: Onload SGLang weights (needed for update_weights)
                if offload_rollout:
                    logger.info(f"Onloading SGLang weights for {model_id}")
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])))

                # Step 3: Sync updated weights from Megatron to SGLang
                logger.info(f"Pushing updated weights to SGLang for {model_id}")
                await asyncio.to_thread(train_group.update_weights)
                logger.info(f"Weights synced to SGLang for {model_id}")

                # Step 4: Onload CUDA graphs and KV cache
                if offload_rollout:
                    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                        logger.info(f"Onloading SGLang CUDA graphs for {model_id}")
                        await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])))

                    logger.info(f"Onloading SGLang KV cache for {model_id}")
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])))

                    logger.info(f"SGLang fully onloaded for {model_id}")

        # Extract learning rates
        learning_rates = extract_learning_rates(client_info.get("optimizer"))

        # Build result
        result = {
            "success": results[0]["success"],
            "grad_norm": results[0]["grad_norm"],
            "learning_rates": learning_rates,
            "model_id": model_id
        }

        logger.info(f"Optimizer step completed for {model_id}")
        return result
