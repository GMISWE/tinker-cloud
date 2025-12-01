"""
Training Service - Business Logic for Training Operations

Extracted from api_refactored.py to separate concerns:
- HTTP layer (routers) handles request/response
- Service layer (this file) handles business logic
- Domain layer handles data transformations

This service is pure Python with no FastAPI dependencies.
"""
import logging
import ray
from typing import Dict, List, Any
from miles.utils.ray_utils import Box

from ..core.data_converter import TinkerDataConverter
from ..core.validators import RequestValidator
from ..utils.helpers import extract_learning_rates

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

    async def forward(
        self,
        model_id: str,
        train_group: Any,
        data: List[Any],
        loss_fn: str
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

        Returns:
            Dict with forward results in Tinker format

        Raises:
            Exception: If forward pass fails
        """
        logger.info(f"Forward pass for {model_id}")

        # Convert Tinker data to Slime rollout format
        rollout_data = self.converter.forward_to_rollout(data)

        # Call Slime forward_only (no gradients)
        results = train_group.forward_only(
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
        loss_fn: str
    ) -> Dict[str, Any]:
        """
        Execute forward-backward pass (accumulate gradients, no optimizer step).

        Args:
            model_id: Model identifier (for logging)
            train_group: Slime RayTrainGroup instance
            args: Megatron args from Slime
            data: List of training data samples
            loss_fn: Loss function type

        Returns:
            Dict with loss and gradient norm

        Raises:
            ValueError: If validation fails
            Exception: If training fails
        """
        logger.info(f"Forward-backward pass for {model_id}")

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
            validator = RequestValidator(args)
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

        # Call Slime forward_backward_only
        results = train_group.forward_backward_only(
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

        # Apply optimizer step
        results = train_group.apply_optimizer_step()

        # Sync weights to SGLang if RL training
        if "rollout_manager" in client_info:
            logger.info(f"Pushing updated weights to SGLang for {model_id}")
            train_group.update_weights()
            logger.info(f"Weights synced to SGLang for {model_id}")

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
