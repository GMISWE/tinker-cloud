"""
Request Validators for Tinker API → Slime Backend

Validates user requests before invoking Slime operations to provide clear
error messages and prevent cryptic backend failures.
"""
import logging
from typing import List, Optional, Any
from argparse import Namespace

logger = logging.getLogger(__name__)

class RequestValidator:
    """
    Validates training requests against Slime backend requirements.

    Primary purpose: Catch user misconfigurations early with actionable error messages.
    """

    def __init__(self, slime_args: Namespace, allow_partial_batches: bool = False):
        """
        Initialize validator with Slime configuration.

        Args:
            slime_args: Megatron args from Slime backend containing:
                - data_parallel_size: Number of DP workers
                - global_batch_size: Global batch size
                - balance_data: Whether to use balanced data partitioning
                - n_samples_per_prompt: Samples per prompt (for RL training)
        """
        self.dp_size = slime_args.data_parallel_size
        self.global_batch_size = slime_args.global_batch_size
        self.balance_data = getattr(slime_args, 'balance_data', False)
        self.n_samples_per_prompt = getattr(slime_args, 'n_samples_per_prompt', 1)
        self.allow_partial_batches = allow_partial_batches

    def validate_sample_count(
        self,
        num_samples: int,
        is_rl: bool = False
    ) -> Optional[str]:
        """
        Validate that sample count is compatible with DP configuration.

        Args:
            num_samples: Number of samples in the request
            is_rl: Whether this is RL training (requires group alignment)

        Returns:
            Error message if validation fails, None if valid
        """
        # Check 1: Minimum samples (must have at least DP size samples)
        if num_samples < self.dp_size:
            return (
                f"Invalid sample count: Received {num_samples} sample(s) but "
                f"data parallel size is {self.dp_size}.\n"
                f"Required: At least {self.dp_size} samples (one per DP worker).\n"
                f"Suggestion: Send {self.dp_size}, {self.dp_size * 2}, or {self.dp_size * 4} samples."
            )

        # Check 2: DP divisibility (samples must distribute evenly across DP workers)
        if num_samples % self.dp_size != 0:
            if self.allow_partial_batches:
                logger.warning(
                    "ALLOW_PARTIAL_BATCHES enabled: proceeding with %s samples (dp=%s). "
                    "Miles will rely on dynamic global batch scaling.",
                    num_samples,
                    self.dp_size,
                )
            else:
                next_valid = ((num_samples // self.dp_size) + 1) * self.dp_size
                prev_valid = (num_samples // self.dp_size) * self.dp_size
                return (
                    f"Invalid sample count: Received {num_samples} samples but "
                    f"data parallel size is {self.dp_size}.\n"
                    f"Required: Sample count must be divisible by {self.dp_size}.\n"
                    f"Suggestion: Use {prev_valid} or {next_valid} samples instead."
                )

        # Check 3: Global batch size divisibility
        if num_samples % self.global_batch_size != 0:
            next_valid = ((num_samples // self.global_batch_size) + 1) * self.global_batch_size
            logger.warning(
                "Forward/backward request size (%s) is not divisible by configured global_batch_size (%s). "
                "Proceeding with gradient accumulation semantics; consider using %s samples for even scaling.",
                num_samples,
                self.global_batch_size,
                next_valid,
            )

        # Check 4: RL group alignment (if using balanced data with grouped samples)
        if (
            is_rl
            and self.balance_data
            and self.n_samples_per_prompt > 1
            and not self.allow_partial_batches
        ):
            required_multiple = self.n_samples_per_prompt * self.dp_size
            if num_samples % required_multiple != 0:
                next_valid = ((num_samples // required_multiple) + 1) * required_multiple
                return (
                    f"Invalid sample count for RL training: Received {num_samples} samples.\n"
                    f"Required: Sample count must be divisible by "
                    f"(n_samples_per_prompt × dp_size) = ({self.n_samples_per_prompt} × {self.dp_size}) = {required_multiple}.\n"
                    f"Suggestion: Use {next_valid} samples."
                )

        # All checks passed
        return None

    def validate_forward_backward_request(
        self,
        data: List[Any],
        is_rl: bool = False
    ) -> Optional[str]:
        """
        Validate a forward_backward request.

        Args:
            data: List of training data samples
            is_rl: Whether this is RL training

        Returns:
            Error message if validation fails, None if valid
        """
        num_samples = len(data)

        # Validate sample count against DP configuration
        sample_count_error = self.validate_sample_count(num_samples, is_rl)
        if sample_count_error:
            return sample_count_error

        # Future: Add more validations here
        # - Validate tensor data format
        # - Validate sequence lengths
        # - Validate required fields for SFT vs RL

        return None

    def get_config_summary(self) -> str:
        """Get a summary of current Slime configuration for debugging."""
        return (
            f"Slime Configuration:\n"
            f"  - Data Parallel Size: {self.dp_size}\n"
            f"  - Global Batch Size: {self.global_batch_size}\n"
            f"  - Balance Data: {self.balance_data}\n"
            f"  - Samples Per Prompt: {self.n_samples_per_prompt}\n"
            f"  - Min Samples Required: {self.dp_size}\n"
            f"  - Valid Sample Counts: {self.dp_size}, {self.dp_size * 2}, {self.dp_size * 4}, ..."
        )
