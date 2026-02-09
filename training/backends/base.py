"""
Abstract base classes for the backend abstraction layer.

Defines TrainingBackend, ArgumentBuilder, DataConverter, BackendHandle,
and BackendError — the contracts that Miles and NeMo RL implementations
must satisfy.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BackendError(Exception):
    """Common error type for all backend operations."""

    def __init__(
        self,
        message: str,
        backend: str,
        operation: str,
        original_error: Optional[Exception] = None,
    ):
        self.backend = backend
        self.operation = operation
        self.original_error = original_error
        detail = f"[{backend}] {operation}: {message}"
        if original_error:
            detail += f" (caused by {type(original_error).__name__}: {original_error})"
        super().__init__(detail)


@dataclass
class BackendHandle:
    """Opaque handle to backend-specific runtime state."""

    model_id: str
    backend_type: str  # "miles" | "nemo_rl"


class TrainingBackend(ABC):
    """
    Contract for post-training backends.

    TinkerCloud services call these methods; backend implementations
    translate to Miles or NeMo RL native APIs.
    """

    @abstractmethod
    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False,
        checkpoint_path: Optional[str] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> BackendHandle:
        """
        Initialize training actors and inference engine.

        Returns:
            BackendHandle with backend-specific state.

        Raises:
            BackendError: If actors fail to initialize.
        """
        ...

    @abstractmethod
    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        """
        Forward-only pass (no gradient computation).

        Returns:
            {"loss_fn_outputs": [...], "metrics": {...}}
        """
        ...

    @abstractmethod
    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        """
        Accumulate gradients for the given data.

        Multiple forward_backward calls accumulate gradients before
        a single apply_optimizer_step call.

        Backend behavior:
        - Miles: Executes forward+backward immediately. Returns real
          loss, logprobs, and metrics. deferred=False.
        - NeMo RL: Buffers data (no GPU work). Returns deferred result.
          Real training happens at apply_optimizer_step().

        Deferred result contract (NeMo RL):
            When deferred=True, the response contains:
            - "metrics": {} (empty dict — no training has occurred)
            - "loss_fn_outputs": [] (empty list — logprobs not yet computed)
            - "deferred": True
            Real metrics and logprobs are returned by apply_optimizer_step()
            as a single batch covering all buffered forward_backward calls.

        Logprobs resolution (CHK011):
            NeMo RL computes training logprobs at optim_step time, not
            per forward_backward call. Clients that need per-minibatch
            logprobs (e.g., tinker-cookbook compute_kl_sample_train) should
            use get_logprobs() for a separate forward-only pass, or accept
            batch-level logprobs from apply_optimizer_step().

        Returns:
            {
                "loss_fn_outputs": [...],
                "metrics": {...},
                "deferred": bool,  # True if metrics are estimated
            }
        """
        ...

    @abstractmethod
    async def apply_optimizer_step(
        self,
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply accumulated gradients via optimizer step, then sync
        weights to inference engine.

        NeMo RL behavior:
            Concatenates all buffered forward_backward data and calls
            policy.train() once. NeMo RL internally micro-batches the
            concatenated data according to train_micro_batch_size.

        Ordering guarantees (CHK019):
            Buffered microbatches are concatenated in FIFO order (order
            of forward_backward calls). However, NeMo RL may internally
            process micro-batches in any order. For GRPO training, micro-
            batch ordering does not affect correctness — each sample's
            advantage is pre-computed and independent.

        Buffer-to-batch-size relationship (CHK027):
            NeMo RL processes ALL provided data regardless of
            train_global_batch_size config. If 4 forward_backward calls
            buffer 1024 samples each (4096 total) but train_global_batch_size
            is 2048, NeMo RL processes all 4096 samples using micro-batching.
            Ensure train_global_batch_size matches expected total buffered
            samples to avoid unexpected gradient accumulation behavior.

        Failure recovery (CHK024):
            The buffer is cleared BEFORE policy.train() executes. If
            policy.train() fails (e.g., OOM), buffered data is lost.
            The client must re-send all forward_backward() data and retry.
            There is no automatic retry mechanism.

        Returns:
            {"success": bool, "grad_norm": float, "metrics": {...}}
        """
        ...

    @abstractmethod
    async def update_inference_weights(
        self,
        handle: BackendHandle,
    ) -> None:
        """Sync training weights to inference engine without optimizer step."""
        ...

    @abstractmethod
    async def save_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        """
        Save model checkpoint.

        Returns:
            Actual path where checkpoint was saved.
        """
        ...

    @abstractmethod
    async def load_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
    ) -> None:
        """
        Load model weights from a checkpoint.

        After loading, syncs weights to the inference engine via
        update_inference_weights / refit_policy_generation.

        Args:
            handle: Backend handle returned by create_model.
            checkpoint_path: Path to the checkpoint directory.

        Raises:
            BackendError: If checkpoint loading fails.
        """
        ...

    @abstractmethod
    async def delete_model(
        self,
        handle: BackendHandle,
    ) -> None:
        """Release all GPU resources and Ray actors."""
        ...

    @abstractmethod
    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        """
        Compute log probabilities for given data using training model.

        Returns:
            List of per-sample logprob tensors.
        """
        ...


class UnsupportedFeatureError(BackendError):
    """Raised when a backend-specific feature is requested on the wrong backend."""

    def __init__(self, feature: str, backend: str, suggestion: str = ""):
        msg = f"Feature {feature!r} is not supported on the {backend} backend"
        if suggestion:
            msg += f". {suggestion}"
        super().__init__(msg, backend=backend, operation="feature_check")


class ArgumentBuilder(ABC):
    """
    Translates HuggingFace model config + user settings into
    backend-native configuration.
    """

    @abstractmethod
    def build_args(
        self,
        base_model: str,
        num_gpus: int = 4,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Generate backend-specific configuration.

        Returns:
            Miles: argparse.Namespace (Megatron CLI args)
            NeMo RL: dict (MasterConfig TypedDict)
        """
        ...


class DataConverter(ABC):
    """
    Converts between canonical Tinker Datum format and
    backend-native training data format.
    """

    @abstractmethod
    def forward_to_backend(
        self,
        data: List[Dict],
        args: Any,
    ) -> Any:
        """
        Convert Tinker Datum list to backend format for forward pass.

        Returns:
            Miles: dict with List[Tensor] values (rollout_data)
            NeMo RL: BatchedDataDict with Tensor [B, S] values
        """
        ...

    @abstractmethod
    def forward_backward_to_backend(
        self,
        data: List[Dict],
        loss_fn: str,
        args: Any,
    ) -> Any:
        """
        Convert Tinker Datum list to backend format for training.

        Returns:
            Backend-native training data.
        """
        ...

    @abstractmethod
    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Convert backend forward result to Tinker ForwardOutput format.

        Returns:
            {"loss_fn_outputs": [...], "metrics": {...}}
        """
        ...

    @abstractmethod
    def backend_to_forward_backward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Convert backend training result to Tinker ForwardBackwardOutput.

        Returns:
            {"loss_fn_outputs": [...], "metrics": {...}}
        """
        ...
