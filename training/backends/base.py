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
