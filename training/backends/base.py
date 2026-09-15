"""
Abstract base classes for the backend abstraction layer.

Defines TrainingBackend, ArgumentBuilder, DataConverter, BackendHandle,
and BackendError — the contracts that Miles and NeMo RL implementations
must satisfy.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional


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
    # Inference-engine context window in tokens: a sample() must satisfy
    # len(prompt) + max_tokens <= context_length. Each backend fills it at
    # create_model from the number its engine was actually booted with; None
    # means unknown and the service skips the check.
    context_length: Optional[int] = None
    # Optimizer steps applied since create; pins samplers and checkpoint records.
    weight_version: int = 0


class TrainingBackend(ABC):
    """
    Contract for post-training backends.

    TinkerCloud services call these methods; backend implementations
    translate to Miles or NeMo RL native APIs.
    """

    # False for in-process backends (no Ray actors); the server then skips ray.init.
    needs_ray: bool = True
    # Loss names (core.loss_registry) this backend can train; TrainingService
    # rejects others with UnsupportedFeatureError before any GPU work.
    SUPPORTED_LOSS_FNS: FrozenSet[str] = frozenset()

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
        resume_from: Optional[Path] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        staleness_k: int = 0,
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> BackendHandle:
        """
        Initialize training actors and inference engine.

        Checkpoint directories come resolved from the checkpoint store
        (training/checkpoints); a backend never sees a tinker:// URI.
        `resume_from` is the root of a completed `weights` checkpoint to load
        weights-only from (fresh optimizer, RNG and iteration count).
        `native_root` is this model's private directory under the checkpoint
        base -- the place for anything the engine writes on its own terms
        (Megatron --save, per-adapter step checkpoints); the service always
        passes it, and it outlives the model so a later resume can find what
        was written there.

        The objective axis (feature 004) is additive: language_modeling is the
        default and leaves the existing causal path unchanged. LM-only backends
        (Miles, NeMo RL) must reject classification objectives with a
        BackendError; classification backends (Automodel, Megatron-Bridge)
        require num_labels. See specs/004-bionemo-classification/plan.md.

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
        loss_fn_config: Optional[Dict[str, float]] = None,
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
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Accumulate gradients for the given data.

        `loss_fn_config` carries the per-call hyperparameters validated by
        core.loss_registry (e.g. PPO clip thresholds); a backend that cannot
        honour a given value must raise UnsupportedFeatureError, never ignore it.

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
            The EMPTY loss_fn_outputs list is the deferred signal as seen by
            clients — the "deferred" flag itself does not cross the wire, so
            a backend must not return placeholder logprobs when deferring.
            Real metrics and logprobs are returned by apply_optimizer_step()
            as a single batch covering all buffered forward_backward calls
            (the aggregate loss arrives as metrics["total_loss"] there, which
            clients prefer when fb returned no logprobs).

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
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply accumulated gradients via optimizer step, then sync
        weights to inference engine.

        adam_params (P4): full client AdamParams (beta1/beta2/eps/weight_decay/
        grad_clip_norm). Backends honor what they can and MUST warn on values
        they cannot apply.

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

            Client-visibility rule: any value SDK clients must see goes in
            "metrics" — the SDK's response models follow the upstream tinker
            shapes and silently drop unknown TOP-LEVEL fields (top level
            reaches raw-HTTP callers only). The service layer mirrors
            grad_norm into metrics; a backend adding a new client-facing
            value must put it in metrics itself.
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
        root: Path,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        """
        Write a checkpoint under `root` (an existing, empty directory the store
        created). What goes inside is the backend's: its native format wherever
        the engine can write one, optimizer state wherever it can be restored,
        and the cross-backend interchange adapter (training/checkpoints/
        interchange.py) so another engine can import it. `step` is the store's
        per-model counter for this save, unique for the life of the model --
        the label for counter-named native artifacts (Megatron iter_*). It is
        None, and `persist` False, for an ephemeral sampler save: the weights
        already reached the inference engine and nothing need be written.

        Raises:
            BackendError: If the checkpoint cannot be written; the store then
                marks the save failed and nothing under `root` is trusted.
        """
        ...

    @abstractmethod
    async def load_checkpoint(
        self,
        handle: BackendHandle,
        root: Path,
        optimizer: bool = False,
    ) -> None:
        """
        Load the checkpoint under `root` (a completed `weights` checkpoint, as
        resolved by the store) into a live model.

        `optimizer=False` restores weights only and leaves the model's optimizer
        state as it is (fresh on a model that has not trained). `optimizer=True`
        also restores the optimizer state the checkpoint carries, and MUST raise
        BackendError when the checkpoint has none or the backend cannot restore
        it: never a silent partial resume. create_model(checkpoint_path=...) is
        the weights-only load; save_checkpoint writes optimizer state wherever
        the backend can.

        After loading, syncs weights to the inference engine via
        update_inference_weights / refit_policy_generation.

        Args:
            handle: Backend handle returned by create_model.
            root: The checkpoint directory.
            optimizer: Also restore optimizer state.

        Raises:
            BackendError: If checkpoint loading fails, or optimizer state was
                requested and is unavailable.
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

    @abstractmethod
    async def sample(
        self,
        handle: BackendHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate num_samples completions for one prompt via the backend's
        inference engine.

        Backend behavior:
        - Miles: per-sample HTTP calls to the SGLang router.
        - NeMo RL: requests are batch-accumulated and flushed as a single
          Policy.generate() call (PERF-002).

        Returns:
            {
                "sequences": [
                    {"tokens": [...], "logprobs": [...],
                     "text": Optional[str], "stop_reason": "stop" | "length"},
                    ...
                ],
                "prompt_logprobs": Optional[List],  # [None, lp1, ...] when requested
            }

        Raises:
            BackendError: If the inference engine is unavailable.
        """
        ...

    @abstractmethod
    async def prepare_for_generation(
        self,
        handle: BackendHandle,
    ) -> None:
        """
        Ensure the inference engine is ready to serve sampling requests.

        Backend behavior:
        - Miles: validates the SGLang router is available (no state change).
        - NeMo RL: safety-net refit + wake if the engine was left in
          training state (fast no-op when already generation-ready).

        Raises:
            BackendError: If the engine cannot be made ready.
        """
        ...


class SampleRequestError(BackendError):
    """A sample() request the engine cannot honor as stated (missing max_tokens,
    prompt + max_tokens past the context window). Terminal: the future fails
    with this message and the client sees HTTP 400, never an engine crash."""

    def __init__(self, message: str, backend: str):
        super().__init__(message, backend=backend, operation="sample")


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
