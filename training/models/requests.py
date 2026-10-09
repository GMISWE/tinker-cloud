"""
Request models for the training API.

This module defines Pydantic models for all API request payloads,
providing validation and documentation.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, validator, model_validator

# loss_fn_config values: numbers, or text on the keys core.loss_registry declares
# as text (the SDK's loss_fn_config_v2 carries both).
LossFnConfig = Dict[str, Union[float, str]]


class LoraConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = Field(default=0, ge=0, description="LoRA rank (0 = no LoRA)")
    alpha: Optional[int] = Field(default=None, ge=0, description="LoRA alpha parameter (defaults to rank if not set)")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="LoRA dropout rate")
    seed: Optional[int] = Field(default=None, description="Random seed for LoRA")
    train_unembed: bool = Field(default=True, description="Train unembedding layer")
    train_mlp: bool = Field(default=True, description="Train MLP layers")
    train_attn: bool = Field(default=True, description="Train attention layers")


class ParallelismConfig(BaseModel):
    """Model parallelism configuration."""

    tensor_parallel_size: int = Field(default=1, ge=1, le=8, description="Tensor parallelism degree")
    pipeline_parallel_size: int = Field(default=1, ge=1, le=8, description="Pipeline parallelism degree")
    num_gpus: Optional[int] = Field(default=None, ge=1, le=128, description="Total number of GPUs (auto-detected if not set)")


SERVED_OPTIMIZER = "adamw"


class OptimizerConfig(BaseModel):
    """Optimizer identity, fixed at model creation (SDK `optimizer_config`).

    Only `adamw` is served; the router rejects any other `type` with 400
    (API-CONTRACT: Dimuon UNSUPPORTED). Extra keys are kept so the
    rejection message can echo what the client sent."""
    model_config = ConfigDict(extra="allow")

    type: str = Field(default=SERVED_OPTIMIZER, description="Optimizer family")


class OptimParams(BaseModel):
    """Per-step parameters of a non-Adam optimizer (the SDK sends them as
    `optimizer_params`; Adam keeps the `adam_params` key). Parsed only to be
    rejected with a message that names the family."""
    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="Optimizer family")


class CreateModelRequest(BaseModel):
    """Request to create a new training client."""

    # Session tracking (required for session management)
    session_id: str = Field(..., description="Session ID (required)")
    model_seq_id: int = Field(..., description="Model sequence ID within session (required)")
    user_metadata: Optional[Dict[str, Any]] = Field(default=None, description="User-provided metadata")

    # Model configuration. Everything the server can read from the model's own
    # config or its backend configuration is not on the wire (specs/025, D17):
    # context length, batch shape, classification head, debug/staleness knobs.
    base_model: str = Field(..., description="HF model id or local model directory")
    lora_config: Optional[LoraConfig] = Field(default=None, description="LoRA configuration")
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer family; only adamw is served")
    parallelism_config: Optional[ParallelismConfig] = Field(default=None, description="Parallelism settings (server-side callers only)")


class DeleteModelRequest(BaseModel):
    """Request to delete a training client."""

    model_id: str = Field(..., description="Model ID to delete")


class UnloadModelRequest(BaseModel):
    """Request to unload a model (Tinker SDK compatible).

    This is the Tinker-standard way to release model resources.
    Functionally equivalent to DeleteModelRequest.
    """

    model_id: str = Field(..., description="Model ID to unload")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")
    type: str = Field(default="unload_model", description="Request type")


class SaveWeightsRequest(BaseModel):
    """Request to save model weights."""

    model_id: str = Field(..., description="Model ID")
    path: Optional[str] = Field(default=None, description="Checkpoint name/path")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")


# ttl bounds the SDK documents for save_weights_external: 1 hour to 10 years.
EXTERNAL_TTL_MIN_S = 3600
EXTERNAL_TTL_MAX_S = 10 * 365 * 24 * 3600


class SaveWeightsExternalRequest(BaseModel):
    """Request to export model weights in HF format (SDK SaveWeightsExternalRequest)."""

    model_id: str = Field(..., description="Model ID")
    path: str = Field(..., min_length=1, description="Name of the external weights checkpoint")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")
    ttl_seconds: Optional[int] = Field(
        default=None, ge=EXTERNAL_TTL_MIN_S, le=EXTERNAL_TTL_MAX_S,
        description="Checkpoint lifetime in seconds (None = never expires)",
    )
    age_encryption_recipients: Optional[List[str]] = Field(
        default=None, description="age recipients; this server does not encrypt and rejects a non-empty list",
    )
    type: Literal["save_weights_external"] = "save_weights_external"


class LoadWeightsRequest(BaseModel):
    """Request to load model weights (permitted only as a model's first request)."""

    model_id: str = Field(..., description="Model ID")
    path: str = Field(..., description="tinker://<run>/weights/<name> to load from")
    optimizer: bool = Field(default=False, description="Also restore optimizer state (the checkpoint must carry it)")
    optimizer_config: Optional[OptimizerConfig] = Field(default=None, description="Optimizer family for the restored run; only adamw is served")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for ordering")


class RetrieveFutureRequest(BaseModel):
    """Request to retrieve async operation result."""

    request_id: str = Field(..., description="Request ID to retrieve")
    # SDK >= 0.25 sends this; results are always returned inline, so it is accepted and ignored.
    allow_metadata_only: bool = Field(default=False, description="Accepted for SDK compatibility; no effect")


class SamplingParams(BaseModel):
    """Sampling parameters for text generation."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, gt=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=-1, description="Top-k sampling (-1 for no limit)")
    # Required: the SDK drops a None max_tokens on the wire, and a server default
    # silently truncated every turn to 256; no cap — the model context is the
    # only limit, enforced per request (SampleRequestError) against the engine.
    max_tokens: int = Field(..., ge=1, description="Maximum tokens to generate (required)")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    stop_token_ids: Optional[List[int]] = Field(default=None, description="Stop token IDs")
    # SDK SamplingParams carries seed; without this field pydantic silently
    # drops it and seeded sampling is non-reproducible (found in verl M2 G8)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible generation")

    @model_validator(mode="before")
    def convert_integer_stop_to_token_ids(cls, values):
        """
        If integers were passed to 'stop', move them to 'stop_token_ids'.
        This handles the common case where clients pass token IDs to the stop field.
        """
        stop = values.get("stop")
        if stop is not None and stop and isinstance(stop[0], int):
            # Move integers from stop to stop_token_ids
            existing_stop_token_ids = values.get("stop_token_ids", [])
            if existing_stop_token_ids is None:
                existing_stop_token_ids = []
            values["stop_token_ids"] = existing_stop_token_ids + stop
            values["stop"] = None  # Clear the stop field
        return values


class GetInfoRequest(BaseModel):
    """Request for model information."""

    model_id: str = Field(..., description="Model ID")

class SamplingSessionFuturesTarget(BaseModel):
    """Poll target of /retrieve_futures: one (possibly cloned) sampling session.
    `cloned_sampler_id` is seq_id // 1_000_000_000 (0 for the original client). """
    type: Literal["sampling_session"] = "sampling_session"
    sampling_session_id: str = Field(..., description="Sampling session the samples were submitted under")
    cloned_sampler_id: int = Field(default=0, ge=0, description="seq_id block of the (cloned) SamplingClient")

class SessionFuturesPollRequest(BaseModel):
    """Per session completion poll (SDK type FuturesRetrieveRequest). """
    target: SamplingSessionFuturesTarget
    prev_cursor: int = Field(default=0, ge=0, description="Cursor from the preious response; entries below it are acknowledged")
    timeout: Optional[float] = Field(default=None, ge=0, description="Requested hold in seconds; the server caps it")

class CancelFutureRequest(BaseModel):
    """Cancel an in-flight sample the SDK has abandaned."""
    request_id: str = Field(..., description="Sample request to cancel")

class CleanupFuturesRequest(BaseModel):
    """Request to cleanup old futures."""

    max_age_hours: int = Field(default=24, ge=0, description="Maximum age in hours")


class TelemetryRequest(BaseModel):
    """Telemetry data submission."""

    event_type: str = Field(..., description="Type of telemetry event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")


# ============= Tensor Data Models =============

class TensorData(BaseModel):
    """Tensor serialization format for Tinker API."""
    data: List[Any] = Field(..., description="Tensor data (flattened)")
    shape: Optional[List[int]] = Field(default=None, description="Tensor shape")
    dtype: Optional[str] = Field(default=None, description="Data type")


# ============= Forward/Forward-Backward Models =============

class LossFnInputs(BaseModel):
    """Base class for loss function inputs."""
    pass


class RLLossFnInputs(LossFnInputs):
    """RL training loss inputs (PPO/GRPO)."""
    target_tokens: TensorData = Field(..., description="Target token IDs")
    logprobs: TensorData = Field(..., description="Old action log probabilities")
    advantages: TensorData = Field(..., description="Advantage estimates")
    mask: Optional[TensorData] = Field(default=None, description="Loss mask")
    ref_logprobs: Optional[TensorData] = Field(default=None, description="Reference logprobs")
    values: Optional[TensorData] = Field(default=None, description="Value estimates")
    returns: Optional[TensorData] = Field(default=None, description="Returns")


class SFTLossFnInputs(LossFnInputs):
    """Supervised fine-tuning loss inputs."""
    target_tokens: Optional[TensorData] = Field(default=None, description="Target token IDs")
    target: Optional[TensorData] = Field(default=None, description="Target tokens (alt format)")
    weights: Optional[TensorData] = Field(default=None, description="Token weights")
    mask: Optional[TensorData] = Field(default=None, description="Loss mask")


class ModelInputChunk(BaseModel):
    """Chunk in model input - text or image."""
    type: str = Field(default="encoded_text", description="Chunk type: encoded_text or image")
    # text fields (used when type == "encoded_text")
    tokens: Optional[List[int]] = Field(default=None, description="Token IDs")
    # image fields (used when type == "image")
    data: Optional[str] = Field(default=None, description="Base64-encoded image bytes")
    format: Optional[str] = Field(default=None, description="Image format: png/jpeg")
    expected_tokens: Optional[int] = Field(default=None, description="Expected token count")

    @model_validator(mode='after')
    def infer_image_type(self):
        # Defensive: older SDK versions use model_dump(exclude_unset=True) which
        # strips ImageChunk.type (default value never explicitly set). Infer
        # type from presence of image-only fields so server works with either
        # well-behaved or buggy clients.
        if self.data is not None and self.type == "encoded_text":
            self.type = "image"
        return self




class ModelInput(BaseModel):
    """Flexible model input format."""
    chunks: Optional[List[ModelInputChunk]] = Field(default=None, description="Chunked input")
    tokens: Optional[List[int]] = Field(default=None, description="Direct tokens")
    input_ids: Optional[List[int]] = Field(default=None, description="Input IDs")


class Datum(BaseModel):
    """The wire datum after boundary validation: what every backend converter
    reads. JSON and proto bodies both validate into this shape."""
    model_input: ModelInput = Field(..., description="Input tokens")
    loss_fn_inputs: Dict[str, TensorData] = Field(..., description="Loss function inputs")


class ForwardDatum(Datum):
    """Single forward data sample."""


class ForwardInput(BaseModel):
    """Batch of forward data."""
    data: List[ForwardDatum] = Field(..., description="Batch data")
    loss_fn: str = Field(default="cross_entropy", description="Loss function")
    loss_fn_config: Optional[LossFnConfig] = Field(default=None, description="Loss hyperparameters (see core.loss_registry)")


class ForwardBackwardDatum(Datum):
    """Single forward_backward data sample."""


class ForwardBackwardInput(BaseModel):
    """Batch of forward_backward data."""
    data: List[ForwardBackwardDatum] = Field(..., description="Batch data")
    loss_fn: str = Field(default="cross_entropy", description="Loss function name (see core.loss_registry)")
    loss_fn_config: Optional[LossFnConfig] = Field(default=None, description="Loss hyperparameters (see core.loss_registry)")


# ============= Sampling Models =============

class PromptChunk(BaseModel):
    """Chunk in prompt."""
    tokens: List[int] = Field(..., description="Token IDs")
    type: str = Field(default="encoded_text", description="Chunk type")


class PromptInput(BaseModel):
    """Flexible prompt format."""
    chunks: Optional[List[PromptChunk]] = Field(default=None, description="Chunked format")
    tokens: Optional[List[int]] = Field(default=None, description="Direct tokens")
    input_ids: Optional[List[int]] = Field(default=None, description="Input IDs")

    def get_tokens(self) -> List[int]:
        """Extract tokens from whichever format was provided."""
        if self.chunks:
            tokens = []
            for chunk in self.chunks:
                tokens.extend(chunk.tokens)
            return tokens
        elif self.tokens:
            return self.tokens
        elif self.input_ids:
            return self.input_ids
        raise ValueError("No tokens found in prompt")


# ============= Other Requests =============

class SaveWeightsForSamplerRequest(BaseModel):
    """Save weights for sampler request."""
    model_id: str = Field(..., description="Model ID")
    name: Optional[str] = Field(default=None, description="Checkpoint name (deprecated, use path)")
    path: Optional[str] = Field(default=None, description="Checkpoint path/name")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for ordering")
    sampling_session_seq_id: Optional[int] = Field(default=None, description="Sampling session sequence ID for ephemeral saves")


# ============= Updated Request Models (New Format) =============

class ForwardRequest(BaseModel):
    """Forward pass request (new format)."""
    model_id: str = Field(..., description="Model ID")
    forward_input: ForwardInput = Field(..., description="Forward pass data")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")


class ForwardBackwardRequest(BaseModel):
    """Forward-backward pass request (supports both old and new formats)."""
    model_id: str = Field(..., description="Model ID")
    forward_backward_input: Optional[ForwardBackwardInput] = Field(default=None, description="Training data (new format)")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")
    # Old format fields (for backward compatibility with HTTP tests)
    data: Optional[List[ForwardBackwardDatum]] = Field(default=None, description="Training data (old format)")
    loss_fn: Optional[str] = Field(default=None, description="Loss function (old format)")

    @model_validator(mode='before')
    @classmethod
    def wrap_old_format(cls, values):
        """Convert old format to new format for backward compatibility."""
        if isinstance(values, dict):
            # flat {data, loss_fn} form -> nested forward_backward_input
            if 'data' in values and 'forward_backward_input' not in values:
                values['forward_backward_input'] = {
                    'data': values.pop('data'),
                    'loss_fn': values.pop('loss_fn', 'cross_entropy')
                }
        return values

    @validator('forward_backward_input', always=True)
    @classmethod
    def ensure_forward_backward_input(cls, v):
        """Ensure forward_backward_input is set."""
        if v is None:
            raise ValueError("Either 'forward_backward_input' or 'data' must be provided")
        return v


class AdamParams(BaseModel):
    """Adam optimizer parameters."""
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    beta1: float = Field(default=0.9, description="Beta1 coefficient")
    beta2: float = Field(default=0.95, description="Beta2 coefficient")
    eps: float = Field(default=1e-12, description="Epsilon for numerical stability")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    grad_clip_norm: float = Field(default=0.0, description="Gradient clip norm (0.0 = no clipping)")


class OptimStepRequest(BaseModel):
    """Request to perform optimizer step (new format)."""
    model_id: str = Field(..., description="Model ID")
    adam_params: Optional[AdamParams] = Field(default=None, description="Adam optimizer parameters")
    optimizer_params: Optional[OptimParams] = Field(default=None, description="Non-Adam parameters (rejected: only adamw is served)")
    step_num: Optional[int] = Field(default=None, ge=0, description="Step number for logging")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")


class ASampleRequest(BaseModel):
    """Async sampling request (new format)."""
    num_samples: int = Field(default=1, ge=1, le=100, description="Number of samples")
    prompt: PromptInput = Field(..., description="Input prompt")
    sampling_params: Optional[SamplingParams] = Field(default=None, description="Sampling parameters")
    base_model: Optional[str] = Field(default=None, description="Base model")
    model_path: Optional[str] = Field(default=None, description="Model path")
    sampling_session_id: Optional[str] = Field(default=None, description="Sampling session ID (alternative to base_model/model_path)")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID within sampling session")
    prompt_logprobs: bool = Field(default=False, description="Return prompt logprobs")
    topk_prompt_logprobs: int = Field(default=0, ge=0, description="Top-k prompt logprobs to return")


class SampleRequest(BaseModel):
    """Sync sampling request (new format)."""
    prompts: List[List[int]] = Field(..., description="List of tokenized prompts")
    num_samples: int = Field(default=1, ge=1, le=100, description="Number of samples")
    sampling_params: Optional[SamplingParams] = Field(default=None, description="Sampling parameters")
    base_model: Optional[str] = Field(default=None, description="Base model")
    model_path: Optional[str] = Field(default=None, description="Model path")
    sampling_session_id: Optional[str] = Field(default=None, description="Sampling session ID (alternative to base_model/model_path)")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID within sampling session")


class CreateSamplingClientRequest(BaseModel):
    """Create sampling client request (new format)."""
    model_path: Optional[str] = Field(default=None, description="Tinker URI path")
    base_model: Optional[str] = Field(default=None, description="HuggingFace model path")
    sampling_params: Optional[SamplingParams] = Field(default=None, description="Default sampling parameters")


# ============= Session Models =============

class CreateSessionRequest(BaseModel):
    """Request to create a new client session."""
    tags: List[str] = Field(default_factory=list, description="Session tags")
    user_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata")
    sdk_version: str = Field(default="unknown", description="SDK version")
    type: str = Field(default="create_session", description="Request type")


class SessionHeartbeatRequest(BaseModel):
    """Request to send session heartbeat."""
    session_id: str = Field(..., description="Session ID to heartbeat")
    type: str = Field(default="session_heartbeat", description="Request type")


class CreateSamplingSessionRequest(BaseModel):
    """Request to create a sampling session."""
    session_id: str = Field(..., description="Parent session ID")
    sampling_session_seq_id: int = Field(..., description="Sequence ID within session")
    base_model: Optional[str] = Field(default=None, description="Base model for sampling")
    model_path: Optional[str] = Field(default=None, description="Tinker path to model weights")
    type: str = Field(default="create_sampling_session", description="Request type")


class JoinSamplingSessionRequest(BaseModel):
    """Request from a cloned SamplingClient for its own client id."""
    sampling_session_id: str = Field(..., description="Existing sampling session to join")
    type: str = Field(default="join_sampling_session", description="Request type")


class FinishReason(BaseModel):
    """Why the client finished its session."""
    type: Literal["success", "errored", "interrupted"] = Field(..., description="Terminal outcome")


class FinishSessionRequest(BaseModel):
    """Request to mark a session terminal (POST /api/v1/sessions/{id}/finish)."""
    reason: FinishReason = Field(..., description="Terminal outcome; first-wins")
    detail: Optional[str] = Field(default=None, description="Human-readable explanation")


# ============= Weights Info Models =============

class WeightsInfoRequest(BaseModel):
    """Request to get weights/checkpoint info from tinker path."""
    tinker_path: str = Field(..., description="Tinker URI path (e.g. tinker://model_xxx/weights/checkpoint_name)")