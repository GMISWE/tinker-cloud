"""
Response models for the training API.

This module defines Pydantic models for all API response payloads,
providing structured responses and documentation.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AsyncOperationResponse(BaseModel):
    """Response for async operations returning a request_id."""

    request_id: str = Field(..., description="Request ID for tracking")
    model_id: Optional[str] = Field(default=None, description="Associated model ID")


class FutureStatus(BaseModel):
    """Status of an async future."""

    request_id: str = Field(..., description="Request ID")
    status: str = Field(..., description="Status: pending, completed, failed")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result data if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class ModelInfo(BaseModel):
    """Model configuration information."""

    model_name: str = Field(..., description="Model identifier/path")
    max_context_length: int = Field(..., description="Maximum context length")
    supports_lora: bool = Field(default=True, description="Whether LoRA is supported")


class ServerCapabilities(BaseModel):
    """Server capabilities and supported models."""

    supported_models: List[ModelInfo] = Field(..., description="List of supported models")
    features: List[str] = Field(default_factory=list, description="Supported features")
    version: str = Field(default="3.0.0", description="API version")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Health status")
    version: str = Field(default="3.0.0", description="API version")
    timestamp: Optional[str] = Field(default=None, description="Current server time")
    ray_initialized: Optional[bool] = Field(default=None, description="Ray cluster initialization status")
    active_training_clients: Optional[int] = Field(default=None, description="Number of active training clients")
    model_ids: Optional[List[str]] = Field(default=None, description="List of active model IDs")
    futures_count: Optional[int] = Field(default=None, description="Number of pending async operations")


class TensorData(BaseModel):
    """Tensor data in serialized format."""

    data: List[float] = Field(..., description="Flattened tensor data")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field(default="float32", description="Data type")


class LossFnOutput(BaseModel):
    """Loss function output for a single sample."""

    loss: TensorData = Field(..., description="Loss value")
    logprobs: Optional[TensorData] = Field(default=None, description="Log probabilities")


class ForwardBackwardResult(BaseModel):
    """Result from forward-backward pass."""

    loss_fn_output_type: str = Field(..., description="Loss function type")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Per-sample outputs")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    logprobs: Optional[TensorData] = Field(default=None, description="Batch log probabilities")


class OptimStepResultOld(BaseModel):
    """DEPRECATED: Old result from optimizer step."""

    step_num: int = Field(..., description="Step number")
    grad_norm: float = Field(..., description="Gradient norm")
    success: bool = Field(default=True, description="Whether step succeeded")


class OptimStepResult(BaseModel):
    """Result from optimizer step (new format)."""
    success: bool = Field(default=True, description="Whether step succeeded")
    grad_norm: float = Field(..., description="Gradient norm")
    learning_rates: Dict[str, float] = Field(default_factory=dict, description="Learning rates per param group")
    status: str = Field(default="completed", description="Operation status")
    step_num: Optional[int] = Field(default=None, description="Step number")


class CheckpointInfo(BaseModel):
    """Checkpoint save information."""

    path: str = Field(..., description="Checkpoint path (tinker:// URI)")
    created_at: str = Field(..., description="Creation timestamp")
    model_id: str = Field(..., description="Associated model ID")
    type: str = Field(default="save_weights", description="Checkpoint type")


class SamplingSequence(BaseModel):
    """Generated sequence from sampling."""

    stop_reason: str = Field(..., description="Reason for stopping: stop or length")
    tokens: List[int] = Field(..., description="Generated token IDs")
    logprobs: List[float] = Field(..., description="Log probabilities")
    text: Optional[str] = Field(default=None, description="Decoded text")


class SampleResult(BaseModel):
    """Result from sampling operation."""

    sequences: List[SamplingSequence] = Field(..., description="Generated sequences")
    type: str = Field(default="sample", description="Operation type")
    prompt_logprobs: Optional[List[Optional[float]]] = Field(
        default=None,
        description="Prompt log probabilities (None for first token)"
    )


class SpecialTokens(BaseModel):
    """Tokenizer special tokens."""
    pad_token: Optional[str] = Field(default=None, description="Padding token")
    eos_token: Optional[str] = Field(default=None, description="End of sequence token")
    bos_token: Optional[str] = Field(default=None, description="Beginning of sequence token")
    unk_token: Optional[str] = Field(default=None, description="Unknown token")


class TokenizerInfoOld(BaseModel):
    """DEPRECATED: Old tokenizer information."""

    tokenizer_path: str = Field(..., description="Path to tokenizer")
    vocab_size: int = Field(..., description="Vocabulary size")
    special_tokens: Optional[Dict[str, int]] = Field(default=None, description="Special token IDs")


class TokenizerInfo(BaseModel):
    """Tokenizer information (new format)."""
    tokenizer_type: str = Field(default="HuggingFace", description="Tokenizer type")
    vocab_size: int = Field(..., description="Vocabulary size")
    model_max_length: int = Field(..., description="Maximum sequence length")
    pad_token_id: Optional[int] = Field(default=None, description="Padding token ID")
    eos_token_id: Optional[int] = Field(default=None, description="End of sequence token ID")
    bos_token_id: Optional[int] = Field(default=None, description="Beginning of sequence token ID")
    special_tokens: SpecialTokens = Field(..., description="Special tokens")
    hf_checkpoint: str = Field(..., description="HuggingFace checkpoint path")


class ModelInfoResponse(BaseModel):
    """Detailed model information."""

    model_id: str = Field(..., description="Model ID")
    base_model: str = Field(..., description="Base model path")
    lora_config: Optional[Dict[str, Any]] = Field(default=None, description="LoRA configuration")
    parallelism_config: Optional[Dict[str, Any]] = Field(default=None, description="Parallelism settings")
    created_at: str = Field(..., description="Creation timestamp")
    status: str = Field(..., description="Current status")


class TrainingRun(BaseModel):
    """Training run metadata."""

    training_run_id: str = Field(..., description="Training run ID")
    model_id: str = Field(..., description="Associated model ID")
    base_model: str = Field(..., description="Base model path")
    checkpoints: List[str] = Field(default_factory=list, description="List of checkpoint names")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class CleanupResult(BaseModel):
    """Result from cleanup operation."""

    futures_cleaned: int = Field(..., description="Number of futures cleaned")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Associated request ID")


# ============= Model Info Responses =============

class ModelData(BaseModel):
    """Model metadata."""
    arch: str = Field(..., description="Model architecture")
    model_name: str = Field(..., description="HuggingFace model path")


class GetInfoResponse(BaseModel):
    """Model info response."""
    type: str = Field(default="get_info", description="Response type")
    model_id: str = Field(..., description="Model ID")
    model_data: ModelData = Field(..., description="Model metadata")
    is_lora: bool = Field(..., description="Whether LoRA is enabled")
    lora_rank: Optional[int] = Field(default=None, description="LoRA rank if enabled")
    model_name: str = Field(..., description="Model name")


class DeleteModelResponse(BaseModel):
    """Delete model response."""
    model_id: str = Field(..., description="Deleted model ID")
    status: str = Field(default="deleted", description="Deletion status")
    message: str = Field(..., description="Deletion details")
    resources_freed: List[str] = Field(default_factory=list, description="List of freed resources")


class UnloadModelResponse(BaseModel):
    """Unload model response (Tinker SDK compatible).

    This is the Tinker-standard response for releasing model resources.
    Simpler than DeleteModelResponse to match Tinker SDK expectations.
    """
    model_id: str = Field(..., description="Unloaded model ID")
    type: Optional[str] = Field(default="unload_model", description="Response type")


# ============= Training Run Responses =============

class CheckpointMetadata(BaseModel):
    """Checkpoint metadata."""
    path: str = Field(..., description="Checkpoint path")
    created_at: str = Field(..., description="Creation timestamp")
    step_id: Optional[int] = Field(default=None, description="Training step ID")


class TrainingRunResponse(BaseModel):
    """Training run metadata response."""
    training_run_id: str = Field(..., description="Training run ID")
    base_model: str = Field(..., description="Base model path")
    model_owner: str = Field(default="kgateway-user", description="Model owner")
    is_lora: bool = Field(..., description="Whether LoRA is enabled")
    corrupted: bool = Field(default=False, description="Whether run is corrupted")
    lora_rank: int = Field(..., description="LoRA rank (0 if not LoRA)")
    last_request_time: str = Field(..., description="Last access time")
    last_checkpoint: Optional[CheckpointMetadata] = Field(default=None, description="Latest checkpoint")
    last_sampler_checkpoint: Optional[CheckpointMetadata] = Field(default=None, description="Latest sampler checkpoint")


# ============= Forward/Training Responses =============

class ForwardResult(BaseModel):
    """Forward pass result."""
    type: str = Field(default="forward", description="Operation type")
    loss_fn_output_type: str = Field(..., description="Loss function type")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Per-sample outputs")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")


# ============= Sampling Responses =============

class CreateSamplingClientResult(BaseModel):
    """Create sampling client result."""
    sampling_client_id: str = Field(..., description="Sampling client ID")
    model_path: str = Field(..., description="Model path")
    status: str = Field(default="ready", description="Client status")


class SaveWeightsForSamplerResult(BaseModel):
    """Save weights for sampler result."""
    type: str = Field(default="save_weights_for_sampler", description="Response type")
    path: Optional[str] = Field(default=None, description="Tinker URI path (for persistent saves)")
    sampling_session_id: Optional[str] = Field(default=None, description="Sampling session ID (for ephemeral saves)")
    checkpoint_path: Optional[str] = Field(default=None, description="Filesystem path")
    step_id: Optional[int] = Field(default=None, description="Checkpoint step ID")
    name: Optional[str] = Field(default=None, description="Checkpoint name")
    status: str = Field(default="completed", description="Operation status")


# ============= Other Responses =============

class DeprecatedEndpointError(BaseModel):
    """Deprecated endpoint error."""
    error: str = Field(..., description="Error message")
    reason: str = Field(..., description="Why deprecated")
    solution: Dict[str, Any] = Field(..., description="How to achieve same result")


# ============= Session Responses =============

class CreateSessionResponse(BaseModel):
    """Response from session creation."""
    type: str = Field(default="create_session", description="Response type")
    session_id: str = Field(..., description="Generated session ID")
    info_message: Optional[str] = Field(default=None, description="Info message")
    warning_message: Optional[str] = Field(default=None, description="Warning message")
    error_message: Optional[str] = Field(default=None, description="Error message")


class SessionHeartbeatResponse(BaseModel):
    """Response from session heartbeat."""
    type: str = Field(default="session_heartbeat", description="Response type")


class CreateSamplingSessionResponse(BaseModel):
    """Response from creating a sampling session."""
    type: str = Field(default="create_sampling_session", description="Response type")
    sampling_session_id: str = Field(..., description="Generated sampling session ID")


class GetSessionResponse(BaseModel):
    """Response for getting session details."""
    training_run_ids: List[str] = Field(..., description="List of model IDs associated with this session")
    sampler_ids: List[str] = Field(..., description="List of sampler IDs associated with this session")


class ListSessionsResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[str] = Field(..., description="List of session IDs")


class GetSamplerResponse(BaseModel):
    """Response for getting sampler details."""
    sampler_id: str = Field(..., description="The sampler ID (sampling_session_id)")
    base_model: str = Field(..., description="The base model name")
    model_path: Optional[str] = Field(default=None, description="Optional model path")


# ============= Weights Info Responses =============

class WeightsInfoResponse(BaseModel):
    """Minimal information for loading checkpoints (matches Tinker API)."""
    base_model: str = Field(..., description="Base model path")
    is_lora: bool = Field(..., description="Whether LoRA is enabled")
    lora_rank: Optional[int] = Field(default=None, description="LoRA rank if enabled")