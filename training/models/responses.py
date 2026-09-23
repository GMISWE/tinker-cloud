"""
Response models for the training API.

This module defines Pydantic models for all API response payloads,
providing structured responses and documentation.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AsyncOperationResponse(BaseModel):
    """Response for async operations returning a request_id."""

    request_id: str = Field(..., description="Request ID for tracking")
    model_id: Optional[str] = Field(default=None, description="Associated model ID")
    # Sampling submissions only: one id per sequence the request will return,
    # fixed at submission. SDK >= 0.25 stamps them onto SampledSequence.sequence_id
    # and refuses a sample result whose submission carried none.
    sample_sequence_ids: Optional[List[str]] = Field(default=None, description="Per-sequence ids (sampling only)")


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


class ClientConfigResponse(BaseModel):
    """Feature flags the SDK (>= 0.25) fetches once at client construction
    (`POST /api/v1/client/config`). Fields the SDK does not know are ignored
    client-side; fields omitted here take the SDK's defaults. Every value
    below is the setting this server actually honours."""

    pjwt_auth_enabled: bool = Field(default=False, description="False: API keys only, no JWT exchange")
    proto_compress_fwdbwd: bool = Field(default=False, description="True only when the server can zstd-decompress bodies")
    create_model_via_load_weights: bool = Field(default=False, description="False: create_model then load_weights")
    sample_use_retrieve_futures: bool = Field(default=True, description="True: sample completions are delivered by per-session /retrieve_futures poller")
    use_pyqwest_transport: bool = Field(default=False, description="False: the SDK uses its httpx default transport")

class ClientDynamicConfigResponse(BaseModel):
    """Flags the SDK refetches while running (`POST /api/v1/client/dynamic_config`)"""

    refresh_interval_sec: int = Field(default=300, description="SDK refetch cadence")
    sample_cancel_enabled: bool = Field(default=True, description="True: an abandoned sample future sends cancel_future")
    sample_cancel_max_batch_size: int = Field(default=64, description="Cancels the SDK dispatches in parallel per dtrain")

class SessionFutureCompletion(BaseModel):
    """One entry of the /retrieve_futures response (SDK FutureFinished | FutureFailed)."""

    state: Literal["finished", "failed"]
    request_id: str
    response_payload_uncompressed_size: Optional[int] = Field(
        default=None, description="Proto payload bytes of a finished sample; absent on failed"
    )

class SessionFuturesPollResponse(BaseModel):
    """Per session completion poll result( SDK type FuturesRetrieveResponse)."""
    
    completions: List[SessionFutureCompletion]
    cursor: int=Field(..., description="Pass back as prev_cursor on the next poll")


class TensorData(BaseModel):
    """Tensor data in serialized format."""

    data: List[float] = Field(..., description="Flattened tensor data")
    shape: List[int] = Field(..., description="Tensor shape")
    dtype: str = Field(default="float32", description="Data type")


# Per-sample loss function output: named tensors ("logprobs", "logits", ...),
# exactly the SDK's LossFnOutput = Dict[str, TensorData].
LossFnOutput = Dict[str, TensorData]


class _Result(BaseModel):
    """Base for async-operation results: required fields are what the SDK
    requires; backend-specific extras (deferred, weight_version, ...) pass through."""

    model_config = ConfigDict(extra="allow")


class ForwardBackwardResult(_Result):
    """Result from forward-backward pass (SDK ForwardBackwardOutput)."""

    loss_fn_output_type: str = Field(..., description="Loss function type")
    loss_fn_outputs: List[LossFnOutput] = Field(..., description="Per-sample outputs")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")


class ForwardResult(ForwardBackwardResult):
    """Forward pass result (same shape, type='forward')."""

    type: str = Field(default="forward", description="Operation type")


class OptimStepResult(_Result):
    """Result from optimizer step (SDK OptimStepResponse)."""

    metrics: Dict[str, float] = Field(default_factory=dict, description="Step metrics (grad_norm mirrored here)")
    loss_fn_outputs: Optional[List[LossFnOutput]] = Field(
        default=None, description="Per-sample outputs when the backend defers them to the step")
    grad_norm: Optional[float] = Field(default=None, description="Gradient norm")
    success: bool = Field(default=True, description="Whether step succeeded")


class CreateModelResult(_Result):
    """Result from create_model (SDK CreateModelResponse)."""

    model_id: str = Field(..., description="Model ID")
    type: Literal["create_model"] = "create_model"


class SaveWeightsResult(_Result):
    """Result from save_weights (SDK SaveWeightsResponse)."""

    path: str = Field(..., description="tinker:// URI of the checkpoint")
    type: Literal["save_weights"] = "save_weights"


class LoadWeightsResult(_Result):
    """Result from load_weights (SDK LoadWeightsResponse)."""

    path: Optional[str] = Field(default=None, description="Path that was loaded")
    type: Literal["load_weights"] = "load_weights"


class CheckpointInfo(BaseModel):
    """Checkpoint save information."""

    path: str = Field(..., description="Checkpoint path (tinker:// URI)")
    created_at: str = Field(..., description="Creation timestamp")
    model_id: str = Field(..., description="Associated model ID")
    type: str = Field(default="save_weights", description="Checkpoint type")


class SamplingSequence(_Result):
    """Generated sequence from sampling (SDK SampledSequence)."""

    stop_reason: str = Field(..., description="Reason for stopping: stop or length")
    tokens: List[int] = Field(..., description="Generated token IDs")
    logprobs: Optional[List[float]] = Field(default=None, description="Log probabilities")
    text: Optional[str] = Field(default=None, description="Decoded text")


class SampleResult(_Result):
    """Result from sampling operation (SDK SampleResponse)."""

    sequences: List[SamplingSequence] = Field(..., description="Generated sequences")
    type: str = Field(default="sample", description="Operation type")
    prompt_logprobs: Optional[List[Optional[float]]] = Field(
        default=None,
        description="Prompt log probabilities (None for first token)"
    )
    # ver(S) certificate (A4, specs/012): version actually served vs latest.
    # None on backends that don't stamp versions.
    weight_version: Optional[int] = Field(
        default=None, description="Weight version the sampler served (ver(S))"
    )
    latest_weight_version: Optional[int] = Field(
        default=None, description="Latest trained weight version at serve time"
    )


class SpecialTokens(BaseModel):
    """Tokenizer special tokens."""
    pad_token: Optional[str] = Field(default=None, description="Padding token")
    eos_token: Optional[str] = Field(default=None, description="End of sequence token")
    bos_token: Optional[str] = Field(default=None, description="Beginning of sequence token")
    unk_token: Optional[str] = Field(default=None, description="Unknown token")


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


class UnloadModelResponse(_Result):
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


# ============= Sampling Responses =============

class CreateSamplingClientResult(_Result):
    """Create sampling client result."""
    sampling_client_id: str = Field(..., description="Sampling client ID")
    model_path: str = Field(..., description="Model path")
    status: str = Field(default="ready", description="Client status")


class SaveWeightsForSamplerResult(_Result):
    """Save weights for sampler result."""
    type: str = Field(default="save_weights_for_sampler", description="Response type")
    path: Optional[str] = Field(default=None, description="Tinker URI path (for persistent saves)")
    sampling_session_id: Optional[str] = Field(default=None, description="Sampling session ID (for ephemeral saves)")
    uri: Optional[str] = Field(default=None, description="Identity of the saved sampler weights, ephemeral saves included")
    step: Optional[int] = Field(default=None, description="The store's per-model save counter (None for ephemeral)")
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


# ============= Result validation at the futures boundary =============

# operation name (TaskManager) -> model a completed result must satisfy
RESULT_MODELS: Dict[str, type] = {
    "create_model": CreateModelResult,
    "forward": ForwardResult,
    "forward_backward": ForwardBackwardResult,
    "optim_step": OptimStepResult,
    "save_weights": SaveWeightsResult,
    "save_weights_for_sampler": SaveWeightsForSamplerResult,
    "load_weights": LoadWeightsResult,
    "sample": SampleResult,
    "asample": SampleResult,
    "create_sampling_client": CreateSamplingClientResult,
    "unload_model": UnloadModelResponse,
}


class ResultShapeError(ValueError):
    """A backend/service returned a result the operation's response model rejects."""


def validate_result(operation: str, result: Any) -> Dict[str, Any]:
    """Check `result` against the operation's response model and return it as a
    plain dict. Only fields present in the input (plus coercions) are emitted,
    so the wire shape is unchanged for a well-formed result. Operations without
    a registered model pass through untouched."""
    model = RESULT_MODELS.get(operation)
    if model is None:
        return result
    try:
        parsed = model.model_validate(result)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()[:5])
        raise ResultShapeError(f"{operation} result does not match {model.__name__}: {errs}") from None
    return parsed.model_dump(exclude_unset=True)
