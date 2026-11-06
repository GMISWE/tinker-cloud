"""
Request models for the training API.

This module defines Pydantic models for all API request payloads,
providing validation and documentation.
"""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, ValidationError, validator, model_validator

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


OPTIMIZER_FAMILIES = ("adamw", "dimuon")   # what the wire can name; a backend serves a subset (SUPPORTED_OPTIMIZERS)


class OptimizerConfig(BaseModel):
    """Optimizer identity, fixed at model creation (SDK `optimizer_config`): the
    family and, for dimuon, the version of TinkerCloud's stand-in for the
    unpublished algorithm. The router calls `check()`; the service then refuses
    a family the backend does not declare."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default="adamw", description="Optimizer family")
    version: Optional[StrictInt] = Field(default=None, description="Family version; dimuon serves 1 only")

    def check(self) -> None:
        """Raises ValueError (a 400) for a family the wire does not define or a version it does not serve."""
        if self.type not in OPTIMIZER_FAMILIES:
            raise ValueError(f"optimizer_config.type {self.type!r} is not an optimizer family; "
                             f"{' and '.join(repr(f) for f in OPTIMIZER_FAMILIES)} are")
        if self.type == "dimuon" and self.version != 1:
            raise ValueError(f"optimizer_config.version {self.version!r} is not served for dimuon; version 1 is")
        if self.type == "adamw" and self.version is not None:
            raise ValueError("optimizer_config.version is not a field of adamw")

    def record(self) -> Dict[str, Any]:
        """The form recorded with the model and its checkpoints and echoed by get_info / weights_info."""
        return {"type": self.type} if self.version is None else {"type": self.type, "version": self.version}


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
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Optimizer family, fixed for the model's life")
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
    """Request to load model weights, addressed one of two ways (the SDK's own split):
    `model_id` (+ `seq_id`): into a model just created, as its first request;
    `session_id` + `model_seq_id`: create-via-load, the load creates the model
    `{session_id}:train:{model_seq_id}` with the checkpoint's base model and LoRA config."""

    model_id: Optional[str] = Field(default=None, description="Model to load into (legacy addressing)")
    seq_id: Optional[int] = Field(default=None, description="Sequence ID for ordering")
    session_id: Optional[str] = Field(default=None, description="Create-via-load: the session that owns the new model")
    model_seq_id: Optional[int] = Field(default=None, description="Create-via-load: the new model's sequence id in its session")
    base_model: Optional[str] = Field(default=None, description="Create-via-load: must be the checkpoint's base model")
    user_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Create-via-load: accepted and not stored, as on create_model")
    weights_access_token: Optional[str] = Field(default=None, description="Another account's token; no accounts are served")
    path: str = Field(..., description="tinker://<run>/weights/<name> to load from")
    optimizer: bool = Field(default=False, description="Also restore optimizer state (the checkpoint must carry it)")
    optimizer_config: Optional[OptimizerConfig] = Field(default=None, description="Optimizer family of the restored run; must be the model's own")

    def creates(self) -> bool:
        """True for create-via-load addressing. Raises ValueError (a 400) when the
        request mixes the two forms, or names an account token."""
        legacy = self.model_id is not None or self.seq_id is not None
        session = self.session_id is not None or self.model_seq_id is not None
        if legacy == session:
            raise ValueError("load_weights takes either model_id (+ seq_id) or session_id + model_seq_id, not both or neither")
        if session and (self.session_id is None or self.model_seq_id is None):
            raise ValueError("create-via-load needs both session_id and model_seq_id")
        if legacy and self.model_id is None:
            raise ValueError("seq_id without model_id")
        if not session and (self.base_model is not None or self.user_metadata is not None):
            raise ValueError("base_model and user_metadata are create-via-load fields; the model already exists")
        if self.weights_access_token is not None:
            raise ValueError("weights_access_token names another account; this server serves one account")
        return session


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
    retrieved_request_ids: List[str] = Field(default_factory=list, description="Results the client has fetched; the server frees them")

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
    # the SDK (>= 0.38) always sends it; "snapshot" is its default (types/sampling_client_kind.py)
    kind: Literal["snapshot", "inflight"] = Field(default="snapshot", description="snapshot: fixed weights; inflight: the model's shared, moving weights")


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
    """One AdamW step's hyperparameters (SDK `AdamParams`, sent as `adam_params`)."""
    type: Literal["adamw"] = "adamw"   # the SDK's adam_params carry no type: the field names the family
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    beta1: float = Field(default=0.9, description="Beta1 coefficient")
    beta2: float = Field(default=0.95, description="Beta2 coefficient")
    eps: float = Field(default=1e-12, description="Epsilon for numerical stability")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    grad_clip_norm: float = Field(default=0.0, description="Gradient clip norm (0.0 = no clipping)")


class DimuonParams(BaseModel):
    """One Dimuon step's hyperparameters (SDK `DimuonParams`, sent as
    `optimizer_params`); the constraints are the SDK's."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["dimuon"] = "dimuon"   # the SDK's own default: one value, the family's name
    learning_rate: float = Field(..., ge=0, description="Step size")
    beta1: float = Field(default=0.9, ge=0, lt=1, description="Momentum EMA coefficient")
    grad_clip_norm: float = Field(default=0.0, description="Max global grad norm; a nonpositive value disables clipping")


OptimParams = Union[AdamParams, DimuonParams]


class OptimStepRequest(BaseModel):
    """Request to perform optimizer step (new format)."""
    model_id: str = Field(..., description="Model ID")
    adam_params: Optional[AdamParams] = Field(default=None, description="AdamW step parameters (SDK AdamParams)")
    optimizer_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Another family's step parameters, named by `type` (SDK DimuonParams)")
    step_num: Optional[int] = Field(default=None, ge=0, description="Step number for logging")
    seq_id: Optional[int] = Field(default=None, description="Per-model sequence number (idempotent retries)")

    def optim_params(self) -> OptimParams:
        """The one family's parameters this step carries. Raises ValueError (a
        400): both keys or neither, a family optimizer_params cannot name, or a
        field outside the family's constraints."""
        if (self.adam_params is None) == (self.optimizer_params is None):
            raise ValueError("optim_step takes exactly one of adam_params (adamw) and optimizer_params (dimuon)")
        if self.adam_params is not None:
            return self.adam_params
        assert self.optimizer_params is not None
        if "type" not in self.optimizer_params:
            raise ValueError("optimizer_params names no type")
        family = self.optimizer_params["type"]
        if family != "dimuon":
            raise ValueError(f"optimizer_params.type {family!r} is not a family optimizer_params carries; "
                             "'dimuon' is (adamw goes as adam_params)")
        try:
            return DimuonParams.model_validate(self.optimizer_params)
        except ValidationError as e:
            fields = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
            raise ValueError(f"optimizer_params (dimuon): {fields}") from e


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
    topk_sample_logprobs: int = Field(default=0, ge=0, description="Top-k logprobs per sampled token")
    # Score only the last N prompt tokens; earlier prompt_logprobs / top-k rows
    # are None. Requires prompt_logprobs and 1 <= N <= len(prompt) - 1 (router).
    prompt_logprobs_last_n: Optional[int] = Field(default=None, description="Score only the last N prompt tokens")
    # SDK 0.33 fields this server does not serve; accepted so the router can
    # refuse them by name (400) instead of pydantic silently dropping them.
    target_prompt_logprobs: Optional[TensorData] = Field(default=None, description="Unsupported: token ids to score")
    prompt_alt_tokens_k: int = Field(default=0, ge=0, description="Unsupported: alternative prompt draws")


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
    sampling_session_id: Optional[str] = Field(default=None, description="A sampler the gateway holds; its model arrives in the headers")
    model_path: Optional[str] = Field(default=None, description="Tinker URI path")
    base_model: Optional[str] = Field(default=None, description="HuggingFace model path")
    sampling_params: Optional[SamplingParams] = Field(default=None, description="Default sampling parameters")


# ============= Session Models =============

class CreateSessionRequest(BaseModel):
    """Request to create a new client session."""
    tags: List[str] = Field(default_factory=list, description="Session tags")
    user_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Custom metadata")
    sdk_version: str = Field(default="unknown", description="SDK version")
    display_name: Optional[str] = Field(default=None, description="Non-unique name for display and search")
    type: str = Field(default="create_session", description="Request type")


# Session search (SDK >= 0.38 types/session_search_request.py): the server mirrors the SDK's
# defaults and bounds. One implicit owner and project: this server has no accounts.
LOCAL_OWNER = "urn:tinkercloud:owner:local"
DEFAULT_PROJECT = "default"


class SessionStatusQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["active", "cancelled", "failed", "client_finished"]
    reasons: Optional[List[Literal["success", "errored", "interrupted"]]] = Field(default=None, min_length=1)


class SearchOwner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["own", "accessible", "user"] = "own"   # the SDK's default owner (types/search_owner.py)
    owner_urns: Optional[List[str]] = Field(default=None, min_length=1, max_length=10)

    @model_validator(mode="after")
    def _urns_with_user(self) -> "SearchOwner":
        if (self.type == "user") != (self.owner_urns is not None):
            raise ValueError("owner_urns is given exactly when owner type is 'user'")
        return self


class SessionSearchCursor(BaseModel):
    last_created_at: AwareDatetime
    last_session_id: str


class SessionSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: Optional[str] = None
    project_ids: Optional[List[str]] = None
    user_metadata: Dict[str, str] = Field(default_factory=dict)
    display_name: Optional[str] = Field(default=None, min_length=1)
    owner: SearchOwner = Field(default_factory=SearchOwner)
    statuses: Optional[List[SessionStatusQuery]] = Field(default=None, min_length=1)
    limit: int = Field(default=50, ge=1, le=200)
    cursor: Optional[SessionSearchCursor] = None


class SessionHeartbeatRequest(BaseModel):
    """Request to send session heartbeat."""
    session_id: str = Field(..., description="Session ID to heartbeat")
    type: str = Field(default="session_heartbeat", description="Request type")


class ResolveSamplerTargetRequest(BaseModel):
    """The gateway asks the engine which model a sampler's path names (create_sampling_session)."""
    model_path: Optional[str] = Field(default=None, description="Tinker path to model weights")
    base_model: Optional[str] = Field(default=None, description="Base model for sampling")


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