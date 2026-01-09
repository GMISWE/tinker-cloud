---
layout: default
title: Types
parent: API Reference
nav_order: 2
---

# API Reference: Types

Request and response schemas for tinkercloud API.

---

## Core Data Types

### ModelInput

Flexible token input format.

```python
class ModelInput:
    chunks: List[Chunk]  # Preferred format
    # OR
    tokens: List[int]    # Direct token list
    # OR
    input_ids: List[int] # HuggingFace format
```

**JSON Examples:**

```json
// Chunks format (preferred)
{
  "chunks": [{"tokens": [1, 2, 3, 4, 5]}]
}

// Direct tokens
{
  "tokens": [1, 2, 3, 4, 5]
}

// HuggingFace format
{
  "input_ids": [1, 2, 3, 4, 5]
}
```

### TensorData

Serialized tensor format for loss function inputs/outputs.

```python
class TensorData:
    data: List[float | int]  # Flattened tensor data
    shape: Optional[List[int]] = None  # Shape (defaults to 1D)
    dtype: str = "float32"  # numpy dtype string
```

**JSON Example:**

```json
{
  "data": [0.5, -0.3, 0.8, 0.2],
  "shape": [4],
  "dtype": "float32"
}
```

**Supported dtypes:** `float32`, `float64`, `int32`, `int64`, `bool`

### Datum

Single training sample with model input and loss function inputs.

```python
class Datum:
    model_input: ModelInput
    loss_fn_inputs: Dict[str, TensorData]
```

**JSON Example (Supervised Learning):**

```json
{
  "model_input": {
    "chunks": [{"tokens": [1, 2, 3, 4, 5]}]
  },
  "loss_fn_inputs": {
    "target_tokens": {"data": [3, 4, 5], "dtype": "int64"},
    "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32"}
  }
}
```

**JSON Example (RL with Importance Sampling):**

```json
{
  "model_input": {
    "chunks": [{"tokens": [1, 2, 3, 4, 5, 6, 7, 8]}]
  },
  "loss_fn_inputs": {
    "target_tokens": {"data": [5, 6, 7, 8], "dtype": "int64"},
    "logprobs": {"data": [-2.1, -1.8, -3.2, -2.5], "dtype": "float32"},
    "advantages": {"data": [0.5, -0.3, 0.8, 0.2], "dtype": "float32"}
  }
}
```

---

## Configuration Types

### LoraConfig

Low-rank adaptation configuration.

```python
class LoraConfig:
    rank: int = 32           # LoRA rank (8, 16, 32, 64, 128)
    train_mlp: bool = True   # Train MLP layers
    train_attn: bool = True  # Train attention layers
    train_unembed: bool = True  # Train unembedding layer
```

**JSON Example:**

```json
{
  "rank": 32,
  "train_mlp": true,
  "train_attn": true,
  "train_unembed": true
}
```

### AdamParams

Adam optimizer parameters.

```python
class AdamParams:
    learning_rate: float    # Required
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
```

**JSON Example:**

```json
{
  "learning_rate": 1e-6,
  "beta1": 0.9,
  "beta2": 0.95,
  "eps": 1e-8,
  "weight_decay": 0.01
}
```

### SamplingParams

Text generation parameters.

```python
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
```

**JSON Example:**

```json
{
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": ["<|endoftext|>", "\n\n"]
}
```

### RLVEConfig

Server-side RLVE (RL with Verifiable Environments) configuration.

```python
class RLVEConfig:
    enabled: bool = False
    environment_list: List[str] = []
    rollout_max_response_len: int = 4096
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    num_rollout: int = 6
    over_sampling_batch_size: int = 96
    balance_data: bool = True
    partial_rollout: bool = True
    use_dynamic_sampling_filter: bool = True
```

**JSON Example:**

```json
{
  "enabled": true,
  "environment_list": ["Division", "Multiplication", "GCDOne_Counting"],
  "n_samples_per_prompt": 8,
  "rollout_max_response_len": 4096
}
```

### WandbConfig

Weights & Biases logging configuration.

```python
class WandbConfig:
    project: str
    name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = []
```

**JSON Example:**

```json
{
  "project": "rlve-training",
  "name": "qwen-7b-run-001",
  "tags": ["rlve", "qwen"]
}
```

---

## Request Types

### CreateModelRequest

```python
class CreateModelRequest:
    session_id: str
    model_seq_id: int
    model_name: str
    lora_config: Optional[LoraConfig] = None
    max_batch_size: int = 256
    max_seq_len: int = 4096
    debug_train_only: bool = False
    checkpoint_path: Optional[str] = None
    rlve_config: Optional[RLVEConfig] = None
    wandb_config: Optional[WandbConfig] = None
```

### ForwardBackwardRequest

```python
class ForwardBackwardRequest:
    model_id: str
    data: List[Datum]
    loss_fn: str  # "cross_entropy", "importance_sampling", "ppo"
    loss_fn_config: Optional[Dict[str, float]] = None
```

### OptimStepRequest

```python
class OptimStepRequest:
    model_id: str
    adam_params: AdamParams
```

### ASampleRequest

```python
class ASampleRequest:
    model_id: str
    prompt: ModelInput
    num_samples: int = 1
    sampling_params: Optional[SamplingParams] = None
    include_prompt_logprobs: bool = False
```

### SaveWeightsRequest

```python
class SaveWeightsRequest:
    model_id: str
    name: str
```

### CreateSessionRequest

```python
class CreateSessionRequest:
    tags: List[str] = []
    user_metadata: Optional[Dict[str, Any]] = None
    sdk_version: str = "unknown"
```

### SessionHeartbeatRequest

```python
class SessionHeartbeatRequest:
    session_id: str
```

---

## Response Types

### AsyncOperationResponse

Returned for async operations.

```python
class AsyncOperationResponse:
    request_id: str
    message: Optional[str] = None
```

**JSON Example:**

```json
{
  "request_id": "req-abc-123",
  "message": "Operation started"
}
```

### FutureStatus

Status when polling for results.

```python
class FutureStatus:
    status: str  # "pending", "completed", "failed"
    result: Optional[Any] = None
    error: Optional[str] = None
```

### ForwardBackwardResult

Result of forward/forward_backward operations.

```python
class ForwardBackwardResult:
    loss_fn_outputs: List[Dict[str, TensorData]]
    metrics: Dict[str, float]
```

**JSON Example:**

```json
{
  "loss_fn_outputs": [
    {
      "logprobs": {"data": [-2.0, -1.9, -3.1], "dtype": "float32"}
    }
  ],
  "metrics": {
    "loss:sum": 2.5,
    "ppo_kl": 0.01,
    "pg_clipfrac": 0.05
  }
}
```

### OptimStepResult

Result of optimizer step.

```python
class OptimStepResult:
    learning_rates: List[float]
    step_count: int
```

### SampleResult

Result of sampling operations.

```python
class SampleResult:
    sequences: List[SamplingSequence]
    prompt_logprobs: Optional[List[float]] = None

class SamplingSequence:
    tokens: List[int]
    logprobs: List[float]
    finish_reason: str  # "stop", "length", "eos"
```

**JSON Example:**

```json
{
  "sequences": [
    {
      "tokens": [4, 5, 6, 7, 8],
      "logprobs": [-1.2, -0.8, -2.1, -1.5, -0.9],
      "finish_reason": "stop"
    }
  ],
  "prompt_logprobs": null
}
```

### SaveWeightsResult

Result of checkpoint save.

```python
class SaveWeightsResult:
    path: str
    tinker_path: Optional[str] = None
    sampling_session_id: Optional[str] = None
    router_address: Optional[str] = None
```

### CreateSessionResponse

```python
class CreateSessionResponse:
    session_id: str
    info_message: Optional[str] = None
    warning_message: Optional[str] = None
    error_message: Optional[str] = None
```

### GetSessionResponse

```python
class GetSessionResponse:
    training_run_ids: List[str]
    sampler_ids: List[str]
```

### ModelInfoResponse

```python
class ModelInfoResponse:
    model_data: ModelData

class ModelData:
    model_id: str
    model_name: str
    lora_rank: int
    created_at: str
```

### HealthResponse

```python
class HealthResponse:
    status: str
    version: str
    ray_status: str
    active_models: int
    active_sessions: int
```

---

## Loss Function Input Requirements

| Loss Function | Required Inputs |
|---------------|-----------------|
| `cross_entropy` | `target_tokens`, `weights` |
| `importance_sampling` | `target_tokens`, `logprobs`, `advantages` |
| `ppo` | `target_tokens`, `logprobs`, `advantages` |

### cross_entropy inputs

```json
{
  "target_tokens": {"data": [3, 4, 5], "dtype": "int64"},
  "weights": {"data": [1.0, 1.0, 1.0], "dtype": "float32"}
}
```

### importance_sampling / ppo inputs

```json
{
  "target_tokens": {"data": [5, 6, 7, 8], "dtype": "int64"},
  "logprobs": {"data": [-2.1, -1.8, -3.2, -2.5], "dtype": "float32"},
  "advantages": {"data": [0.5, -0.3, 0.8, 0.2], "dtype": "float32"}
}
```

---

## Python Type Hints

For tinker_gmi client users:

```python
from tinker import types

# Core types
types.ModelInput
types.TensorData
types.Datum

# Configs
types.LoraConfig
types.AdamParams
types.SamplingParams
types.RLVEConfig

# Responses
types.ForwardBackwardOutput
types.OptimStepResponse
types.SampleResponse
types.SaveWeightsResponse
```
