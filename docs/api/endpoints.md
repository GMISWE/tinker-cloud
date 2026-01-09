---
layout: default
title: API Endpoints
parent: API Reference
nav_order: 1
---

# API Reference: Endpoints

Complete HTTP API reference for tinkercloud.

## Authentication

All endpoints require API key authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: slime-dev-key" http://localhost:8000/api/v1/health
```

---

## Health & Capabilities

### GET /health

Simple health check.

**Response:**
```json
{"status": "healthy"}
```

### GET /api/v1/health

Detailed health check with Ray cluster status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ray_status": "connected",
  "active_models": 1,
  "active_sessions": 2
}
```

### GET /api/v1/get_server_capabilities

Query server capabilities and supported models.

**Response:**
```json
{
  "supported_models": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
  "max_batch_size": 256,
  "features": ["lora", "rlve", "sampling"]
}
```

---

## Model Lifecycle

### POST /api/v1/create_model

Create a training session with model initialization.

**Request:**
```json
{
  "session_id": "abc-123",
  "model_seq_id": 0,
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "lora_config": {
    "rank": 32,
    "train_mlp": true,
    "train_attn": true,
    "train_unembed": true
  },
  "max_batch_size": 256,
  "max_seq_len": 4096,
  "debug_train_only": false,
  "checkpoint_path": null,
  "rlve_config": null,
  "wandb_config": null
}
```

**Response:**
```json
{
  "request_id": "req-xyz-789",
  "message": "Model creation started"
}
```

**Polling Result:**
```json
{
  "model_id": "model-abc-123",
  "status": "ready",
  "router_address": "http://10.0.0.1:30000"
}
```

### POST /api/v1/delete_model

Delete a model and free GPU resources.

**Request:**
```json
{
  "model_id": "model-abc-123"
}
```

**Response:**
```json
{
  "status": "deleted",
  "model_id": "model-abc-123"
}
```

### POST /api/v1/get_info

Get model information.

**Request:**
```json
{
  "model_id": "model-abc-123"
}
```

**Response:**
```json
{
  "model_data": {
    "model_id": "model-abc-123",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "lora_rank": 32,
    "created_at": "2025-01-09T10:00:00Z"
  }
}
```

### GET /api/v1/get_tokenizer

Get tokenizer for a model.

**Query Parameters:**
- `model_id` (required): Model identifier

**Response:** Tokenizer JSON configuration

---

## Training Operations

### POST /api/v1/forward

Compute forward pass (logprobs only, no gradients).

**Request:**
```json
{
  "model_id": "model-abc-123",
  "data": [
    {
      "model_input": {
        "chunks": [{"tokens": [1, 2, 3, 4, 5]}]
      },
      "loss_fn_inputs": {
        "target_tokens": {"data": [3, 4, 5], "dtype": "int64"}
      }
    }
  ],
  "loss_fn": "cross_entropy"
}
```

**Response:**
```json
{
  "request_id": "req-fwd-001"
}
```

**Polling Result:**
```json
{
  "loss_fn_outputs": [
    {
      "logprobs": {"data": [-2.1, -1.8, -3.2], "dtype": "float32"}
    }
  ],
  "metrics": {
    "loss:sum": 7.1
  }
}
```

### POST /api/v1/forward_backward

Compute forward and backward pass (gradients).

**Request:**
```json
{
  "model_id": "model-abc-123",
  "data": [
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
  ],
  "loss_fn": "importance_sampling"
}
```

**Loss Functions:**
- `cross_entropy` - Supervised learning (requires `target_tokens`, `weights`)
- `importance_sampling` - RL on-policy (requires `target_tokens`, `logprobs`, `advantages`)
- `ppo` - RL with clipping (requires `target_tokens`, `logprobs`, `advantages`)

**Response:**
```json
{
  "request_id": "req-fb-001"
}
```

**Polling Result:**
```json
{
  "loss_fn_outputs": [
    {
      "logprobs": {"data": [-2.0, -1.9, -3.1, -2.4], "dtype": "float32"}
    }
  ],
  "metrics": {
    "loss:sum": 2.5,
    "ppo_kl": 0.01,
    "pg_clipfrac": 0.05
  }
}
```

### POST /api/v1/optim_step

Apply accumulated gradients with Adam optimizer.

**Request:**
```json
{
  "model_id": "model-abc-123",
  "adam_params": {
    "learning_rate": 1e-6,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,
    "weight_decay": 0.01
  }
}
```

**Response:**
```json
{
  "request_id": "req-opt-001"
}
```

**Polling Result:**
```json
{
  "learning_rates": [1e-6],
  "step_count": 1
}
```

---

## Sampling Operations

### POST /api/v1/asample

Async text generation with logprobs.

**Request:**
```json
{
  "model_id": "model-abc-123",
  "prompt": {
    "chunks": [{"tokens": [1, 2, 3]}]
  },
  "num_samples": 1,
  "sampling_params": {
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "stop": ["<|endoftext|>"]
  },
  "include_prompt_logprobs": false
}
```

**Response:**
```json
{
  "request_id": "req-sample-001"
}
```

**Polling Result:**
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

### POST /api/v1/sample

Synchronous text generation (blocking).

**Request:** Same as `/api/v1/asample`

**Response:** Same as polling result above (no request_id)

### POST /api/v1/create_sampling_client

Create a sampling client for a session.

**Request:**
```json
{
  "session_id": "abc-123",
  "sampling_session_seq_id": 0,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_path": null
}
```

**Response:**
```json
{
  "sampling_session_id": "sampler-xyz-001"
}
```

---

## Checkpoint Operations

### POST /api/v1/save_weights

Save model weights (full checkpoint).

**Request:**
```json
{
  "model_id": "model-abc-123",
  "name": "checkpoint-step-100"
}
```

**Response:**
```json
{
  "request_id": "req-save-001"
}
```

**Polling Result:**
```json
{
  "path": "/data/checkpoints/model-abc-123/checkpoint-step-100",
  "tinker_path": "tinker://model-abc-123/weights/checkpoint-step-100"
}
```

### POST /api/v1/save_weights_for_sampler

Save weights and sync to SGLang for inference.

**Request:**
```json
{
  "model_id": "model-abc-123",
  "name": "sampler-step-100"
}
```

**Response:**
```json
{
  "request_id": "req-save-sampler-001"
}
```

**Polling Result:**
```json
{
  "path": "/data/checkpoints/model-abc-123/sampler-step-100",
  "sampling_session_id": "sampler-xyz-002",
  "router_address": "http://10.0.0.1:30000"
}
```

### POST /api/v1/weights_info

Get information about saved weights.

**Request:**
```json
{
  "path": "tinker://model-abc-123/weights/checkpoint-step-100"
}
```

**Response:**
```json
{
  "exists": true,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "lora_rank": 32,
  "created_at": "2025-01-09T10:30:00Z"
}
```

---

## Session Management

### POST /api/v1/create_session

Create a client session.

**Request:**
```json
{
  "tags": ["rlve", "qwen"],
  "user_metadata": {"experiment": "run_001"},
  "sdk_version": "1.0.0"
}
```

**Response:**
```json
{
  "session_id": "session-abc-123"
}
```

### POST /api/v1/session_heartbeat

Keep session alive.

**Request:**
```json
{
  "session_id": "session-abc-123"
}
```

**Response:**
```json
{}
```

### POST /api/v1/create_sampling_session

Create a sampling session under a parent session.

**Request:**
```json
{
  "session_id": "session-abc-123",
  "sampling_session_seq_id": 0,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_path": null
}
```

**Response:**
```json
{
  "sampling_session_id": "session-abc-123_0_xyz"
}
```

### GET /api/v1/sessions

List all sessions.

**Query Parameters:**
- `limit` (optional, default 20): Max sessions to return
- `offset` (optional, default 0): Pagination offset

**Response:**
```json
{
  "sessions": ["session-abc-123", "session-def-456"]
}
```

### GET /api/v1/sessions/{session_id}

Get session details.

**Response:**
```json
{
  "training_run_ids": ["model-abc-123"],
  "sampler_ids": ["sampler-xyz-001", "sampler-xyz-002"]
}
```

### GET /api/v1/samplers/{sampler_id}

Get sampler details.

**Response:**
```json
{
  "sampler_id": "sampler-xyz-001",
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "model_path": "/data/checkpoints/model-abc-123/sampler-step-100"
}
```

---

## Async Operations

### POST /api/v1/retrieve_future/{request_id}

Poll for async operation result.

**Response (pending):**
```json
{
  "status": "pending",
  "result": null
}
```

**Response (completed):**
```json
{
  "status": "completed",
  "result": { ... }
}
```

**Response (failed):**
```json
{
  "status": "failed",
  "error": "Error message"
}
```

### POST /api/v1/cleanup_futures

Clean up old completed/failed futures.

**Request:**
```json
{
  "max_age_hours": 24
}
```

**Response:**
```json
{
  "cleaned_count": 15
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing the problem"
}
```

**Common HTTP Status Codes:**
- `400` - Bad request (invalid parameters)
- `401` - Unauthorized (missing/invalid API key)
- `404` - Not found (model/session doesn't exist)
- `500` - Internal server error
- `503` - Service unavailable (Ray cluster issues)

---

## Rate Limits

Currently no rate limits are enforced. For production deployments, consider adding:
- Request rate limiting per API key
- Concurrent model limit per session
- GPU allocation quotas
