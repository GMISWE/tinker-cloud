---
layout: default
title: Home
nav_order: 1
description: "Tinkercloud is a Tinker-compatible middleware for Miles distributed training"
permalink: /
---

# Tinkercloud: Tinker-Compatible Middleware for Miles

Tinkercloud is a FastAPI middleware that bridges the [Tinker API](https://tinker-docs.thinkingmachines.ai/) with [Miles](https://github.com/radixark/miles) for distributed GPU training. Miles orchestrates Megatron-LM for training and SGLang for inference.

## What is Tinkercloud?

Tinkercloud provides a **Tinker-compatible HTTP API** that:

- Accepts requests from `tinker_gmi` Python SDK or `tinker-cookbook`
- Translates them to Miles/Megatron distributed training operations
- Manages Ray actors for GPU-distributed training
- Handles SGLang inference for sampling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARCHITECTURE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  tinker-cookbook (Client)                                                    │
│  ├── TrainingClient.forward_backward()                                       │
│  ├── TrainingClient.optim_step()                                            │
│  └── SamplingClient.sample()                                                │
│                           │                                                  │
│                           │ HTTP (X-API-Key)                                 │
│                           ▼                                                  │
│  tinkercloud (Middleware) :8000                                              │
│  ├── Routers: models, training, sampling, checkpoints, sessions             │
│  ├── Services: ModelService, TrainingService, SamplingService               │
│  └── Core: SlimeArgumentBuilder, TinkerDataConverter, TaskManager           │
│                           │                                                  │
│                           │ Ray Remote Calls                                 │
│                           ▼                                                  │
│  Miles (Backend)                                                             │
│  ├── RayTrainGroup - Distributed training actors (TP/PP/CP/DP)              │
│  ├── RolloutManager - SGLang inference engines                              │
│  └── Uses Megatron-LM internally for distributed training                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Tinker API Compatibility** | Drop-in replacement for Tinker API endpoints |
| **LoRA Fine-tuning** | Low-rank adaptation with configurable rank |
| **Distributed Training** | TP/PP/CP/DP parallelism via Megatron-LM |
| **SGLang Sampling** | Fast inference with continuous batching |
| **Async Operations** | Non-blocking API with polling pattern |
| **Session Management** | Track client connections with heartbeat |
| **Checkpoint Support** | Save/load weights and optimizer state |

## Core API Operations

Tinkercloud's main functionality is contained in a few key endpoints:

- **`POST /api/v1/create_model`** - Create a training session with model + SGLang inference
- **`POST /api/v1/forward_backward`** - Compute gradients for training (RL or SL)
- **`POST /api/v1/optim_step`** - Update model parameters with Adam optimizer
- **`POST /api/v1/asample`** - Generate text with logprobs from SGLang
- **`POST /api/v1/save_weights_for_sampler`** - Save weights and sync to SGLang

## Supported Loss Functions

| Loss Function | Use Case | Description |
|---------------|----------|-------------|
| `cross_entropy` | Supervised Learning | Standard NLL loss |
| `importance_sampling` | RL (on-policy) | Policy gradient with importance weights |
| `ppo` | RL (clipped) | PPO-style clipped objective |

## Quick Example

```python
import tinker

# Create service client
client = tinker.ServiceClient(base_url="http://localhost:8000")

# Create training client (allocates GPUs, initializes model)
training_client = await client.create_lora_training_client_async(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    rank=32
)

# Training loop
for batch in dataset:
    # Compute gradients
    fwd_bwd = await training_client.forward_backward_async(batch, "cross_entropy")
    result = await fwd_bwd.result_async()

    # Update weights
    await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-6))

# Get sampling client with updated weights
sampling_client = await training_client.save_weights_and_get_sampling_client_async()

# Generate text
response = await sampling_client.sample_async(prompt, num_samples=1, params)
```

## Documentation Sections

### Getting Started
- [Installation](install.md) - Set up tinkercloud
- [Quickstart](quickstart.md) - Your first training run
- [Architecture](architecture.md) - System design overview

### API Reference
- [Endpoints](api/endpoints.md) - Complete HTTP API reference
- [Types](api/types.md) - Request/response schemas

### Core Concepts
- [Async & Futures](concepts/async-futures.md) - Polling pattern
- [Data Conversion](concepts/data-conversion.md) - Tinker ↔ Miles format
- [Sessions](concepts/sessions.md) - Client lifecycle management
- [GPU Memory](concepts/gpu-memory.md) - Offload/onload choreography

### Configuration
- [SlimeBuilder](config/slime-builder.md) - Megatron args from HF config
- [Parallelism](config/parallelism.md) - TP/PP/CP/DP settings
- [LoRA Config](config/lora-config.md) - Low-rank adaptation parameters

### Integration Guides
- [tinker-cookbook](guides/tinker-cookbook.md) - Using with cookbook recipes
- [Miles Integration](guides/miles-integration.md) - Backend architecture

## What's Next?

Tinkercloud works with any tinker-cookbook recipe:

- **RL Training** - See the [RL Training Guide](cookbook/rl-training.md) (e.g., RLVE, GRPO)
- **Supervised Learning** - See the [SL Guide](cookbook/supervised-training.md)
- **Troubleshooting** - See [Common Issues](appendix/troubleshooting.md)
