---
layout: default
title: Architecture
nav_order: 4
---

# Architecture Overview

This document explains the high-level architecture of tinkercloud and how it integrates with [Miles](https://github.com/radixark/miles) for distributed training. Miles is the orchestration layer that uses Megatron-LM for training and SGLang for inference.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  tinker_gmi (Python SDK)                                                     │
│  ├── ServiceClient       - Entry point, create training/sampling clients    │
│  ├── TrainingClient      - forward(), forward_backward(), optim_step()      │
│  └── SamplingClient      - sample(), compute_logprobs()                     │
│                                                                              │
│  tinker-cookbook                                                             │
│  ├── rl/train.py         - RL training orchestration                        │
│  ├── supervised/train.py - SL training orchestration                        │
│  └── recipes/            - Ready-to-use training recipes                    │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               │ HTTP (X-API-Key: slime-dev-key)
                               │ POST /api/v1/forward_backward
                               │ POST /api/v1/optim_step
                               │ POST /api/v1/asample
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIDDLEWARE (tinkercloud)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FastAPI Server (:8000)                                                      │
│  ├── Routers                                                                 │
│  │   ├── models.py      - create_model, delete_model, get_info              │
│  │   ├── training.py    - forward, forward_backward, optim_step             │
│  │   ├── sampling.py    - asample, sample, create_sampling_client           │
│  │   ├── checkpoints.py - save_weights, save_weights_for_sampler            │
│  │   ├── session.py     - create_session, heartbeat, list_sessions          │
│  │   └── futures.py     - retrieve_future, cleanup_futures                  │
│  │                                                                           │
│  ├── Services                                                                │
│  │   ├── ModelService      - Ray actor lifecycle, placement groups          │
│  │   ├── TrainingService   - GPU offload/onload, Miles calls                │
│  │   ├── SamplingService   - SGLang HTTP client                             │
│  │   ├── CheckpointService - File I/O, tinker:// URI parsing                │
│  │   └── SessionService    - Heartbeat tracking, cleanup                    │
│  │                                                                           │
│  ├── Core                                                                    │
│  │   ├── SlimeArgumentBuilder  - HF config → Megatron args                  │
│  │   ├── TinkerDataConverter   - Tinker JSON ↔ Miles tensors                │
│  │   └── TaskManager           - Async futures with polling                 │
│  │                                                                           │
│  └── Storage                                                                 │
│      ├── FuturesStorage     - SQLite async operation tracking               │
│      ├── MetadataStorage    - JSON model/checkpoint metadata                │
│      └── SessionStorage     - SQLite session persistence                    │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               │ Ray Remote Calls
                               │ train_group.forward_backward_only.remote()
                               │ rollout_manager.offload.remote()
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND (Miles)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Miles orchestrates:                                                         │
│  ├── Megatron-LM for distributed training                                   │
│  └── SGLang for inference                                                   │
│                                                                              │
│  RayTrainGroup (Training)                                                    │
│  ├── forward_only() - Compute logprobs without gradients                    │
│  ├── forward_backward_only() - Compute gradients                            │
│  ├── apply_optimizer_step() - Update weights with Adam                      │
│  └── offload() / onload() - GPU memory management                           │
│                                                                              │
│  RolloutManager (Inference)                                                  │
│  ├── SGLang inference engines (one per GPU)                                 │
│  ├── offload() / onload() - Free GPU memory for training                    │
│  ├── update_weights() - Sync weights from training                          │
│  └── get_router_address() - HTTP endpoint for sampling                      │
│                                                                              │
│  Parallelism (via Megatron-LM)                                               │
│  ├── Tensor Parallelism (TP) - Split weights across GPUs                    │
│  ├── Pipeline Parallelism (PP) - Split layers across GPUs                   │
│  ├── Context Parallelism (CP) - Split sequence across GPUs                  │
│  └── Data Parallelism (DP) - Replicate across GPU groups                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Request Flow

### Training Request (forward_backward)

```
1. Client sends POST /api/v1/forward_backward
   │
   ▼
2. training.py router validates request
   │
   ▼
3. TaskManager creates async task + future
   │
   ▼
4. TrainingService.forward_backward() called
   │
   ├── 4a. SGLang offload (free GPU memory)
   │        rollout_manager.offload.remote()
   │
   ├── 4b. Convert Tinker → Miles format
   │        TinkerDataConverter.forward_backward_to_rollout()
   │
   ├── 4c. Call Miles training
   │        train_group.forward_backward_only.remote(Box(data))
   │
   └── 4d. Convert Miles → Tinker format
            TinkerDataConverter.rollout_to_forward_backward_result()
   │
   ▼
5. FuturesStorage updated with result
   │
   ▼
6. Client polls POST /api/v1/retrieve_future/{id}
   │
   ▼
7. Result returned to client
```

### Model Creation Flow

```
1. POST /api/v1/create_model
   │
   ▼
2. ModelService.create_model()
   │
   ├── 2a. SlimeArgumentBuilder.build_args()
   │        - Load HF config
   │        - Auto-detect parallelism (TP/PP/CP)
   │        - Build Miles/Megatron CLI args
   │
   ├── 2b. Create Ray placement group
   │        - Request GPU bundles
   │        - Wait for allocation
   │
   ├── 2c. Create RayTrainGroup (via Miles)
   │        - Initialize training actors
   │        - Load model weights
   │
   └── 2d. Create RolloutManager (if not debug_train_only)
            - Start SGLang engines
            - Initialize router
   │
   ▼
3. Model registered in MetadataStorage
   │
   ▼
4. Return model_id to client
```

## GPU Memory Management

Tinkercloud uses a colocated mode where training (via Miles/Megatron) and inference (SGLang) share GPUs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPU MEMORY CHOREOGRAPHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE: SAMPLING (SGLang Active)                                             │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐  │
│  │ GPU 0           │ GPU 1           │ GPU 2           │ GPU 3           │  │
│  │ SGLang (100%)   │ SGLang (100%)   │ SGLang (100%)   │ SGLang (100%)   │  │
│  │ KV Cache        │ KV Cache        │ KV Cache        │ KV Cache        │  │
│  │ CUDA Graphs     │ CUDA Graphs     │ CUDA Graphs     │ CUDA Graphs     │  │
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘  │
│                               │                                              │
│                               │ rollout_manager.offload()                    │
│                               ▼                                              │
│  PHASE: TRAINING (Megatron Active)                                           │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐  │
│  │ GPU 0           │ GPU 1           │ GPU 2           │ GPU 3           │  │
│  │ Megatron (80%)  │ Megatron (80%)  │ Megatron (80%)  │ Megatron (80%)  │  │
│  │ Model Weights   │ Model Weights   │ Model Weights   │ Model Weights   │  │
│  │ Activations     │ Activations     │ Activations     │ Activations     │  │
│  │ Gradients       │ Gradients       │ Gradients       │ Gradients       │  │
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘  │
│                               │                                              │
│                               │ train_group.offload()                        │
│                               │ rollout_manager.onload(WEIGHTS)              │
│                               │ rollout_manager.update_weights()             │
│                               │ rollout_manager.onload(CUDA_GRAPH)           │
│                               │ rollout_manager.onload(KV_CACHE)             │
│                               ▼                                              │
│  PHASE: SAMPLING (SGLang Active, Updated Weights)                            │
│  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐  │
│  │ GPU 0           │ GPU 1           │ GPU 2           │ GPU 3           │  │
│  │ SGLang (100%)   │ SGLang (100%)   │ SGLang (100%)   │ SGLang (100%)   │  │
│  │ NEW Weights     │ NEW Weights     │ NEW Weights     │ NEW Weights     │  │
│  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Format Conversion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FORMAT FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TINKER API (JSON)                                                           │
│  {                                                                           │
│    "model_input": {                                                          │
│      "chunks": [{"tokens": [1, 2, 3, 4, 5]}]                                │
│    },                                                                        │
│    "loss_fn_inputs": {                                                       │
│      "target_tokens": {"data": [3, 4, 5], "dtype": "int64"},                │
│      "logprobs": {"data": [-2.1, -1.8, -3.2], "dtype": "float32"},          │
│      "advantages": {"data": [0.5, -0.3, 0.8], "dtype": "float32"}           │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
│                    ↓ TinkerDataConverter.forward_backward_to_rollout()       │
│                                                                              │
│  MILES ROLLOUT_DATA (torch.Tensor)                                           │
│  {                                                                           │
│    "tokens": [torch.tensor([1, 2, 3, 4, 5])],                               │
│    "loss_masks": [torch.tensor([0, 0, 1, 1, 1])],                           │
│    "log_probs": [torch.tensor([-2.1, -1.8, -3.2])],                         │
│    "rollout_log_probs": [torch.tensor([-2.1, -1.8, -3.2])],                 │
│    "advantages": [torch.tensor([0.5, -0.3, 0.8])]                           │
│  }                                                                           │
│                                                                              │
│                    ↓ Miles forward_backward_only()                           │
│                                                                              │
│  MILES RESULT                                                                │
│  {                                                                           │
│    "loss": {"sum": 2.5, "log_probs": [...]},                                │
│    "metrics": {"ppo_kl": 0.01, "pg_clipfrac": 0.05}                         │
│  }                                                                           │
│                                                                              │
│                    ↓ TinkerDataConverter.rollout_to_forward_backward_result()│
│                                                                              │
│  TINKER API RESPONSE (JSON)                                                  │
│  {                                                                           │
│    "loss_fn_outputs": [{"logprobs": {"data": [...], "dtype": "float32"}}],  │
│    "metrics": {"loss:sum": 2.5, "ppo_kl": 0.01}                             │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Async Operations Pattern

All long-running operations use a polling pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASYNC POLLING PATTERN                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Client                    Server                      Background Task       │
│    │                         │                              │                │
│    │  POST /forward_backward │                              │                │
│    │ ──────────────────────► │                              │                │
│    │                         │  Create task + future        │                │
│    │                         │ ────────────────────────────►│                │
│    │  {"request_id": "abc"}  │                              │                │
│    │ ◄────────────────────── │                              │                │
│    │                         │                              │ Processing...  │
│    │  POST /retrieve_future  │                              │                │
│    │ ──────────────────────► │                              │                │
│    │  {"status": "pending"}  │                              │                │
│    │ ◄────────────────────── │                              │                │
│    │                         │                              │                │
│    │  ... (poll every 100ms) │                              │ Done!          │
│    │                         │                              │                │
│    │  POST /retrieve_future  │                              │                │
│    │ ──────────────────────► │                              │                │
│    │  {"status":"completed", │                              │                │
│    │   "result": {...}}      │                              │                │
│    │ ◄────────────────────── │                              │                │
│    │                         │                              │                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **SlimeArgumentBuilder** | `core/slime_builder.py` | Build Miles/Megatron args from HF config |
| **TinkerDataConverter** | `core/data_converter.py` | Format conversion between Tinker and Miles |
| **TaskManager** | `core/task_manager.py` | Async task lifecycle management |
| **FuturesStorage** | `storage/futures.py` | SQLite persistence for async operations |
| **SessionStorage** | `storage/session_storage.py` | SQLite persistence for client sessions |
| **ModelService** | `services/model_service.py` | Miles actor creation and management |
| **TrainingService** | `services/training_service.py` | Training operations with GPU choreography |
| **SamplingService** | `services/sampling_service.py` | SGLang HTTP client wrapper |

## Next Steps

- [API Reference](api/endpoints.md) - Complete endpoint documentation
- [Data Conversion](concepts/data-conversion.md) - Format details
- [GPU Memory](concepts/gpu-memory.md) - Offload/onload explained
