# TinkerCloud Training API

FastAPI server that bridges the Tinker client SDK to pluggable training backends.

```
Tinker Client --> FastAPI --> BackendFactory --> Miles or NeMo RL --> Ray + Megatron GPUs
```

## Backends

| | Miles | NeMo RL |
|---|---|---|
| Orchestration | Ray `TrainGroup` + `RolloutManager` | `nemo_rl.Policy` push-mode API |
| Generation | SGLang (via Miles) | vLLM (via `nemo_rl.models.generation`) |
| Base image | `gmicloudai/tinkercloud:dev-local` | `nvcr.io/nvidia/nemo-rl:v0.5.0` |
| Selection | `TINKERCLOUD_BACKEND=miles` (default) | `TINKERCLOUD_BACKEND=nemo_rl` |

## Quickstart (NeMo RL)

### 1. Build

From the monorepo root (parent of `tinker-cloud/`, `RL/`, `tinker_gmi/`, `tinker-cookbook/`):

```bash
cd tinker-cloud
./docker/build_dev.sh --backend nemo_rl
```

### 2. Run

```bash
docker run -d --name tinkercloud-nemo-rl \
  --gpus all \
  -v /path/to/models:/data/models \
  --network host \
  --shm-size=16g \
  -e ALLOW_PARTIAL_BATCHES=true \
  -e NCCL_NVLS_ENABLE=1 \
  -e NCCL_SHM_DISABLE=1 \
  -e NCCL_IGNORE_DISABLED_P2P=1 \
  gmicloudai/tinkercloud:dev-nemo-rl
```

The entrypoint starts Ray (auto-detects GPUs) and the training API on port 8000.

### 3. Run a recipe

```bash
docker exec -d tinkercloud-nemo-rl bash -c '
python -m tinker_cookbook.recipes.math_rl.train \
  model_name=/data/models/Llama-3.2-1B \
  base_url=http://localhost:8000 \
  group_size=4 \
  groups_per_batch=100 \
  learning_rate=1e-4 \
  lora_rank=32 \
  log_path=/tmp/math_rl
'
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TINKERCLOUD_BACKEND` | `miles` or `nemo_rl` | `miles` |
| `TRAINING_HOST` / `TRAINING_PORT` | Bind address | `0.0.0.0` / `8000` |
| `TINKER_API_KEY` | API auth key | `tml-dev-key` |
| `RAY_ADDRESS` | Ray cluster endpoint | auto (local head) |
| `ALLOW_PARTIAL_BATCHES` | Pad batches smaller than DP size | `false` |
| `HF_HOME` | HuggingFace model cache | `/data/models` |

## API

The API is backend-agnostic. All endpoints work with both Miles and NeMo RL.

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/create_model` | Initialize model on GPUs |
| `POST /api/v1/forward_backward` | Forward/backward pass |
| `POST /api/v1/forward` | Forward-only (logprobs) |
| `POST /api/v1/optim_step` | Apply optimizer step |
| `POST /api/v1/sample` | Generate with latest weights |
| `POST /api/v1/save_weights_for_sampler` | Save weights + get sampling endpoint |
| `POST /api/v1/retrieve_future` | Poll async tasks |
| `GET /api/v1/health` | Readiness probe |

## Project Layout

```
training/
├── api.py              # FastAPI app factory + startup
├── config.py           # Pydantic config from env vars
├── server.py           # uvicorn runner
├── backends/           # Backend abstraction
│   ├── base.py         # TrainingBackend ABC
│   ├── factory.py      # BackendFactory
│   ├── miles/          # Miles/Slime implementation
│   └── nemo_rl/        # NeMo RL implementation
├── routers/            # HTTP handlers
├── services/           # Business logic
├── models/             # Request/response schemas
├── storage/            # SQLite persistence
└── utils/              # Auth, model config helpers
```
