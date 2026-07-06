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

## Verified tinker-cookbook Recipes (NeMo RL backend)

Validated end-to-end against this server with each recipe's README hyperparameters.

| Recipe | Type | Model | Verdict |
|---|---|---|---|
| math_rl (arithmetic) | RL/GRPO | Llama-3.2-1B | **Pass** — correct 0.74 → 1.00 in 4 batches, stable through 21 |
| rl_basic | RL | Llama-3.2-1B | **Pass** — loss/reward healthy from batch 0 |
| rl_loop | RL (sync) | Qwen2.5-0.5B | **Pass** — 130+ batches, reward 0.0 → 0.5 |
| multiplayer/guess_number | multi-turn RL | Qwen3-4B-Instruct | **Pass** — reward 0.32 → 0.43 over 6 batches |
| preference/shorter | pairwise-preference RL | Qwen3-4B-Instruct | **Pass** — ac_tokens/turn 63.5 → 36.6 by step 22 (target: significant drop by 40) |
| sl_basic | SFT | Llama-3.2-1B | **Pass** — 13+ steps, loss decreasing, grad_norm stable |
| sl_loop | SFT (sync) | Qwen2.5-0.5B | **Pass** — 27+ steps, checkpointing OK |
| chat_sl (no_robots) | SFT (async) | Qwen3-8B-Base | **Pass** — test/nll 1.871 → 1.800 by step 20 (recipe ref: 1.788 @ 140) |
| preference/dpo | DPO | Llama-3.2-1B | **Pass** (mechanics) — 34 steps error-free, gradients flowing; loss ~ln(2) flat in 34 steps at documented lr=1e-5, longer run needed for convergence verdict |

Full status for all 23 recipe entries, bugs, and gaps: `specs/002-nemorl-recipe-testing/compatibility-matrix.md` in the monorepo.

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
