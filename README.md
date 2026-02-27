# TinkerCloud Training API

Exposes the Tinker training surface while delegating RL training jobs to pluggable backends (Miles or NeMo RL).

## Overview

TinkerCloud is a FastAPI server that bridges the Tinker client API to one of two training backends — **Miles** (NVIDIA internal, Ray-based) or **NeMo RL** (open-source, push-mode). A `TrainingBackend` abstraction lets services stay backend-agnostic; the active backend is selected at startup via the `TINKERCLOUD_BACKEND` environment variable.

The `training/` package bundles:

- **HTTP Surface** – Modular FastAPI routers (`routers/`) that stay thin and forward work to services.
- **Business Logic** – Services (`services/`) that orchestrate data conversion, validation, and async task handling before delegating to the active backend.
- **Backend Abstraction** – `TrainingBackend` base class (`backends/base.py`) with Miles and NeMo RL implementations, wired through `BackendFactory`.
- **Core Utilities** – Validators, converters, and task managers (`core/`) that keep the HTTP layer stateless.
- **Storage** – Filesystem + SQLite helpers (`storage/`) for futures and metadata tracking.

```
                                          ┌─▶ Miles backend ─▶ Ray TrainGroup ─▶ Megatron actors
Tinker Client ─▶ FastAPI ─▶ Services ─▶ BackendFactory ─┤
                                          └─▶ NeMo RL backend ─▶ Policy.train() ─▶ Megatron workers
```

## Backends

| | Miles | NeMo RL |
|---|---|---|
| **Orchestration** | Ray `TrainGroup` + `RolloutManager` | `nemo_rl.Policy` push-mode API |
| **Generation** | SGLang (via Miles) | SGLang (native `nemo_rl.models.generation.sglang`) |
| **Parallelism** | Megatron TP/PP/DP | Megatron TP/PP/DP |
| **LoRA** | Supported | Supported (with weight merge for vLLM sync) |
| **Base image** | `gmicloudai/tinkercloud:dev` | `nvcr.io/nvidia/nemo-rl:v0.5.0` |
| **Selection** | `TINKERCLOUD_BACKEND=miles` (default) | `TINKERCLOUD_BACKEND=nemo_rl` |

## Features

- Drop-in compatibility with the Tinker training API (forward, forward-backward, optimizer, sampling, checkpointing).
- **Dual-backend support** — switch between Miles and NeMo RL via a single env var; the API surface is identical.
- Configurable runtime via `TrainingConfig` (env vars, `.env`, or explicit objects).
- Async background task tracking and polling (`/api/v1/retrieve_future`).
- Optional API key auth for parity with production kgateway deployments.
- Docker images for both backends.

## Requirements

- Python 3.10+
- Ray cluster (or local Ray runtime) reachable via `RAY_ADDRESS`
- **Miles backend**: Access to Miles/Slime codebase on the same filesystem or Python path
- **NeMo RL backend**: NeMo RL v0.5+ (`pip install nemo-rl` or the NVIDIA container)

## Quickstart

### Miles (default)

```bash
git clone <repo>/tinkercloud
cd tinkercloud
pip install -r requirements.txt

export TINKERCLOUD_BACKEND=miles
uvicorn training.api:app --reload --host 0.0.0.0 --port 8000
```

### NeMo RL

```bash
export TINKERCLOUD_BACKEND=nemo_rl
uvicorn training.api:app --reload --host 0.0.0.0 --port 8000
```

Once running, hit `http://localhost:8000/docs` for interactive OpenAPI docs, or call endpoints with the Tinker client by pointing `TINKER_BASE_URL` at your local server.

## Configuration

`training.config.TrainingConfig` reads environment variables at startup. Important knobs:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TINKERCLOUD_BACKEND` | Training backend (`miles` or `nemo_rl`) | `miles` |
| `TRAINING_HOST` / `TRAINING_PORT` | ASGI bind address | `0.0.0.0` / `8000` |
| `TINKER_API_KEY` | Shared secret for routers | `slime-dev-key` |
| `RAY_ADDRESS` / `RAY_NAMESPACE` | Connection info for Ray cluster | `auto` / `kgateway` |
| `METADATA_DIR` / `FUTURES_DB_NAME` | Local persistence for training metadata | `./data/metadata` / `futures.db` |
| `SUPPORTED_MODELS` | JSON array surfaced by `/api/v1/get_server_capabilities` | `[]` |
| `ALLOW_PARTIAL_BATCHES` | Allow batches not divisible by DP size (NeMo RL pads automatically) | `false` |

To supply a custom config object (useful for tests), import `create_app`:

```python
from training.api import create_app
from training.config import TrainingConfig

config = TrainingConfig.from_file("config/training.yaml")
app = create_app(config)
```

## Common Workflows

### Development Server

```bash
uvicorn training.api:app --reload --host 0.0.0.0 --port 8000
```

### Formatting & Linting

```bash
ruff check training
black training
```

### Running Tests

```bash
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=slime-dev-key

# Unit & parity tests
pytest tests/test_backend_interface.py tests/test_backend_parity.py

# End-to-end (NeMo RL)
bash tests/test_e2e_nemo_rl.sh
```

## Docker

### Miles

```bash
cd docker
docker build -t opentinker/miles-training:latest .
docker run -p 8000:8000 \
  -e TINKERCLOUD_BACKEND=miles \
  -e RAY_ADDRESS=ray://miles-ray:10001 \
  -e TINKER_API_KEY=slime-dev-key \
  opentinker/miles-training:latest
```

### NeMo RL

```bash
cd docker
docker build -f Dockerfile.dev.nemo_rl -t opentinker/nemo-rl-training:latest .
docker run --gpus all -p 8000:8000 \
  -e TINKERCLOUD_BACKEND=nemo_rl \
  -e TINKER_API_KEY=slime-dev-key \
  opentinker/nemo-rl-training:latest
```

## Sessions

See [docs/sessions.md](docs/sessions.md) for detailed documentation on session management.

## API Surface

The API is identical regardless of backend. All endpoints work with both Miles and NeMo RL.

| Category | Endpoint | Description |
|----------|----------|-------------|
| Training | `POST /api/v1/forward_backward` | Execute forward/backward pass (DPO/SFT/RL) |
| Training | `POST /api/v1/forward` | Forward-only reference run (logprobs) |
| Training | `POST /api/v1/optim_step` | Apply optimizer step once gradients are accumulated |
| Training | `POST /api/v1/retrieve_future` | Poll background tasks |
| Sampling | `POST /api/v1/sample` | Generate sequences with the latest weights |
| Checkpoints | `POST /api/v1/save_weights` | Persist checkpoint to disk |
| Checkpoints | `POST /api/v1/save_weights_for_sampler` | Save weights and get sampling client |
| Sessions | See [docs/sessions.md](docs/sessions.md) | Session management endpoints |
| Health | `GET /api/v1/health` | Lightweight readiness probe |

See `training/routers` for the full list of endpoints.

## Project Layout

```
training/
├── api.py                  # FastAPI app factory
├── config.py               # Pydantic config (incl. BackendConfig)
├── backends/
│   ├── base.py             # TrainingBackend ABC, BackendHandle, BackendError
│   ├── factory.py          # BackendFactory.create("miles"|"nemo_rl")
│   ├── miles/              # Miles backend (Ray TrainGroup)
│   │   ├── backend.py
│   │   ├── builder.py
│   │   └── converter.py
│   └── nemo_rl/            # NeMo RL backend (Policy.train push-mode)
│       ├── backend.py
│       ├── builder.py
│       └── converter.py
├── routers/                # FastAPI route handlers
├── services/               # Business logic (backend-agnostic)
├── core/                   # Validators, converters, task managers
└── storage/                # Filesystem + SQLite persistence
```

## Troubleshooting

- **Cannot reach Ray** – verify `RAY_ADDRESS` and that Ray head is running (`ray status`). Local dev can use `ray start --head`.
- **Authentication failures** – set `TINKER_API_KEY` on both the server and client. Disable auth via config if needed.
- **Reference logprob mismatches** – ensure dataset weights include the prompt mask; the converter trims/pads according to incoming `loss_fn_inputs["weights"]`.
- **NeMo RL generation stops ignored** – check that `sampling_params.stop` is set; the NeMo RL backend forwards stop strings via `_tinker_`-prefixed BatchedDataDict fields.
- **NeMo RL partial batches** – set `ALLOW_PARTIAL_BATCHES=true`; the NeMo RL converter pads to the nearest DP-divisible size automatically.
