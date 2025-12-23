# OpenTinker Miles Training API

Exposes the Tinker training surface while delegating RL training jobs to Miles/Slime Ray actors.

## Overview

This repo was originally built within the kgateway training module so it can run independently for local iteration, CI, and custom deployments. The `training/` package bundles:

- **HTTP Surface** – Modular FastAPI routers (`routers/`) that stay thin and forward work to services.
- **Business Logic** – Services (`services/`) that orchestrate data conversion, validation, and async task handling before talking to Miles.
- **Core Utilities** – Validators, converters, and task managers (`core/`) that keep the HTTP layer stateless.
- **Storage** – Filesystem + SQLite helpers (`storage/`) for futures and metadata tracking.

```
Tinker Client ─▶ FastAPI (training/api.py) ─▶ Ray TrainGroup ─▶ Miles / Megatron actors
```

## Features

- Drop-in compatibility with the Tinker training API (forward, forward-backward, optimizer, sampling, checkpointing).
- Configurable runtime via `TrainingConfig` (env vars, `.env`, or explicit objects).
- Async background task tracking and polling (`/api/v1/retrieve_future`).
- Optional API key auth for parity with production kgateway deployments.
- Docker image for parity testing in cluster environments.

## Requirements

- Python 3.11+
- Ray cluster (or local Ray runtime) reachable via `RAY_ADDRESS`
- Access to Miles/Slime codebase on the same filesystem or Python path (see `requirements.txt`)

## Quickstart

```bash
git clone <repo>/opentinker-miles
cd opentinker-miles
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the API with live reload on :8000
uvicorn training.api:app --reload --host 0.0.0.0 --port 8000
```

Once running, hit `http://localhost:8000/docs` for interactive OpenAPI docs, or call endpoints with the Tinker client by pointing `TINKER_BASE_URL` at your local server.

## Configuration

`training.config.TrainingConfig` reads environment variables at startup. Important knobs:

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRAINING_HOST` / `TRAINING_PORT` | ASGI bind address | `0.0.0.0` / `8000` |
| `TINKER_API_KEY` | Shared secret for routers | `slime-dev-key` |
| `RAY_ADDRESS` / `RAY_NAMESPACE` | Connection info for Ray cluster | `auto` / `kgateway` |
| `METADATA_DIR` / `FUTURES_DB_NAME` | Local persistence for training metadata | `./data/metadata` / `futures.db` |
| `SUPPORTED_MODELS` | JSON array surfaced by `/api/v1/get_server_capabilities` | `[]` |

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

Unit tests live alongside modules; integration tests can be run via Tinker:

```bash
export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=slime-dev-key
pytest tests  # or run tinker client flows
```

## Docker

Build a reproducible image that bundles the FastAPI app and dependencies:

```bash
cd docker
docker build -t opentinker/miles-training:latest .
docker run -p 8000:8000 \
  -e RAY_ADDRESS=ray://miles-ray:10001 \
  -e TINKER_API_KEY=slime-dev-key \
  opentinker/miles-training:latest
```

## Sessions

Sessions provide a way to manage multiple models within a single training workflow (e.g., DPO with separate policy and reference models).

### Creating a Session

```python
import requests

# Create a new session
resp = requests.post(
    "http://localhost:8000/api/v1/session",
    headers={"X-API-Key": "slime-dev-key"},
    json={"session_id": "my-dpo-session"}  # optional, auto-generated if omitted
)
session = resp.json()
print(session["session_id"])
```

### Loading Models into a Session

```python
# Load policy model
requests.post(
    "http://localhost:8000/api/v1/session/my-dpo-session/models",
    headers={"X-API-Key": "slime-dev-key"},
    json={
        "model_name": "/data/models/Qwen2.5-0.5B-Instruct",
        "role": "policy"  # optional metadata
    }
)

# Load reference model (same weights, different instance)
requests.post(
    "http://localhost:8000/api/v1/session/my-dpo-session/models",
    headers={"X-API-Key": "slime-dev-key"},
    json={
        "model_name": "/data/models/Qwen2.5-0.5B-Instruct",
        "role": "reference"
    }
)
```

### Session Lifecycle

```python
# List models in session
resp = requests.get(
    "http://localhost:8000/api/v1/session/my-dpo-session",
    headers={"X-API-Key": "slime-dev-key"}
)

# Delete session (unloads all models)
requests.delete(
    "http://localhost:8000/api/v1/session/my-dpo-session",
    headers={"X-API-Key": "slime-dev-key"}
)
```

## API Surface

### Training
- `POST /api/v1/forward_backward` – Execute forward/backward pass (DPO/SFT/RL)
- `POST /api/v1/forward` – Forward-only reference run (logprobs)
- `POST /api/v1/optim_step` – Apply optimizer step once gradients are accumulated
- `POST /api/v1/retrieve_future` – Poll background tasks

### Sampling
- `POST /api/v1/sample` – Generate sequences with the latest weights

### Checkpoints
- `POST /api/v1/save_weights` / `save_weights_for_sampler` – Persist checkpoints

### Sessions
- `POST /api/v1/session` – Create a new session
- `GET /api/v1/session/{session_id}` – Get session details and models
- `DELETE /api/v1/session/{session_id}` – Delete session and unload models
- `POST /api/v1/session/{session_id}/models` – Load a model into session
- `DELETE /api/v1/session/{session_id}/models/{model_id}` – Unload model from session

### Health
- `GET /api/v1/health` – Lightweight readiness probe

See `training/routers` for the full list of endpoints.

## Troubleshooting

- **Cannot reach Ray** – verify `RAY_ADDRESS` and that Ray head is running (`ray status`). Local dev can use `ray start --head`.
- **Authentication failures** – set `TINKER_API_KEY` on both the server and client. Disable auth via config if needed.
- **Reference logprob mismatches** – ensure dataset weights include the prompt mask; the converter trims/pads according to incoming `loss_fn_inputs["weights"]`.
