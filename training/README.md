# Training API Module

Implements the Tinker API for RL training with Slime backend integration.

## Overview

This module exposes the Tinker-compatible training surface as a FastAPI application. It now boots via an explicit `create_app()` factory (see `api.py`) that wires configuration, storage, runtime state, and router/service dependencies onto `app.state`. That keeps the HTTP layer decoupled from the underlying Slime/Ray runtime and makes unit testing or alternate configs straightforward.

## Architecture

```
Tinker Client → kgateway (Envoy) → Training API → Slime Backend (Ray)
```

Key pieces:
- `api.py` — FastAPI app factory + startup/shutdown hooks; stores config, `TrainingRuntimeState`, storages, auth, and service singletons on `app.state`.
- `routers/` — Thin HTTP handlers that resolve dependencies via `Depends(request.app.state...)` and delegate to services.
- `services/` — Business logic (model creation, training, sampling, checkpoints) that talk to Slime actors through Ray.
- `core/` — Shared helpers (argument builders, validators, task manager, dependency utilities).
- `storage/` — Futures/metadata abstractions backed by SQLite + filesystem.

## Endpoints

### Model Management
- `POST /api/v1/create_model` - Initialize training client with LoRA
- `GET /api/v1/get_server_capabilities` - Query available models
- `GET /api/v1/get_tokenizer` - Get tokenizer information

### Training Operations
- `POST /api/v1/forward_backward` - Execute forward-backward pass
- `POST /api/v1/optim_step` - Apply optimizer step

### Sampling (GRPO Support)
- `POST /api/v1/save_weights_for_sampler` - Save checkpoint for inference
- `POST /api/v1/create_sampling_client` - Create sampling client
- `POST /api/v1/sample` - Generate text samples

### State Management
- `POST /api/v1/retrieve_future` - Poll async operation status
- `POST /api/v1/save_weights` - Save model checkpoint

## Configuration

`TrainingConfig` (see `config.py`) loads from environment by default, but you can also provide a JSON/YAML file and pass it to the factory:

```python
from pathlib import Path
from kgateway.python.ai_extension.training.config import TrainingConfig
from kgateway.python.ai_extension.training.api import create_app

config = TrainingConfig.from_file(Path("config/training.yaml"))
app = create_app(config)
```

Common env vars:
- `TRAINING_HOST` / `TRAINING_PORT` — server binding (defaults `0.0.0.0:8000`)
- `KGATEWAY_LOG_LEVEL` — logging level (default `INFO`)
- `TINKER_API_KEY` — API key (default `slime-dev-key`, override in production)
- `SUPPORTED_MODELS` — JSON list consumed by `/api/v1/get_server_capabilities`
- `METADATA_DIR`, `FUTURES_DB_NAME` — storage overrides
- `RAY_ADDRESS`, `RAY_NAMESPACE` — Ray connection info
- `ALLOW_PARTIAL_BATCHES` — set to `true` to let forward_backward accept batches not divisible by the data-parallel size (Miles scales gradients dynamically)

## Running

### Standalone
```bash
uvicorn kgateway.python.ai_extension.training.api:app --host 0.0.0.0 --port 8000
```

### Custom app (alternate config, test harness, etc.)
```bash
from kgateway.python.ai_extension.training.api import create_app
custom_app = create_app(my_training_config)
```

## Integration with Slime

The Training API communicates with Slime via Ray:
1. Initializes Ray actors for training (RayTrainGroup)
2. Submits training batches to actors
3. Collects metrics and gradients
4. Manages model checkpoints

## Testing

Run E2E tests using Tinker client:
```bash
# From tinker_gmi directory
cd /tmp/tmux-tmp/tinker_gmi/tests_integration/e2e_tinker_api

export TINKER_BASE_URL=http://localhost:8000
export TINKER_API_KEY=slime-dev-key

python test_tinker_gmi_wrapper.py
```

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `ray` - Distributed compute for Slime
- `torch` - PyTorch for training
- `transformers` - Tokenizer support
