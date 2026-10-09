# `training/` — the TinkerCloud server

FastAPI implementation of the Tinker training API over pluggable backends. One
server process serves one backend, chosen at startup; clients never change.

```
tinker SDK ──HTTP──▶ routers/ ──▶ services/ ──▶ backends/<backend>/ ──▶ Ray actors (GPUs)
                        │              │
                   models/ (pydantic)  core/ (TaskManager, ordering, loss registry, validators)
                                       storage/ (futures, sessions, checkpoint metadata)
```

## Layout

| Path | Role |
|---|---|
| `api.py` | `create_app(config)`: wires storage, the selected backend, services and routers onto `app.state`; startup/shutdown hooks (Ray init, session reaper, model teardown). `app` is the env-configured instance for `uvicorn training.api:app`. |
| `__main__.py` | CLI: `python -m training [--backend B] [--host H] [--port P]`. |
| `config.py` | `TrainingConfig` (server, storage, Ray, auth, backend selection) from the environment or a JSON/YAML file. |
| `routers/` | Thin HTTP handlers, one module per area: `training`, `models`, `sampling`, `checkpoints`, `session`, `futures`, `health`. |
| `services/` | Backend-agnostic orchestration: `model_service`, `training_service`, `sampling_service`, `checkpoint_service`, `session_service`, `ordering` (per-model program order and barriers). |
| `backends/` | `base.py` ABCs (`TrainingBackend`, `ArgumentBuilder`, `DataConverter`), `factory.py` (`SUPPORTED_BACKENDS`), `env_config.py` (typed per-backend config base), one package per backend: `miles/`, `nemo_rl/`, `verl/`, `automodel/`, `megatron_bridge/`, `fake/` (deterministic CPU backend for the protocol suite). |
| `core/` | Backend-agnostic: `task_manager` (async futures + result validation), `loss_registry` (`loss_fn` / `loss_fn_config`), `validators`, `dependencies`. |
| `models/` | Pydantic request and response models; `responses.RESULT_MODELS` is what every async result is validated against. |
| `storage/` | `FuturesStorage` (SQLite, seq_id idempotency), `SessionStorage`, `MetadataStorage` (checkpoints). |
| `utils/` | Auth, id/helpers, backend-agnostic model helpers (`model_config.py`). |
| `test_data/` | The Miles boot-time fallback dataset (see its README). |

Adding a backend: `backends/<name>/{backend,builder,converter,config}.py`,
register in `factory._BACKEND_CLASSES`. Nothing under `core/`, `services/` or
`utils/` may import from a backend package (specs/DESIGN-NOTES D7).

## Endpoints

Async operations return `{request_id}`; retrieve with `POST /api/v1/retrieve_future/{request_id}`
(200 result, 408 still running, 400 failed). Every async result is validated
against its response model before it is stored.

| Area | Endpoints |
|---|---|
| Sessions | `POST create_session`, `POST session_heartbeat`, `GET sessions`, `GET sessions/{id}`, `POST create_sampling_session`, `GET samplers/{id}` |
| Models | `POST create_model` (async), `POST get_info`, `GET get_tokenizer`, `POST unload_model` (async), `POST delete_model`, `GET training_runs/{model_id}` |
| Training | `POST forward` / `forward_backward` / `optim_step` (async; per-model program order, `seq_id` retries idempotent) |
| Checkpoints | `POST save_weights`, `POST save_weights_for_sampler`, `POST load_weights` (first request only), `POST weights_info`, `GET training_runs/{id}/checkpoints`, `DELETE training_runs/{id}/checkpoints/{weights\|sampler_weights\|external_weights}/{ckpt}`, `POST save_weights_external`, `GET training_runs/{id}/checkpoints/external_weights/{name}/external_weights_urls`, `GET external_weights/{id}/{name}/{file}?exp&sig` (signed, no API key) |
| Sampling | `POST create_sampling_client`, `POST sample`, `POST asample` (async) — a request resolves strictly to its sampler's model; bare `base_model` sampling is 400 |
| Ops | `GET /health`, `GET get_server_capabilities`, `POST cleanup_futures`, `POST telemetry` |

All under `/api/v1/` except `/health`. Every route but `/health` and `telemetry`
requires `X-API-Key`.

## Running

```bash
python -m training --backend nemo_rl            # CLI (flag > TINKERCLOUD_BACKEND > miles)
TINKERCLOUD_BACKEND=miles uvicorn training.api:app --host 0.0.0.0 --port 8000
python -m training --backend fake --port 8001   # no Ray, no GPU: the protocol suite's server
```

Programmatic: `from training.api import create_app; app = create_app(TrainingConfig.from_file(path))`.

## Configuration

`TrainingConfig` reads the environment (see the table in the repository README).
Backend knobs are declared fields on `backends/miles/config.py` (`MilesConfig`,
`SLIME_*` / `TINKERCLOUD_MILES_*`) and `backends/nemo_rl/config.py`
(`NemoRLConfig`, `NEMORL_*` / `NRL_*`); `TINKERCLOUD_BACKEND_OVERRIDES` (a JSON
object keyed by field name) wins over the environment, and the effective values
with their sources are logged at startup.

## Backend contract in one paragraph

`forward_backward` on Miles runs on the GPU immediately and returns per-sample
logprobs; on NeMo RL it buffers and `optim_step` runs `policy.train()` once over
the buffer (the R9 deferred contract, DESIGN-NOTES D5), returning the logprobs
with the step. Both return one logprob per target token (D-2026-08-31).
`optim_step`, saves, `load_weights` and `unload_model` are barriers in the
per-model order; passes may overlap.

## Tests

`tests/protocol/` (SDK against a live `fake`-backend server) and the unit tests
under `tests/` need no GPU; `scripts/pod_pytest.sh` runs them on a cluster pod
against your working tree. See `tests/README.md`.
