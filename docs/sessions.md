---
layout: default
title: Sessions
nav_order: 5
---

# TinkerCloud Session Management

This document describes how TinkerCloud implements session management to track client connections, training runs, and sampling endpoints.

## Overview

Sessions provide:
- **Multi-tenant isolation** - Each client gets a unique session_id
- **Resource ownership** - Sessions track all training runs and samplers created within them
- **Keepalive mechanism** - Regular heartbeats prevent session timeout
- **State persistence** - Sessions survive server restarts via SQLite storage

## Session Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ServiceClient (session_id)                                                   │
│ └─ Created automatically when client connects                               │
│                                                                              │
├─── TrainingClient #1 (model_id, model_seq_id=1)                             │
│    └─ Created via create_lora_training_client()                             │
│    └─ Owns Ray actors: RayTrainGroup + RolloutManager                       │
│                                                                              │
├─── TrainingClient #2 (model_id, model_seq_id=2)                             │
│    └─ Multiple training clients can exist per session                       │
│                                                                              │
└─── SamplingClient (sampling_session_id)                                      │
     └─ Created from TrainingClient after weight save                         │
     └─ Points to SGLang inference endpoint                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Session Types

### 1. Parent Session

Created when a `ServiceClient` is instantiated in tinker_gmi:

```python
service_client = tinker.ServiceClient(base_url="http://localhost:8000")
# Automatically creates session via POST /api/v1/create_session
```

**Properties:**
- `session_id` - UUID identifier
- `tags` - Labels for filtering (from `TINKER_TAGS` env var)
- `user_metadata` - Arbitrary key-value metadata
- `sdk_version` - Client SDK version

### 2. Training Session (Model Instance)

Created when a `TrainingClient` is requested:

```python
training_client = await service_client.create_lora_training_client_async(
    "Qwen/Qwen2.5-0.5B-Instruct",
    rank=32
)
```

**Properties:**
- `model_id` - UUID for the model instance
- `model_seq_id` - Sequential ID within the session (for ordering)
- Linked to Ray actors (RayTrainGroup, RolloutManager)

### 3. Sampling Session

Created in two ways:

**A. Explicit creation** (standalone sampler):
```python
sampling_client = await service_client.create_sampling_client_async(
    base_model="Qwen/Qwen2.5-0.5B-Instruct"
)
```

**B. From training client** (after weight update):
```python
sampling_client = await training_client.save_weights_and_get_sampling_client_async()
```

**Properties:**
- `sampling_session_id` - Format: `{session_id}_{seq_id}_{uuid[:8]}`
- Points to SGLang router for inference

## Session Lifecycle

### Creation Flow

```
Client                          TinkerCloud
  │                                  │
  ├─ ServiceClient() ───────────────►│ POST /api/v1/create_session
  │                                  │   → Generate session_id
  │                                  │   → Store in SessionService + SQLite
  │◄─ session_id ────────────────────┤
  │                                  │
  ├─ create_training_client() ──────►│ POST /api/v1/create_model
  │                                  │   → Link model to session
  │                                  │   → Create Ray actors
  │◄─ TrainingClient ────────────────┤
  │                                  │
  ├─ save_weights_and_get_sampler() ►│ POST /api/v1/save_weights_for_sampler
  │                                  │   → Register ephemeral sampler
  │◄─ SamplingClient ────────────────┤
```

### Heartbeat Mechanism

Clients send heartbeats every 10 seconds to keep sessions alive:

```python
# tinker_gmi internal (automatic)
SESSION_HEARTBEAT_PERIOD_SEC = 10
SESSION_MISSED_HEARTBEAT_WARNING_THRESHOLD_SEC = 600  # 10 minutes
```

**Server handling:**
- Updates `last_heartbeat` timestamp
- Logs warning if gap > 600 seconds
- Stale sessions cleaned up after 24 hours

### Cleanup

On server startup:
```python
# api.py startup
session_storage.cleanup_stale_sessions(max_age_hours=24)
```

## API Endpoints

### POST /api/v1/create_session

Create a new parent session.

**Request:**
```json
{
  "tags": ["experiment-1", "rlve"],
  "user_metadata": {"user": "gavin"},
  "sdk_version": "0.1.0"
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### POST /api/v1/session_heartbeat

Keep session alive.

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### POST /api/v1/create_sampling_session

Create a sampling session within a parent session.

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "sampling_session_seq_id": 1,
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "model_path": null
}
```

**Response:**
```json
{
  "sampling_session_id": "550e8400_1_a1b2c3d4"
}
```

### GET /api/v1/sessions

List all active sessions.

**Response:**
```json
{
  "sessions": ["session-1", "session-2", "session-3"]
}
```

### GET /api/v1/sessions/{session_id}

Get session details including owned resources.

**Response:**
```json
{
  "training_run_ids": ["model-1", "model-2"],
  "sampler_ids": ["sampler-1", "sampler-2"]
}
```

## Storage Architecture

### SQLite Schema

```sql
-- Parent sessions
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    sdk_version TEXT,
    tags TEXT,           -- JSON array
    user_metadata TEXT,  -- JSON object
    created_at TIMESTAMP,
    last_heartbeat TIMESTAMP
);

-- Sampling sessions
CREATE TABLE samplers (
    sampler_id TEXT PRIMARY KEY,
    session_id TEXT,
    model_id TEXT,
    base_model TEXT,
    model_path TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Model-to-session mapping
CREATE TABLE session_models (
    session_id TEXT,
    model_id TEXT PRIMARY KEY,
    model_seq_id INTEGER,
    base_model TEXT,
    model_path TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indices
CREATE INDEX idx_samplers_session ON samplers(session_id);
CREATE INDEX idx_sessions_heartbeat ON sessions(last_heartbeat);
CREATE INDEX idx_session_models_seq ON session_models(session_id, model_seq_id);
```

### In-Memory Service

`SessionService` maintains fast lookups:

```python
class SessionService:
    _sessions: Dict[str, SessionInfo]           # session_id → info
    _samplers: Dict[str, SamplerInfo]           # sampler_id → info
    _model_to_session: Dict[str, str]           # model_id → session_id
```

## Session-to-Ray Actor Mapping

When a model is created, it's linked to Ray actors:

```python
# model_service.py
training_clients[model_id] = {
    "train_group": RayTrainGroup,           # Megatron training actors
    "rollout_manager": RolloutManager,      # SGLang inference
    "placement_group": PlacementGroup,      # GPU allocation
    "router_ip": "10.0.0.1",
    "router_port": 30000,
}

# session_service.py - bidirectional mapping
session_service.add_model(session_id, model_id, model_seq_id, ...)
# Creates: session.model_ids.append(model_id)
#          _model_to_session[model_id] = session_id
```

## Integration with Training

### Model Creation with Session Linking

```python
# routers/models.py
@router.post("/api/v1/create_model")
async def create_model(request: CreateModelRequest):
    model_id = generate_model_id()

    # Link to session
    session_service.add_model(
        session_id=request.session_id,
        model_id=model_id,
        model_seq_id=request.model_seq_id,
        base_model=request.base_model
    )

    # Create Ray actors...
```

### Ephemeral Sampler Registration

After training updates weights:

```python
# routers/checkpoints.py
@router.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(request):
    # Push weights to SGLang
    result = await service.save_weights_for_sampler(...)

    # Register ephemeral sampler
    session_service.register_ephemeral_sampler(
        sampler_id=result["sampling_session_id"],
        model_id=request.model_id,
        base_model=base_model
    )
```

### Weight Sync Flow

```
TrainingClient                    TinkerCloud                      Ray Actors
     │                                 │                                │
     ├─ optim_step() ─────────────────►│                                │
     │                                 ├─ apply_optimizer_step() ──────►│
     │                                 │◄─ gradients applied ───────────┤
     │                                 │                                │
     ├─ save_weights_and_get_sampler()►│                                │
     │                                 ├─ train_group.offload() ───────►│
     │                                 ├─ update_weights() ────────────►│ (Megatron→SGLang)
     │                                 ├─ rollout_manager.onload() ────►│
     │                                 │                                │
     │                                 ├─ register_ephemeral_sampler()  │
     │◄─ SamplingClient ───────────────┤                                │
```

## Key Files

| File | Purpose |
|------|---------|
| `training/routers/session.py` | Session API endpoints |
| `training/services/session_service.py` | In-memory session management |
| `training/storage/session_storage.py` | SQLite persistence |
| `training/routers/models.py` | Model creation with session linking |
| `training/routers/checkpoints.py` | Ephemeral sampler registration |
| `training/api.py` | Startup initialization |
