---
layout: default
title: Installation
nav_order: 2
---

# Installation & Setup

This guide covers setting up tinkercloud for local development and deployment.

## Prerequisites

- Python 3.10+
- Ray cluster with GPU nodes
- Miles/Megatron-LM installed
- SGLang installed
- Converted model checkpoints (torch_dist format)

## Environment Setup

### 1. Install Dependencies

```bash
# Install tinkercloud
cd /root/gavin/tinkercloud
pip install -e .

# Install Miles (with RLVE support)
cd /root/gavin/miles
pip install -e .

# Install tinker_gmi client
cd /root/gavin/tinker_gmi
pip install -e .

# Install tinker-cookbook (optional, for recipes)
cd /root/gavin/tinker-cookbook
pip install -e .
```

### 2. Set Environment Variables

```bash
# API key for authentication
export TINKER_API_KEY=slime-dev-key

# Wandb logging (optional)
export WANDB_API_KEY=your-wandb-key

# Python path for Miles/Megatron
export PYTHONPATH=/root/Megatron-LM:/root/miles:$PYTHONPATH
```

### 3. Start Ray Cluster

```bash
# Check Ray status
ray status

# If not running, start Ray head node
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Verify Ray is running
ray status
```

**Important:** Don't stop Ray - the pod liveness probe depends on Ray dashboard (port 8265).

### 4. Prepare Model Checkpoints

Models must be converted to Megatron torch_dist format:

```bash
# Available converted models
ls /data/models/

# Example models:
# /data/models/Qwen2.5-0.5B-Instruct_torch_dist
# /data/models/Qwen2.5-7B-Instruct_torch_dist
# /data/models/Qwen3-4B-Instruct-2507_torch_dist
```

## Starting the Server

### Option A: tinkercloud (Recommended)

```bash
# Kill any existing server
pkill -9 -f "uvicorn.*training" || true

# Clear Python cache
find /root/gavin/tinkercloud -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Start tinkercloud server
cd /root/gavin/tinkercloud
ALLOW_PARTIAL_BATCHES=true \
  PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/miles:$PYTHONPATH \
  nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 \
  >> /data/logs/tinkercloud.log 2>&1 &

# Wait for startup
sleep 5
```

### Option B: kgateway (Original)

```bash
cd /root/gavin/kgateway/python
PYTHONPATH=/root/Megatron-LM:/root/miles:$PYTHONPATH \
  nohup python3 -m uvicorn ai_extension.training.server:app --host 0.0.0.0 --port 8000 \
  >> /data/logs/kgateway-api.log 2>&1 &
```

## Verifying Installation

### 1. Check Server Health

```bash
curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python -m json.tool
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ray_status": "connected"
}
```

### 2. Check Server Capabilities

```bash
curl -s http://localhost:8000/api/v1/get_server_capabilities \
  -H "X-API-Key: slime-dev-key" | python -m json.tool
```

### 3. Test with Python Client

```python
import tinker

client = tinker.ServiceClient(
    base_url="http://localhost:8000",
    api_key="slime-dev-key"
)

# Should return server capabilities
caps = client.get_server_capabilities()
print(caps)
```

## Cleanup Commands

### Cleanup Existing Sessions

Before starting new training, cleanup any existing sessions to free GPUs:

```bash
TINKER_BASE_URL=http://localhost:8000 \
  TINKER_API_KEY=slime-dev-key \
  python /root/gavin/tinker_gmi/tests_integration/cleanup_test_env.py
```

### Restart Ray (Clears All Actors)

```bash
# Stop and restart Ray (clears all GPU allocations)
ray stop --force && sleep 3 && \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# IMPORTANT: Must also restart tinkercloud after Ray restart!
```

### Check GPU Usage

```bash
nvidia-smi
```

## Logs

| Log File | Content |
|----------|---------|
| `/data/logs/tinkercloud.log` | Server logs |
| `/data/logs/tinker-rlve.log` | RLVE training client logs |
| `/data/logs/kgateway-api.log` | kgateway server logs (if using) |

### View Logs

```bash
# Follow server logs
tail -f /data/logs/tinkercloud.log

# Search for errors
grep -E "(ERROR|Exception|Traceback|RuntimeError)" /data/logs/tinkercloud.log | tail -20
```

## Directory Structure

```
/root/gavin/
├── tinkercloud/          # Middleware server
├── miles/                # Training backend
├── tinker_gmi/           # Python client SDK
├── tinker-cookbook/      # Training recipes
├── kgateway/             # Original middleware (deprecated)
└── Megatron-LM/          # Megatron framework

/data/
├── models/               # Converted model checkpoints
├── checkpoints/          # Training checkpoints
└── logs/                 # Server and training logs
```

## Next Steps

- [Quickstart](quickstart.md) - Run your first training
- [Architecture](architecture.md) - Understand the system design
- [API Reference](api/endpoints.md) - Full endpoint documentation
