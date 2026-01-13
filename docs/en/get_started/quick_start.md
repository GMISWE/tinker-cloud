# Quick Start

This guide walks you through setting up a complete tinkercloud test environment, from Docker setup to running your first training.

---

## 1. Docker Environment Setup

Since the software dependencies are complex when tinker/miles/tinkercloud are combined, we strongly recommend using our pre-configured Docker image.

### Pull and Start Docker Container

```bash
# Pull the latest image
docker pull gmicloudai/tinkercloud:dev

# Start the container with GPU access
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data:/data \
  -it gmicloudai/tinkercloud:dev /bin/bash
```

### Create Required Directories

Inside the container:

```bash
mkdir -p /data/logs
mkdir -p /data/cache
mkdir -p /data/checkpoints
mkdir -p /data/metadata
mkdir -p /data/models
mkdir -p /data/datasets
mkdir -p /data/wandb
```

---

## 2. Install Components

Install all required components in editable mode:

```bash
# Install tinkercloud (middleware server)
cd /app && pip install -e . -q

# Install Miles (training backend)
cd /root/miles && pip install -e . -q

# Install tinker_gmi (Python client SDK)
cd /workspace/tinker_gmi && pip install -e . -q

# Install tinker-cookbook (training recipes) - optional
cd /workspace/tinker-cookbook && pip install -e . -q
```

### Set Environment Variables

```bash
# API key for authentication
export TINKER_API_KEY=slime-dev-key

# Wandb logging (optional)
export WANDB_API_KEY=your-wandb-key

# Python path for Miles/Megatron
export PYTHONPATH=/root/Megatron-LM:/root/miles:$PYTHONPATH
```

---

## 3. Start Ray Cluster

Ray is required for distributed training. Check if Ray is running:

```bash
ray status
```

If not running, start the Ray head node:

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

Verify Ray is running:

```bash
ray status
```

**Important:** Don't stop Ray manually - the pod liveness probe depends on Ray dashboard (port 8265).

---

## 4. Start tinkercloud Server

### Option A: Start Server

```bash
# Kill any existing server
pkill -9 -f "uvicorn.*training" || true

# Clear Python cache
find /app -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Start tinkercloud server
cd /app
ALLOW_PARTIAL_BATCHES=true \
  PYTHONPATH=/app:/root/Megatron-LM:/root/miles:$PYTHONPATH \
  nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 \
  >> /data/logs/tinkercloud.log 2>&1 &

# Wait for startup
sleep 5
```

### Verify Server Health

```bash
curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python3 -m json.tool
```

Expected response:
```json
{
    "status": "healthy",
    "version": "3.1.0",
    "ray_initialized": true,
    "active_training_clients": 0
}
```

---

## 5. Run Test Training

### Cleanup Existing Sessions

Before starting new training, cleanup any existing sessions to free GPUs:

```bash
TINKER_BASE_URL=http://localhost:8000 \
  TINKER_API_KEY=slime-dev-key \
  python /workspace/tinker_gmi/tests_integration/cleanup_test_env.py
```

### Quick SFT Test

```python
import asyncio
import tinker

async def test_training():
    # Create service client
    client = tinker.ServiceClient(base_url="http://localhost:8000")

    # Create training client (this allocates GPUs and loads model)
    training_client = await client.create_lora_training_client_async(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        rank=32
    )

    # Get tokenizer
    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded: {tokenizer}")

    # Cleanup
    await training_client.close()
    print("Test completed successfully!")

asyncio.run(test_training())
```

---

## 6. RLVE Training Setup

RLVE (Reinforcement Learning with Verifiable Environments) uses procedurally generated math/logic problems for training.

### Available Models

| Model | Path | Size |
|-------|------|------|
| Qwen2.5-0.5B-Instruct | `/data/models/Qwen2.5-0.5B-Instruct_torch_dist` | 943M |
| Qwen2.5-7B-Instruct | `/data/models/Qwen2.5-7B-Instruct_torch_dist` | 15G |
| Qwen3-4B-Instruct-2507 | `/data/models/Qwen3-4B-Instruct-2507_torch_dist` | 7.5G |

### Available Environments

Default RLVE environments: `Division`, `EuclidGame`, `Multiplication`, `Sorting`, `GCDOne_Counting`, `HamiltonianPath`, and more (400+ total).

### Run RLVE Training

```bash
cd /workspace/tinker-cookbook
export TINKER_API_KEY=slime-dev-key
export WANDB_API_KEY=your-wandb-key

python -m tinker_cookbook.recipes.rlve.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    base_url=http://localhost:8000 \
    environment_list=Division,EuclidGame,Multiplication,Sorting \
    groups_per_batch=64 \
    group_size=16 \
    max_tokens=4096 \
    learning_rate=1e-6 \
    n_batches=100 \
    wandb_project=rlve-test \
    log_path=/data/logs/rlve-test
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `groups_per_batch` | Number of problem groups per batch | 32 |
| `group_size` | Samples per group (n_samples_per_prompt) | 8 |
| `max_tokens` | Max response tokens | 4096 |
| `learning_rate` | Learning rate | 1e-6 |
| `n_batches` | Total training batches | 1000000 |

---

## 7. Troubleshooting

### GPU Cleanup

If GPUs are stuck with old processes:

```bash
# Kill GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Or restart Ray (clears all actors)
ray stop --force && sleep 3 && \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# IMPORTANT: Must also restart tinkercloud server after Ray restart!
```

### Session Cleanup

```bash
TINKER_BASE_URL=http://localhost:8000 \
  TINKER_API_KEY=slime-dev-key \
  python /workspace/tinker_gmi/tests_integration/cleanup_test_env.py
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Session not found` | Server restarted | Create new training client |
| `Model creation timed out` | GPU unavailable | Check `nvidia-smi`, cleanup sessions |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, cleanup sessions |
| `Ray connection failed` | Ray not running | Start Ray with `ray start --head` |

### View Logs

```bash
# Server logs
tail -f /data/logs/tinkercloud.log

# Training logs
tail -f /data/logs/tinker-rlve.log

# Search for errors
grep -E "(ERROR|Exception|Traceback)" /data/logs/tinkercloud.log | tail -20
```

---

## 8. Quick Commands Reference

### Ray Management

```bash
# Check Ray status
ray status

# Restart Ray (clears all actors and GPU allocations)
ray stop --force && sleep 3 && \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

### Server Management

```bash
# Kill server
pkill -9 -f "uvicorn.*training" || true

# Start server
cd /app
ALLOW_PARTIAL_BATCHES=true \
  PYTHONPATH=/app:/root/Megatron-LM:/root/miles:$PYTHONPATH \
  nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 \
  >> /data/logs/tinkercloud.log 2>&1 &

# Check health
curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python3 -m json.tool
```

### GPU Monitoring

```bash
# Check GPU usage
nvidia-smi

# Watch GPU usage continuously
watch -n 1 nvidia-smi

# List GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

### Reinstall Components

After code changes:

```bash
cd /app && pip install -e . -q
cd /root/miles && pip install -e . -q
cd /workspace/tinker_gmi && pip install -e . -q
cd /workspace/tinker-cookbook && pip install -e . -q
```

---

## 9. Standalone Docker Deployment

Run tinkercloud as a standalone container with automatic model setup.

### Start Container

```bash
# Start the container with GPU access (runs entrypoint which starts Ray + API)
docker run -d --name tinkercloud-rlve \
  --gpus all \
  -v /data:/data \
  --network host \
  --shm-size=16g \
  gmicloudai/tinkercloud:latest

# Check startup logs
docker logs -f tinkercloud-rlve
```

The entrypoint automatically:
1. Creates required directories (`/data/models`, `/data/checkpoints`, etc.)
2. Downloads the Qwen2.5-0.5B-Instruct model
3. Starts Ray head node
4. Starts the tinkercloud API server

### Convert Model to Megatron Format

The HF model must be converted to Megatron format for training:

```bash
docker exec tinkercloud-rlve bash -c '
cd /root/miles
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/Qwen2.5-0.5B-Instruct \
  --save /data/models/Qwen2.5-0.5B-Instruct_torch_dist
'
```

### Verify API Health

```bash
curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python3 -m json.tool
```

### Run RLVE Training

```bash
docker exec tinkercloud-rlve bash -c '
cd /workspace/tinker-cookbook
TINKER_API_KEY=slime-dev-key python -m tinker_cookbook.recipes.rlve.train \
    model_name=Qwen/Qwen2.5-0.5B-Instruct \
    base_url=http://localhost:8000 \
    groups_per_batch=4 \
    group_size=4 \
    max_tokens=2048 \
    learning_rate=1e-6 \
    n_batches=10 \
    eval_every=0 \
    save_every=0 \
    behavior_if_log_dir_exists=delete
'
```

### Stop Container

```bash
docker stop tinkercloud-rlve
docker rm tinkercloud-rlve
```

---

## Next Steps

- [Architecture](../architecture.md) - Understand the system design
- [API Reference](../api/endpoints.md) - Full endpoint documentation
- [Quickstart Examples](../quickstart.md) - Code examples for training
