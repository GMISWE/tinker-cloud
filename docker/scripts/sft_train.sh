#!/bin/bash
set -ex

echo "=== 1. Kill GPU processes ==="
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 2
nvidia-smi | grep -E "MiB|%"

echo "=== 2. Restart Ray ==="
ray stop --force 2>/dev/null || true
sleep 3
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "=== 3. Restart tinkercloud ==="
pkill -9 -f "uvicorn.*training" 2>/dev/null || true
sleep 2
cd /root/gavin/tinkercloud
ALLOW_PARTIAL_BATCHES=true \
PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/miles:$PYTHONPATH \
nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 >> /data/logs/tinkercloud.log 2>&1 &
echo "Waiting for server to start..."
sleep 10

# Wait for server to be healthy
for i in {1..30}; do
  if curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | grep -q "healthy"; then
    echo "Server is healthy"
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 2
done

curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | python3 -m json.tool

echo "=== 4. Run cleanup ==="
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
python3 /root/gavin/tinker_gmi/tests_integration/cleanup_test_env.py 2>/dev/null || true

echo "=== 5. Check Wandb API key ==="
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Set it with: export WANDB_API_KEY=your_key"
fi

echo "=== 6. Run SFT on NoRobots ==="
cd /root/gavin/tinker-cookbook
TINKER_API_KEY=slime-dev-key \
TINKER_BASE_URL=http://localhost:8000 \
python3 -m tinker_cookbook.recipes.chat_sl.train \
  model_name=Qwen/Qwen3-8B-Base \
  dataset=no_robots \
  learning_rate=5e-4 \
  batch_size=64 \
  lora_rank=64 \
  eval_every=20 \
  save_every=20 \
  wandb_project=cookbook_sl \
  log_path=/tmp/sft-chat-sl \
  behavior_if_log_dir_exists=delete \
  2>&1 | tee /data/logs/sft-chat-sl.log
