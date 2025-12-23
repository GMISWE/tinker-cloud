#!/bin/bash
# Test DPO with REDUCED max_length to avoid OOM
# Runs directly in docker container (not via kubectl)
#
# Usage: ./test_dpo_reduced.sh [num_steps]
#
# Prerequisites:
#   - opentinker-miles server running on localhost:8000
#   - Model available at /data/models/Qwen2.5-0.5B-Instruct
#   - tinker-cookbook installed

set -e

NUM_STEPS=${1:-3}
MODEL_PATH="/data/models/Qwen2.5-0.5B-Instruct"
LOG_PATH="/tmp/test-dpo-reduced"

echo "=========================================="
echo "DPO Reduced Test"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Steps: $NUM_STEPS"
echo "Log path: $LOG_PATH"
echo "=========================================="

# Check model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download with: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir $MODEL_PATH"
    exit 1
fi

# Check server is running
if ! curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | grep -q "healthy"; then
    echo "ERROR: opentinker-miles server not running on localhost:8000"
    echo "Start with: cd /root/gavin/opentinker-miles && python -m uvicorn training.api:app --host 0.0.0.0 --port 8000"
    exit 1
fi

# Run cleanup first
echo "Running cleanup..."
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
    python /root/gavin/opentinker-miles/tests/cleanup_test_env.py 2>/dev/null || true

# Run DPO training
echo "Starting DPO training..."
TINKER_API_KEY=slime-dev-key \
HF_DATASETS_OFFLINE=0 \
HF_HUB_OFFLINE=0 \
HF_DATASETS_CACHE=/data/datasets \
HF_HOME=/data \
PYTHONUNBUFFERED=1 \
python -m tinker_cookbook.recipes.preference.dpo.train \
    model_name="$MODEL_PATH" \
    renderer_name=qwen3 \
    base_url=http://localhost:8000 \
    dataset=hhh \
    batch_size=4 \
    learning_rate=1e-5 \
    dpo_beta=0.1 \
    max_length=128 \
    n_batches="$NUM_STEPS" \
    log_path="$LOG_PATH" \
    behavior_if_log_dir_exists=delete

echo "=========================================="
echo "DPO test completed!"
echo "Logs: $LOG_PATH"
echo "=========================================="
