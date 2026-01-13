#!/bin/bash
# RLVE Training via tinker-cookbook (HTTP API mode)
#
# Prerequisites:
#   1. Start Ray: ray start --head
#   2. Start tinkercloud server:
#      cd /root/gavin/tinkercloud && ALLOW_PARTIAL_BATCHES=true \
#        PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/gavin/miles:$PYTHONPATH \
#        python -m uvicorn training.api:app --host 0.0.0.0 --port 8000
#
# Usage:
#   ./rlve_tinker.sh                          # Default: Qwen2.5-0.5B-Instruct
#   ./rlve_tinker.sh Qwen/Qwen2.5-7B-Instruct # Specify model

set -e

MODEL_NAME="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
TINKER_API_KEY="${TINKER_API_KEY:-slime-dev-key}"

# Training parameters
GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-4}"
GROUP_SIZE="${GROUP_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
N_BATCHES="${N_BATCHES:-100}"

echo "=== RLVE Training via tinker-cookbook ==="
echo "Model: $MODEL_NAME"
echo "Base URL: $BASE_URL"
echo "Batch: ${GROUPS_PER_BATCH} groups × ${GROUP_SIZE} samples = $((GROUPS_PER_BATCH * GROUP_SIZE)) samples"
echo ""

# Check if server is running
if ! curl -s "$BASE_URL/health" -H "X-API-Key: $TINKER_API_KEY" > /dev/null 2>&1; then
    echo "ERROR: tinkercloud server not responding at $BASE_URL"
    echo "Please start the server first (see Prerequisites above)"
    exit 1
fi

export TINKER_API_KEY
export PYTHONPATH=/root/gavin/tinker-cookbook:/root/gavin/tinker_gmi/src:$PYTHONPATH

python -m tinker_cookbook.recipes.rlve.train \
    model_name="$MODEL_NAME" \
    base_url="$BASE_URL" \
    groups_per_batch="$GROUPS_PER_BATCH" \
    group_size="$GROUP_SIZE" \
    max_tokens="$MAX_TOKENS" \
    learning_rate="$LEARNING_RATE" \
    log_path="/data/logs/rlve-tinker" \
    wandb_project="rlve-tinker" \
    behavior_if_log_dir_exists=delete \
    eval_every=0 \
    save_every=0 \
    n_batches="$N_BATCHES" 2>&1 | tee /data/logs/tinker-rlve.log
