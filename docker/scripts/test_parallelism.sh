#!/bin/bash
# Test different parallelism configurations with math_rl training
# Reports metrics to wandb for comparison

set -e

# Configuration
MODEL_NAME="meta-llama/Llama-3.2-1B"
GROUP_SIZE=4
GROUPS_PER_BATCH=100
LEARNING_RATE=1e-4
WANDB_PROJECT="parallelism-test"
N_BATCHES=10

# API settings
export TINKER_API_KEY=slime-dev-key
export TINKER_BASE_URL=http://localhost:8000
# WANDB_API_KEY should be set in environment before running
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Set it with: export WANDB_API_KEY=your_key"
fi
export PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/gavin/miles:$PYTHONPATH

# Function to restart server with new parallelism
restart_server() {
    local tp=$1
    local cp=$2

    echo "=========================================="
    echo "Restarting server with TP=$tp, CP=$cp"
    echo "=========================================="

    # Kill existing server
    pkill -9 -f uvicorn 2>/dev/null || true
    sleep 3

    # Clear log
    > /data/logs/tinkercloud.log

    # Start server with new parallelism
    SLIME_DEFAULT_TP=$tp SLIME_DEFAULT_CP=$cp ALLOW_PARTIAL_BATCHES=true \
        nohup python3 -m uvicorn training.api:app --host 0.0.0.0 --port 8000 \
        >> /data/logs/tinkercloud.log 2>&1 &

    echo "Waiting for server to start..."
    sleep 15

    # Verify server is healthy
    curl -s http://localhost:8000/health -H "X-API-Key: slime-dev-key" | grep -q "healthy" || {
        echo "ERROR: Server failed to start"
        tail -50 /data/logs/tinkercloud.log
        exit 1
    }
    echo "Server is healthy"

    # Cleanup any existing training sessions
    echo "Cleaning up existing sessions..."
    python /root/gavin/tinker_gmi/tests_integration/cleanup_test_env.py 2>/dev/null || true
}

# Function to run training
run_training() {
    local config_name=$1
    local tp=$2
    local cp=$3
    local dp=$4

    local log_path="/tmp/parallelism_test/${config_name}"
    local wandb_name="${config_name}-$(date +%Y%m%d-%H%M%S)"

    echo "=========================================="
    echo "Running training: $config_name ($N_BATCHES batches)"
    echo "  TP=$tp, CP=$cp, DP=$dp"
    echo "  Log path: $log_path"
    echo "  Wandb name: $wandb_name"
    echo "=========================================="

    # Remove old log dir
    rm -rf "$log_path" 2>/dev/null || true

    # Run training
    python -m tinker_cookbook.recipes.math_rl.train \
        model_name="$MODEL_NAME" \
        group_size=$GROUP_SIZE \
        groups_per_batch=$GROUPS_PER_BATCH \
        learning_rate=$LEARNING_RATE \
        n_batches=$N_BATCHES \
        log_path="$log_path" \
        wandb_project="$WANDB_PROJECT" \
        wandb_name="$wandb_name" \
        behavior_if_log_dir_exists=delete

    # Check parallelism used
    echo ""
    echo "Parallelism used:"
    grep -E "(TP=|parallelism)" /data/logs/tinkercloud.log | tail -3

    # Print final metrics
    echo ""
    echo "Final metrics for $config_name:"
    if [ -f "$log_path/metrics.jsonl" ]; then
        tail -3 "$log_path/metrics.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    step = d.get('progress/batch', '-')
    correct = d.get('env/all/correct', 0)
    reward = d.get('env/all/reward/total', 0)
    train_time = d.get('time/train', 0)
    print(f'  Step {step}: correct={correct:.3f}, reward={reward:.3f}, train={train_time:.1f}s')
"
    else
        echo "  No metrics file found"
    fi
    echo ""
}

# Main execution
echo "============================================"
echo "Parallelism Configuration Test"
echo "============================================"
echo "Model: $MODEL_NAME"
echo "Wandb Project: $WANDB_PROJECT"
echo "Batches per config: $N_BATCHES"
echo "Configs to test:"
echo "  1. dp4-tp1-cp1 (baseline)"
echo "  2. dp2-tp2-cp1 (tensor parallel)"
echo "  3. dp1-tp2-cp2 (tensor + context parallel)"
echo "============================================"
echo ""

# Create output directory
mkdir -p /tmp/parallelism_test

# Config 1: DP=4, TP=1, CP=1 (baseline)
echo ""
echo ">>> CONFIG 1: dp4-tp1-cp1 <<<"
restart_server 1 1
run_training "dp4-tp1-cp1" 1 1 4

# Config 2: DP=2, TP=2, CP=1
echo ""
echo ">>> CONFIG 2: dp2-tp2-cp1 <<<"
restart_server 2 1
run_training "dp2-tp2-cp1" 2 1 2

# Config 3: DP=1, TP=2, CP=2
echo ""
echo ">>> CONFIG 3: dp1-tp2-cp2 <<<"
restart_server 2 2
run_training "dp1-tp2-cp2" 2 2 1

# Summary
echo ""
echo "============================================"
echo "TEST COMPLETE - Summary"
echo "============================================"
echo ""
echo "Results saved to /tmp/parallelism_test/"
echo "Wandb project: $WANDB_PROJECT"
echo ""

# Print comparison table
echo "Final comparison:"
echo ""
printf "%-15s %8s %8s %10s\n" "Config" "Correct" "Reward" "TrainTime"
printf "%-15s %8s %8s %10s\n" "------" "-------" "------" "---------"
for config in dp4-tp1-cp1 dp2-tp2-cp1 dp1-tp2-cp2; do
    if [ -f "/tmp/parallelism_test/${config}/metrics.jsonl" ]; then
        tail -1 "/tmp/parallelism_test/${config}/metrics.jsonl" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
correct = d.get('env/all/correct', 0)
reward = d.get('env/all/reward/total', 0)
train_time = d.get('time/train', 0)
print(f'$config {correct:8.3f} {reward:8.3f} {train_time:10.1f}s')
"
    else
        printf "%-15s %8s %8s %10s\n" "$config" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "Done!"
