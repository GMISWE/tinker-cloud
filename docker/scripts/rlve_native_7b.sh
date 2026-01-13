#!/bin/bash
# RLVE Training Script for Qwen2.5-7B on Miles deployment
# Adapted for 4× H200 GPUs (miles-gmi-tinker namespace)
#
# Original: 8× H100 with TP=2, CP=4
# Adapted:  4× H200 with TP=2, CP=2 (context parallel for long sequences)
#
# Prerequisites:
#   Run ./prepare-model.sh first to download and convert the model
#
# Requirements:
# - Miles pod with Gym environments at /root/miles/Gym
# - Megatron checkpoint at /data/models/Qwen2.5-7B-Instruct_torch_dist
# - RLVE patches applied to Miles code

# Default: 16 environments from the original RLVE paper
DEFAULT_ENVIRONMENTS="Division EuclidGame GCDOne_Counting HamiltonianPath LampChanging LargestConvexPolygon Multiplication PCPPermutation Path_NoGoingBack_Counting SAT ShortestPath Sorting SpiralMatrix SubsequenceReversalLNDS UndamagedSubmatrixCounting WYRLevelingGround"

ENVIRONMENT_LIST=${1:-"$DEFAULT_ENVIRONMENTS"}
SAVE_PATH=${2:-"/data/checkpoints/rlve-7b"}

# Check WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Set it with: export WANDB_API_KEY=your_key"
fi

set -ex

# Configuration for Miles deployment
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"
MEGATRON_CKPT="/data/models/Qwen2.5-7B-Instruct_torch_dist"
NUM_GPUS=4

# Environment setup
export PYTHONPATH=/root/miles:/root/Megatron-LM:/root/miles/examples/RLVE:$PYTHONPATH
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Kill any existing processes and restart Ray
ray stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
sleep 2
ray start --head --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
sleep 3

# Model architecture for Qwen2.5-7B-Instruct (sourced from miles repo)
source /root/miles/scripts/models/qwen2.5-7B.sh

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_PATH}
   --ref-load ${MEGATRON_CKPT}
   --load ${SAVE_PATH}
   --save ${SAVE_PATH}
   --save-interval 100
)

# Rollout config - matching original 8× H100 config
# Original (8× H100): batch=128, n_samples=16, over_sampling=384
# H200 has significantly more memory per GPU (~141GB vs ~80GB H100)
ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rlve
   --data-source-path miles.ray.rollout_data_source.RolloutDataSourceWithBuffer
   --environment-list ${ENVIRONMENT_LIST}

   --custom-prompt-preprocessor TinyZero
   --answer-marker-type "\<answer\>\</answer\>"

   --rm-type rlve
   --reward-key reward

   --num-rollout 500
   --rollout-batch-size 128
   --n-samples-per-prompt 16
   --rollout-max-response-len 4096
   --rollout-temperature 1.0

   # Partial rollout - matching original H100 config
   --over-sampling-batch-size 384
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --partial-rollout

   --num-steps-per-rollout 1
   --balance-data
)

# Parallelism for 4× H200
# Using TP=2, CP=2 similar to GB200 config
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # Dynamic batch size disabled - causes tensor shape mismatch with CP=2
   # --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --router-disable-circuit-breaker
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Wandb logging (online mode)
WANDB_ARGS=(
   --use-wandb
   --wandb-mode online
   --wandb-project "rlve-qwen"
   --wandb-group "7b-4xH200"
   --wandb-dir "/data/wandb"
   --wandb-key "${WANDB_API_KEY}"
)

# Change to the RLVE examples directory where Gym/ is located
# This allows Python to import Gym from the current working directory
cd /root/miles/examples/RLVE

# Build the runtime environment JSON for ray job submit
# Include the RLVE directory in PYTHONPATH so Gym can be imported
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/miles:/root/Megatron-LM:/root/miles/examples/RLVE\"
  }
}"

# Run training via ray job submit from the RLVE directory
# The Gym/ directory in this location will be importable
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /root/miles/train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]}
