#!/bin/bash
# Native Miles RLVE test script

set -ex

export PYTHONPATH=/root/gavin/miles:/root/Megatron-LM:/root/gavin/miles/examples/RLVE:$PYTHONPATH
export DEBUG_ADVANTAGES=1

# Build runtime environment JSON for ray job submit
# DEBUG_ADVANTAGES=1 enables advantage debug logging in loss.py
RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/gavin/miles:/root/Megatron-LM:/root/gavin/miles/examples/RLVE",
    "DEBUG_ADVANTAGES": "1",
    "PYTHONUNBUFFERED": "1"
  }
}'

# Run training via ray job submit to propagate env vars to all workers
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 /root/gavin/miles/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    --hf-checkpoint /data/models/Qwen2.5-0.5B-Instruct \
    --ref-load /data/models/Qwen2.5-0.5B-Instruct_torch_dist \
    --save /tmp/test-rlve-native-v2 \
    --save-interval 1 \
    --disable-rollout-global-dataset \
    --rlve \
    --data-source-path miles.ray.rollout_data_source.RolloutDataSourceWithBuffer \
    --environment-list Multiplication Division \
    --custom-prompt-preprocessor TinyZero \
    --answer-marker-type "\<answer\>\</answer\>" \
    --rm-type rlve \
    --reward-key reward \
    --num-rollout 1 \
    --rollout-batch-size 8 \
    --n-samples-per-prompt 4 \
    --rollout-max-response-len 2048 \
    --rollout-temperature 1.0 \
    --partial-rollout \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 2 \
    --advantage-estimator grpo \
    --entropy-coef 0.00 \
    --eps-clip 0.2 \
    --use-tis \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.01 \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.7 \
    --router-disable-circuit-breaker \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    --swiglu \
    --num-layers 24 \
    --hidden-size 896 \
    --ffn-hidden-size 4864 \
    --num-attention-heads 14 \
    --group-query-attention \
    --num-query-groups 2 \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --add-qkv-bias \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --rotary-base 1000000 \
    --vocab-size 151936 \
    --bf16 \
    --train-env-vars '{"PYTHONPATH": "/root/gavin/miles:/root/Megatron-LM:/root/gavin/miles/examples/RLVE"}'
