#!/bin/bash
# Convert HuggingFace model to Megatron torch_dist format
# This script should be run at container runtime when GPU is available

set -e

MODEL_DIR="/data/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="/data/models/Qwen2.5-0.5B-Instruct_torch_dist"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Converting HF model to Megatron format..."
    echo "Source: $MODEL_DIR"
    echo "Target: $OUTPUT_DIR"

    cd /root/miles
    source scripts/models/qwen2.5-0.5B.sh

    PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "$MODEL_DIR" \
        --save "$OUTPUT_DIR"

    echo "Model conversion completed successfully!"
else
    echo "Model already converted at $OUTPUT_DIR, skipping conversion..."
fi