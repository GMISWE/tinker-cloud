#!/bin/bash
# Download HuggingFace models to /data/models/
#
# Usage:
#   ./prepare_weights.sh                     # Download all default models
#   ./prepare_weights.sh Qwen/Qwen2.5-7B-Instruct  # Download specific model

set -e

MODEL_DIR="/data/models"
mkdir -p "$MODEL_DIR"

# Default models for RLVE training
DEFAULT_MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "meta-llama/Llama-3.2-1B"
    "Qwen/Qwen3-8B-Base"
)

download_model() {
    local model_id="$1"
    local model_name=$(basename "$model_id")
    local target_dir="$MODEL_DIR/$model_name"

    if [ -d "$target_dir" ] && [ -f "$target_dir/config.json" ]; then
        echo "✓ Model already exists: $target_dir"
        return 0
    fi

    echo "Downloading $model_id to $target_dir..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model_id', local_dir='$target_dir')
print('✓ Downloaded: $target_dir')
"
}

if [ $# -gt 0 ]; then
    # Download specific model(s)
    for model in "$@"; do
        download_model "$model"
    done
else
    # Download all default models
    echo "=== Downloading default RLVE models ==="
    for model in "${DEFAULT_MODELS[@]}"; do
        download_model "$model"
    done
fi

echo ""
echo "=== Available models ==="
ls -la "$MODEL_DIR"
