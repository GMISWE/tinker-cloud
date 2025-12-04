#!/bin/bash
# Prepare data: Download models and datasets if not already present
# This script checks for existing data to avoid redundant downloads

set -e

echo "=== Checking data availability ==="

# Check and download Qwen2.5-0.5B-Instruct model
MODEL_DIR="/data/models/Qwen2.5-0.5B-Instruct"
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✓ Model already exists: $MODEL_DIR"
else
    echo "Downloading Qwen2.5-0.5B-Instruct model..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='$MODEL_DIR')"
    echo "✓ Model downloaded: $MODEL_DIR"
fi

# Check and download GSM8K dataset
DATASET_DIR="/data/datasets/gsm8k"
if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR 2>/dev/null)" ]; then
    echo "✓ Dataset already exists: $DATASET_DIR"
else
    echo "Downloading GSM8K dataset..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('zhuzilin/gsm8k', repo_type='dataset', local_dir='$DATASET_DIR')"
    echo "✓ Dataset downloaded: $DATASET_DIR"
fi

echo "=== Data preparation complete ==="
