#!/bin/bash
# Download datasets for training
#
# Usage:
#   ./prepare_datasets.sh                    # Download all default datasets
#   ./prepare_datasets.sh gsm8k              # Download specific dataset

set -e

DATASET_DIR="/data/datasets"
mkdir -p "$DATASET_DIR"

# Dataset configurations
# Format: name|hf_repo|repo_type
DATASETS=(
    "gsm8k|openai/gsm8k|dataset"
    "math|hendrycks/competition_math|dataset"
    "metamath|meta-math/MetaMathQA|dataset"
    "openmath|nvidia/OpenMathInstruct-2|dataset"
)

download_dataset() {
    local name="$1"
    local hf_repo="$2"
    local repo_type="${3:-dataset}"
    local target_dir="$DATASET_DIR/$name"

    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir 2>/dev/null)" ]; then
        echo "✓ Dataset already exists: $target_dir"
        return 0
    fi

    echo "Downloading $name from $hf_repo..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$hf_repo', repo_type='$repo_type', local_dir='$target_dir')
print('✓ Downloaded: $target_dir')
"
}

get_dataset_config() {
    local name="$1"
    for config in "${DATASETS[@]}"; do
        local ds_name=$(echo "$config" | cut -d'|' -f1)
        if [ "$ds_name" = "$name" ]; then
            echo "$config"
            return 0
        fi
    done
    echo ""
}

if [ $# -gt 0 ]; then
    # Download specific dataset(s)
    for name in "$@"; do
        config=$(get_dataset_config "$name")
        if [ -z "$config" ]; then
            echo "✗ Unknown dataset: $name"
            echo "  Available: ${DATASETS[*]}"
            continue
        fi
        hf_repo=$(echo "$config" | cut -d'|' -f2)
        repo_type=$(echo "$config" | cut -d'|' -f3)
        download_dataset "$name" "$hf_repo" "$repo_type"
    done
else
    # Download all default datasets
    echo "=== Downloading default datasets ==="
    for config in "${DATASETS[@]}"; do
        name=$(echo "$config" | cut -d'|' -f1)
        hf_repo=$(echo "$config" | cut -d'|' -f2)
        repo_type=$(echo "$config" | cut -d'|' -f3)
        download_dataset "$name" "$hf_repo" "$repo_type"
    done
fi

echo ""
echo "=== Available datasets ==="
ls -la "$DATASET_DIR"
