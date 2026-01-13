#!/bin/bash
# Convert HuggingFace models to Megatron torch_dist format
#
# Usage:
#   ./convert_weights.sh                              # Convert all HF models
#   ./convert_weights.sh Qwen2.5-7B-Instruct          # Convert specific model
#   ./convert_weights.sh /data/models/Qwen2.5-7B-Instruct  # Full path

set -e

MODEL_DIR="/data/models"
MILES_DIR="${MILES_DIR:-/root/gavin/miles}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"

# Model architecture configurations
# Format: model_pattern|script_name|extra_args
MODEL_CONFIGS=(
    # Qwen 2.5 family
    "Qwen2.5-0.5B|qwen2.5-0.5B.sh|--untie-embeddings-and-output-weights"
    "Qwen2.5-1.5B|qwen2.5-1.5B.sh|--untie-embeddings-and-output-weights"
    "Qwen2.5-3B|qwen2.5-3B.sh|--untie-embeddings-and-output-weights"
    "Qwen2.5-7B|qwen2.5-7B.sh|--untie-embeddings-and-output-weights"
    "Qwen2.5-32B|qwen2.5-32B.sh|--untie-embeddings-and-output-weights"
    # Qwen 3 family
    "Qwen3-0.6B|qwen3-0.6B.sh|"
    "Qwen3-1.7B|qwen3-1.7B.sh|"
    "Qwen3-4B|qwen3-4B.sh|"
    "Qwen3-8B|qwen3-8B.sh|"
    "Qwen3-14B|qwen3-14B.sh|"
    "Qwen3-32B|qwen3-32B.sh|"
    # DeepSeek R1 Distill (uses Qwen 1.5B architecture)
    "DeepSeek-R1-Distill-Qwen-1.5B|qwen2.5-1.5B.sh|--untie-embeddings-and-output-weights"
    # Llama family
    "Llama-3.2-1B|llama3.2-1B.sh|"
    "Llama-3.2-3B|llama3.2-3B-Instruct.sh|"
    "Llama-3.1-8B|llama3.1-8B-Instruct.sh|"
)

get_model_config() {
    local model_name="$1"
    for config in "${MODEL_CONFIGS[@]}"; do
        local pattern=$(echo "$config" | cut -d'|' -f1)
        if [[ "$model_name" == *"$pattern"* ]]; then
            echo "$config"
            return 0
        fi
    done
    echo ""
}

convert_model() {
    local hf_path="$1"
    local model_name=$(basename "$hf_path")
    local output_path="${hf_path}_torch_dist"

    # Skip if already converted
    if [ -d "$output_path" ] && [ -f "$output_path/common.pt" ]; then
        echo "✓ Already converted: $output_path"
        return 0
    fi

    # Skip if source doesn't exist
    if [ ! -d "$hf_path" ] || [ ! -f "$hf_path/config.json" ]; then
        echo "✗ HF model not found: $hf_path"
        return 1
    fi

    # Get model config
    local config=$(get_model_config "$model_name")
    if [ -z "$config" ]; then
        echo "✗ Unknown model architecture: $model_name"
        echo "  Please add configuration to MODEL_CONFIGS"
        return 1
    fi

    local script_name=$(echo "$config" | cut -d'|' -f2)
    local extra_args=$(echo "$config" | cut -d'|' -f3)
    local script_path="$MILES_DIR/scripts/models/$script_name"

    if [ ! -f "$script_path" ]; then
        echo "✗ Model script not found: $script_path"
        return 1
    fi

    echo "Converting $model_name..."
    echo "  Source: $hf_path"
    echo "  Target: $output_path"
    echo "  Config: $script_name"

    # Source the model args
    source "$script_path"

    # Run conversion
    PYTHONPATH="$MEGATRON_DIR:$MILES_DIR:$PYTHONPATH" \
    python3 "$MILES_DIR/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        $extra_args \
        --hf-checkpoint "$hf_path" \
        --save "$output_path"

    echo "✓ Converted: $output_path"
}

# Main
if [ $# -gt 0 ]; then
    # Convert specific model(s)
    for model in "$@"; do
        if [[ "$model" == /* ]]; then
            # Full path provided
            convert_model "$model"
        else
            # Model name provided
            convert_model "$MODEL_DIR/$model"
        fi
    done
else
    # Convert all unconverted HF models
    echo "=== Converting all HF models to torch_dist ==="
    for dir in "$MODEL_DIR"/*/; do
        model_name=$(basename "$dir")
        # Skip already converted models
        if [[ "$model_name" == *"_torch_dist" ]]; then
            continue
        fi
        # Skip if no config.json (not a valid HF model)
        if [ ! -f "$dir/config.json" ]; then
            continue
        fi
        convert_model "${dir%/}"
    done
fi

echo ""
echo "=== Converted models ==="
ls -la "$MODEL_DIR"/*_torch_dist 2>/dev/null || echo "No converted models found"
