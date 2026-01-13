#!/bin/bash
# Complete setup: download models, convert to Megatron format, download datasets
#
# Usage:
#   ./setup_all.sh                    # Full setup with defaults
#   ./setup_all.sh --models-only      # Only download/convert models
#   ./setup_all.sh --datasets-only    # Only download datasets
#   ./setup_all.sh --skip-convert     # Download but don't convert models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
MODELS_ONLY=false
DATASETS_ONLY=false
SKIP_CONVERT=false

for arg in "$@"; do
    case $arg in
        --models-only)
            MODELS_ONLY=true
            ;;
        --datasets-only)
            DATASETS_ONLY=true
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--models-only|--datasets-only|--skip-convert]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "OpenTinker-Miles Data Setup"
echo "========================================"
echo ""

# Step 1: Download models
if [ "$DATASETS_ONLY" = false ]; then
    echo "=== Step 1: Downloading HuggingFace Models ==="
    "$SCRIPT_DIR/prepare_weights.sh"
    echo ""
fi

# Step 2: Convert models
if [ "$DATASETS_ONLY" = false ] && [ "$SKIP_CONVERT" = false ]; then
    echo "=== Step 2: Converting Models to Megatron Format ==="
    "$SCRIPT_DIR/convert_weights.sh"
    echo ""
fi

# Step 3: Download datasets
if [ "$MODELS_ONLY" = false ]; then
    echo "=== Step 3: Downloading Datasets ==="
    "$SCRIPT_DIR/prepare_datasets.sh"
    echo ""
fi

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Models:   /data/models/"
echo "Datasets: /data/datasets/"
echo ""
echo "To start RLVE training:"
echo "  1. Start Ray: ray start --head"
echo "  2. Start API: cd /root/gavin/tinkercloud && python -m uvicorn training.api:app --host 0.0.0.0 --port 8000"
echo "  3. Run training: TINKER_API_KEY=slime-dev-key python -m tinker_cookbook.recipes.rlve.train ..."
