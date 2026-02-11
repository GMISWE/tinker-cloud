#!/bin/bash
# Build development Docker image with local code
#
# This script creates a temporary build context with all required repos,
# then builds the appropriate Dockerfile.dev image for the selected backend.
#
# Usage:
#   cd tinkercloud
#   ./docker/build_dev.sh [--backend miles|nemo_rl] [IMAGE_TAG]
#
# Examples:
#   ./docker/build_dev.sh                                        # Miles (default)
#   ./docker/build_dev.sh --backend miles                        # Miles (explicit)
#   ./docker/build_dev.sh --backend nemo_rl                      # NeMo RL
#   ./docker/build_dev.sh --backend nemo_rl my-image:tag         # NeMo RL with custom tag

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINKERCLOUD_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$TINKERCLOUD_DIR")"

# Parse arguments
BACKEND="miles"
IMAGE_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "ERROR: --backend requires a value (miles or nemo_rl)"
                exit 1
            fi
            BACKEND="$2"
            shift 2
            ;;
        --*)
            echo "ERROR: Unknown flag: $1"
            echo "Usage: $0 [--backend miles|nemo_rl] [IMAGE_TAG]"
            exit 1
            ;;
        *)
            IMAGE_TAG="$1"
            shift
            ;;
    esac
done

# Validate backend
if [[ "$BACKEND" != "miles" && "$BACKEND" != "nemo_rl" ]]; then
    echo "ERROR: Unsupported backend: $BACKEND"
    echo "Supported backends: miles, nemo_rl"
    exit 1
fi

# Set defaults based on backend
if [[ "$BACKEND" == "miles" ]]; then
    IMAGE_TAG="${IMAGE_TAG:-gmicloudai/tinkercloud:dev-local}"
    DOCKERFILE="docker/Dockerfile.dev"
else
    IMAGE_TAG="${IMAGE_TAG:-gmicloudai/tinkercloud:dev-nemo-rl}"
    DOCKERFILE="docker/Dockerfile.dev.nemo_rl"
fi

echo "=== Building development image: $IMAGE_TAG ==="
echo "Backend: $BACKEND"
echo "Dockerfile: $DOCKERFILE"
echo "Source directories:"
echo "  tinkercloud: $TINKERCLOUD_DIR"
if [[ "$BACKEND" == "miles" ]]; then
    echo "  miles: $PARENT_DIR/miles"
elif [[ "$BACKEND" == "nemo_rl" ]]; then
    echo "  RL: $PARENT_DIR/RL"
fi
echo "  tinker_gmi: $PARENT_DIR/tinker_gmi"
echo "  tinker-cookbook: $PARENT_DIR/tinker-cookbook"

# Verify source directories exist
REQUIRED_DIRS=("$PARENT_DIR/tinker_gmi" "$PARENT_DIR/tinker-cookbook")
if [[ "$BACKEND" == "miles" ]]; then
    REQUIRED_DIRS+=("$PARENT_DIR/miles")
elif [[ "$BACKEND" == "nemo_rl" ]]; then
    REQUIRED_DIRS+=("$PARENT_DIR/RL")
fi

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Directory not found: $dir"
        exit 1
    fi
done

# Create temporary build context
BUILD_CONTEXT=$(mktemp -d)
trap "rm -rf $BUILD_CONTEXT" EXIT

echo ""
echo "=== Creating build context at $BUILD_CONTEXT ==="

# Copy tinkercloud (excluding .git, __pycache__, etc.)
RSYNC_EXCLUDES="--exclude=.git --exclude=__pycache__ --exclude=*.pyc --exclude=.pytest_cache"

echo "Copying tinkercloud..."
rsync -a $RSYNC_EXCLUDES "$TINKERCLOUD_DIR/" "$BUILD_CONTEXT/"

# Copy backend-specific repos
if [[ "$BACKEND" == "miles" ]]; then
    echo "Copying miles..."
    rsync -a $RSYNC_EXCLUDES "$PARENT_DIR/miles/" "$BUILD_CONTEXT/miles/"
elif [[ "$BACKEND" == "nemo_rl" ]]; then
    echo "Copying RL (NeMo RL)..."
    rsync -a $RSYNC_EXCLUDES "$PARENT_DIR/RL/" "$BUILD_CONTEXT/RL/"
fi

# Copy tinker_gmi
echo "Copying tinker_gmi..."
rsync -a $RSYNC_EXCLUDES "$PARENT_DIR/tinker_gmi/" "$BUILD_CONTEXT/tinker_gmi/"

# Copy tinker-cookbook
echo "Copying tinker-cookbook..."
rsync -a $RSYNC_EXCLUDES "$PARENT_DIR/tinker-cookbook/" "$BUILD_CONTEXT/tinker-cookbook/"

echo ""
echo "=== Build context contents ==="
ls -la "$BUILD_CONTEXT/"

echo ""
echo "=== Building Docker image ==="
docker build -t "$IMAGE_TAG" -f "$BUILD_CONTEXT/$DOCKERFILE" "$BUILD_CONTEXT"

echo ""
echo "=== Build complete: $IMAGE_TAG ==="
echo ""
echo "To run:"
if [[ "$BACKEND" == "miles" ]]; then
    echo "  docker run -d --name tinkercloud-dev --gpus all -v /data:/data --network host --shm-size=16g -e ALLOW_PARTIAL_BATCHES=true $IMAGE_TAG"
else
    echo "  docker run -d --name tinkercloud-nemo-rl --gpus all -v /data:/data --network host --shm-size=16g $IMAGE_TAG"
fi
