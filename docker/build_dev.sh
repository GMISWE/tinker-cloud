#!/bin/bash
# Build development Docker image with local code
#
# This script creates a temporary build context with all required repos,
# then builds the Dockerfile.dev image.
#
# Usage:
#   cd /root/.work/gavin/tinkercloud
#   ./docker/build_dev.sh [IMAGE_TAG]
#
# Example:
#   ./docker/build_dev.sh gmicloudai/tinkercloud:dev-local

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINKERCLOUD_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$TINKERCLOUD_DIR")"

IMAGE_TAG="${1:-gmicloudai/tinkercloud:dev-local}"

echo "=== Building development image: $IMAGE_TAG ==="
echo "Source directories:"
echo "  tinkercloud: $TINKERCLOUD_DIR"
echo "  miles: $PARENT_DIR/miles"
echo "  tinker_gmi: $PARENT_DIR/tinker_gmi"
echo "  tinker-cookbook: $PARENT_DIR/tinker-cookbook"

# Verify source directories exist
for dir in "$PARENT_DIR/miles" "$PARENT_DIR/tinker_gmi" "$PARENT_DIR/tinker-cookbook"; do
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
echo "Copying tinkercloud..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    "$TINKERCLOUD_DIR/" "$BUILD_CONTEXT/"

# Copy miles
echo "Copying miles..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    "$PARENT_DIR/miles/" "$BUILD_CONTEXT/miles/"

# Copy tinker_gmi
echo "Copying tinker_gmi..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    "$PARENT_DIR/tinker_gmi/" "$BUILD_CONTEXT/tinker_gmi/"

# Copy tinker-cookbook
echo "Copying tinker-cookbook..."
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    "$PARENT_DIR/tinker-cookbook/" "$BUILD_CONTEXT/tinker-cookbook/"

echo ""
echo "=== Build context contents ==="
ls -la "$BUILD_CONTEXT/"

echo ""
echo "=== Building Docker image ==="
docker build -t "$IMAGE_TAG" -f "$BUILD_CONTEXT/docker/Dockerfile.dev" "$BUILD_CONTEXT"

echo ""
echo "=== Build complete: $IMAGE_TAG ==="
echo ""
echo "To run:"
echo "  docker run -d --name tinkercloud-dev --gpus all -v /data:/data --network host --shm-size=16g -e ALLOW_PARTIAL_BATCHES=true $IMAGE_TAG"
