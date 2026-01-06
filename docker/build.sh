#!/bin/bash
# Build and push TinkerCloud all-in-one image
#
# Usage:
#   ./docker/build.sh          # Build only
#   ./docker/build.sh --push   # Build and push to registry

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

REGISTRY="us-west1-docker.pkg.dev/devv-404803/gmi-test-repo"
IMAGE_NAME="tinkercloud"
TAG="${TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "=== Building TinkerCloud Image ==="
echo "  Project: ${PROJECT_DIR}"
echo "  Image: ${FULL_IMAGE}"
echo ""

# Build image
docker build \
    -t "${FULL_IMAGE}" \
    -f docker/Dockerfile \
    .

echo ""
echo "=== Build Complete ==="
echo "  Image: ${FULL_IMAGE}"

# Push if requested
if [[ "$1" == "--push" ]]; then
    echo ""
    echo "=== Pushing to Registry ==="
    docker push "${FULL_IMAGE}"
    echo "Pushed: ${FULL_IMAGE}"
fi

echo ""
echo "=== Done ==="
