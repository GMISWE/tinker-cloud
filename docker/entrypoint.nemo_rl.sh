#!/bin/bash
# TinkerCloud NeMo RL Entrypoint
#
# This script:
# 1. Creates required data directories
# 2. Verifies NeMo RL installation
# 3. Starts Ray head node
# 4. Starts the OpenTinker training API with NeMo RL backend
#
# Environment variables:
#   NUM_GPUS: Number of GPUs for Ray (default: auto-detect)
#   TRAINING_PORT: Port for training API (default: 8000)
#   RAY_DASHBOARD_PORT: Port for Ray dashboard (default: 8265)
#   RAY_CLIENT_PORT: Port for Ray client (default: 10001)
#   SKIP_RAY: Set to "1" to skip Ray startup (for connecting to external Ray)
#   TINKERCLOUD_BACKEND: Must be "nemo_rl" (set in Dockerfile)

set -e

echo "========================================"
echo "TinkerCloud Starting (NeMo RL backend)..."
echo "========================================"

# Create data directories (mounted from host or emptyDir)
echo "Creating data directories..."
mkdir -p /data/models /data/checkpoints /data/datasets /data/trajectories /data/metadata
chmod -R 777 /data 2>/dev/null || true

# Prepare data: download models and datasets if not already present
if [ -f /prepare_data.sh ]; then
    echo "Preparing data..."
    /prepare_data.sh || {
        echo "WARNING: Data preparation failed."
    }
fi

# Verify NeMo RL is available (replaces Miles check from entrypoint.sh)
echo "Checking NeMo RL installation..."
python -c "import nemo_rl; print(f'NeMo RL version: {getattr(nemo_rl, \"__version__\", \"installed\")}')" || {
    echo "ERROR: NeMo RL not found in PYTHONPATH"
    exit 1
}

# Verify backend is set correctly
if [ "${TINKERCLOUD_BACKEND}" != "nemo_rl" ]; then
    echo "WARNING: TINKERCLOUD_BACKEND=${TINKERCLOUD_BACKEND}, expected 'nemo_rl'"
    echo "Setting TINKERCLOUD_BACKEND=nemo_rl"
    export TINKERCLOUD_BACKEND="nemo_rl"
fi

# Start Ray head node (unless SKIP_RAY is set)
if [ "${SKIP_RAY}" != "1" ]; then
    # Auto-detect GPUs if not specified
    if [ -z "${NUM_GPUS}" ]; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        echo "Auto-detected ${NUM_GPUS} GPUs"
    fi

    # Get node IP
    NODE_IP=${MASTER_ADDR:-$(hostname -i)}

    # Clean stale Ray session data to avoid session name mismatch
    rm -rf /tmp/ray 2>/dev/null || true

    echo "Starting Ray head node..."
    echo "  Node IP: ${NODE_IP}"
    echo "  GPUs: ${NUM_GPUS}"
    echo "  Dashboard: ${RAY_DASHBOARD_PORT:-8265}"
    echo "  Client: ${RAY_CLIENT_PORT:-10001}"

    ray start --head \
        --node-ip-address "${NODE_IP}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
        --ray-client-server-port="${RAY_CLIENT_PORT:-10001}"

    # Wait for Ray to be ready
    sleep 2
    ray status

    # Set RAY_ADDRESS for the training API
    export RAY_ADDRESS="ray://localhost:${RAY_CLIENT_PORT:-10001}"
else
    echo "SKIP_RAY=1, not starting Ray head node"
    echo "Expecting RAY_ADDRESS to be set externally: ${RAY_ADDRESS}"
fi

echo ""
echo "========================================"
echo "Starting OpenTinker Training API..."
echo "  Host: ${TRAINING_HOST:-0.0.0.0}"
echo "  Port: ${TRAINING_PORT:-8000}"
echo "  Backend: ${TINKERCLOUD_BACKEND}"
echo "  Ray: ${RAY_ADDRESS}"
echo "========================================"
echo ""

# Start the training API (foreground)
exec python3 -m training
