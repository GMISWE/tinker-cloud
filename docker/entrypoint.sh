#!/bin/bash
# OpenTinker-Miles All-in-One Entrypoint
#
# This script:
# 1. Creates required data directories
# 2. Starts Ray head node
# 3. Starts the OpenTinker training API
#
# Environment variables:
#   NUM_GPUS: Number of GPUs for Ray (default: auto-detect)
#   TRAINING_PORT: Port for training API (default: 8000)
#   RAY_DASHBOARD_PORT: Port for Ray dashboard (default: 8265)
#   RAY_CLIENT_PORT: Port for Ray client (default: 10001)
#   SKIP_RAY: Set to "1" to skip Ray startup (for connecting to external Ray)

set -e

echo "========================================"
echo "OpenTinker-Miles Starting..."
echo "========================================"

# Create data directories (mounted from host or emptyDir)
echo "Creating data directories..."
mkdir -p /data/models /data/checkpoints /data/datasets /data/trajectories /data/metadata
chmod -R 777 /data 2>/dev/null || true

# Convert model to Megatron format if needed (requires GPU)
if [ -f /convert_model.sh ]; then
    echo "Checking if model conversion is needed..."
    /convert_model.sh || {
        echo "WARNING: Model conversion failed. This might be expected if no GPU is available."
        echo "The conversion will need to be done manually when GPU is available."
    }
fi

# Verify Miles is available
echo "Checking Miles installation..."
python -c "import miles; print(f'Miles version: {getattr(miles, \"__version__\", \"installed\")}')" || {
    echo "ERROR: Miles not found in PYTHONPATH"
    exit 1
}

# Start Ray head node (unless SKIP_RAY is set)
if [ "${SKIP_RAY}" != "1" ]; then
    # Auto-detect GPUs if not specified
    if [ -z "${NUM_GPUS}" ]; then
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
        echo "Auto-detected ${NUM_GPUS} GPUs"
    fi

    # Get node IP
    NODE_IP=${MASTER_ADDR:-$(hostname -i)}

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
echo "  Ray: ${RAY_ADDRESS}"
echo "========================================"
echo ""

# Start the training API (foreground)
exec python3 -m training
