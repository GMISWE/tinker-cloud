# OpenTinker-Miles: All-in-One Tinker API + Miles RL Training
#
# This image combines:
# - Miles RL training backend (Ray head + Megatron)
# - OpenTinker training API (Tinker-compatible endpoints)
#
# Replaces the separate miles-statefulset + kgateway-deployment with a single pod.
#
# Usage:
#   cd /tmp/tmux-tmp/opentinker-miles
#   docker build -t us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/opentinker-miles:latest -f docker/Dockerfile .
#   docker push us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/opentinker-miles:latest

ARG BASE_IMAGE=us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/miles_dev-202511120a-dev:latest
FROM ${BASE_IMAGE}

LABEL maintainer="OpenTinker Team"
LABEL description="All-in-One Tinker API + Miles RL Training"
LABEL version="1.0"

# Install opentinker-miles dependencies
# Note: ray, torch, transformers already in Miles base image
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install chz termcolor

# Copy training API source
COPY training ./training

# Copy startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/root/miles:/root/Megatron-LM" \
    TRAINING_HOST="0.0.0.0" \
    TRAINING_PORT="8000" \
    CUDA_DEVICE_MAX_CONNECTIONS="1" \
    NCCL_NVLS_ENABLE="0"

# Ports: 8000 (API), 8265 (Ray dashboard), 10001 (Ray client), 30000 (SGLang)
EXPOSE 8000 8265 10001 30000

ENTRYPOINT ["/entrypoint.sh"]
