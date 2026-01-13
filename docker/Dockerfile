# OpenTinker-Miles: All-in-One Tinker API + Miles RL Training
#
# This image combines:
# - Miles RL training backend (Ray head + Megatron)
# - OpenTinker training API (Tinker-compatible endpoints)
# - Pre-downloaded model (Qwen2.5-0.5B-Instruct) + Megatron conversion
# - Pre-downloaded dataset (GSM8K)
#
# Usage:
#   cd /tmp/tmux-tmp/opentinker-miles
#   docker build -t us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/opentinker-miles:latest -f docker/Dockerfile .
#   docker push us-west1-docker.pkg.dev/devv-404803/gmi-test-repo/opentinker-miles:latest

ARG BASE_IMAGE=gmicloudai/tinkercloud:dev
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================
ARG PATCH_VERSION=latest
ARG TINKER_GMI_COMMIT=2d8e9d5
ARG MILES_COMMIT=22f2527
ARG TINKER_COOKBOOK_COMMIT=9934ede

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

# Create data directories (will be overridden if /data is mounted)
RUN mkdir -p /data/models /data/datasets /data/checkpoints /data/trajectories /data/metadata

# Copy data preparation and model conversion scripts
COPY docker/prepare_data.sh /prepare_data.sh
COPY docker/convert_model.sh /convert_model.sh
RUN chmod +x /prepare_data.sh /convert_model.sh

# Note: Data download/conversion is handled at runtime by entrypoint.sh
# This allows mounting existing data at /data to skip downloads

# ====================================== Tinker GMI Patches ============================================
# Clone tinker at base commit and apply patches
RUN git clone https://github.com/thinking-machines-lab/tinker.git /workspace/tinker_gmi && \
    cd /workspace/tinker_gmi && \
    git checkout ${TINKER_GMI_COMMIT}

COPY docker/patch/${PATCH_VERSION}/tinker_gmi.patch /workspace/tinker_gmi/
RUN cd /workspace/tinker_gmi && \
    git apply tinker_gmi.patch && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "tinker_gmi patch failed to apply cleanly." && \
      exit 1; \
    fi && \
    rm tinker_gmi.patch && \
    pip install -e .

# ====================================== Megatron Patches ============================================
# NOTE: Base image (gmicloudai/tinkercloud:dev) already has Megatron patches applied:
#   - 43a60cf27: Fix BytesIO deserialization in checkpoint loading
#   - 339779dc1: [TMP] Miles patch for Megatron-LM core_v0.14.0
# These enable: LoRA support, partial checkpoint loading, gated attention, MTP+CP
# No additional patching needed.

# ====================================== Miles RLVE ============================================
# Clone Miles from upstream and apply RLVE patches
# Note: Base image may have /root/miles, remove it first
RUN rm -rf /root/miles && \
    git clone https://github.com/radixark/miles.git /root/miles && \
    cd /root/miles && \
    git checkout ${MILES_COMMIT}

COPY docker/patch/${PATCH_VERSION}/miles_rlve.patch /root/miles/
RUN cd /root/miles && \
    git apply miles_rlve.patch && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Miles patch failed to apply cleanly." && \
      exit 1; \
    fi && \
    rm miles_rlve.patch && \
    pip install -e . --no-deps

# ====================================== tinker-cookbook RLVE ============================================
# Clone tinker-cookbook from upstream and apply RLVE patches
RUN git clone https://github.com/thinking-machines-lab/tinker-cookbook.git /workspace/tinker-cookbook && \
    cd /workspace/tinker-cookbook && \
    git checkout ${TINKER_COOKBOOK_COMMIT}

COPY docker/patch/${PATCH_VERSION}/tinker_cookbook_rlve.patch /workspace/tinker-cookbook/
RUN cd /workspace/tinker-cookbook && \
    git apply tinker_cookbook_rlve.patch && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "tinker-cookbook patch failed to apply cleanly." && \
      exit 1; \
    fi && \
    rm tinker_cookbook_rlve.patch && \
    pip install -e .

# Environment - Training API
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/root/miles:/root/Megatron-LM:/root/miles/examples/RLVE:/workspace/tinker-cookbook" \
    TOKENIZERS_PARALLELISM="false" \
    TRAINING_HOST="0.0.0.0" \
    TRAINING_PORT="8000" \
    TINKER_API_KEY="slime-dev-key" \
    LOG_LEVEL="INFO"

# Environment - GPU/NCCL
ENV CUDA_DEVICE_MAX_CONNECTIONS="1" \
    NCCL_NVLS_ENABLE="0" \
    NCCL_DEBUG="INFO" \
    NCCL_IB_DISABLE="0" \
    NCCL_SOCKET_IFNAME="eth0"

# Environment - Ray
ENV NUM_GPUS="4"

# Environment - HuggingFace
ENV HF_HOME="/data" \
    HF_HUB_OFFLINE="0" \
    TRANSFORMERS_OFFLINE="0"

# Ports: 8000 (API), 8265 (Ray dashboard), 10001 (Ray client), 30000 (SGLang)
EXPOSE 8000 8265 10001 30000

ENTRYPOINT ["/entrypoint.sh"]
