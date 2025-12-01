#!/bin/bash
# Setup script for miles-gmi-tinker namespace
# Copies existing Slime/Miles assets into the miles hostPath on the worker node.

set -euo pipefail

SOURCE_PATH=${SOURCE_PATH:-/mnt/slime-data}
DEST_PATH=${DEST_PATH:-/mnt/miles-data-tinker}
NODE=${NODE:-hgxcn24}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=========================================="
log "Miles-GMI-Tinker Data Setup"
log "=========================================="
log "Source: ${SOURCE_PATH}"
log "Destination: ${DEST_PATH}"
log "Node: ${NODE}"
log "=========================================="

log "Checking source directory on ${NODE}..."
if ! ssh "${NODE}" "[ -d '${SOURCE_PATH}' ]"; then
  log "ERROR: Source directory ${SOURCE_PATH} does not exist on ${NODE}."
  log "       Set SOURCE_PATH to a directory containing models/datasets."
  exit 1
fi
log "✓ Source directory exists"

log "Ensuring destination directories exist..."
ssh "${NODE}" "mkdir -p ${DEST_PATH}/{models,checkpoints,datasets,trajectories,metadata}"
log "✓ Destination structure ready"

copy_dir() {
  local label=$1
  local subdir=$2

  if ssh "${NODE}" "[ -d '${SOURCE_PATH}/${subdir}' ]"; then
    SIZE=$(ssh "${NODE}" "du -sh ${SOURCE_PATH}/${subdir} | cut -f1")
    log "Copying ${label} (${SIZE})..."
    ssh "${NODE}" "cp -rf ${SOURCE_PATH}/${subdir}/. ${DEST_PATH}/${subdir}/"
    log "✓ ${label} copied"
  else
    log "⚠ No ${label} found at ${SOURCE_PATH}/${subdir}, skipping"
  fi
}

copy_dir "models" "models"
copy_dir "datasets" "datasets"
copy_dir "checkpoints" "checkpoints"
copy_dir "trajectories" "trajectories"
copy_dir "metadata" "metadata"

log "Setting permissions on ${DEST_PATH}..."
ssh "${NODE}" "chmod -R 777 ${DEST_PATH}"
log "✓ Permissions updated"

log "Destination contents:"
ssh "${NODE}" "ls -lh ${DEST_PATH}"
log "Disk usage by subdirectory:"
ssh "${NODE}" "du -sh ${DEST_PATH}/* || true"

log "=========================================="
log "✓ Data setup complete"
log "Mount path ready for miles-gmi-tinker pods"
log "=========================================="

log "Next: kubectl apply -f miles-gmi-tinker"
