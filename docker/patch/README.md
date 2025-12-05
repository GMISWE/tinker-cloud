# Tinker GMI Patches

This directory contains patches for tinker_gmi to add integration testing infrastructure.

## Patch Versions

### latest/
- **Base commit**: `9ba155a` (Sync contents)
- **Target commit**: `e7a43affeb3cc037acd0b5bd969e4d5c0f50f852`
- **Description**: Integration testing infrastructure for the Tinker GMI Wrapper (#1)
- **Repository**: https://github.com/thinking-machines-lab/tinker.git

## Patch Contents

The `tinker_gmi.patch` includes:
- Mock server test scaffold (`mock_server.py`)
- Integration tests for GMI HTTP endpoints (`tests_integration/gmi_http/`)
- End-to-end Tinker API tests (`tests_integration/e2e_tinker_api/`)
- Checkpoint resume functionality
- RL rollout pattern tests
- Deployment verification script (`verify_deployment.sh`)

## Generating a New Patch

To generate a patch from tinker_gmi repository changes:

```bash
cd /path/to/tinker_gmi
git diff <base_commit>..<target_commit> > ../opentinker-miles/docker/patch/latest/tinker_gmi.patch
```

## Applying Patches

Patches are applied in the Dockerfile using:

```dockerfile
COPY docker/patch/${PATCH_VERSION}/tinker_gmi.patch /workspace/tinker_gmi/
RUN cd /workspace/tinker_gmi && \
    git apply tinker_gmi.patch && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm tinker_gmi.patch
```
