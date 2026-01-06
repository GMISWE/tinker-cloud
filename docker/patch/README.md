# Tinker GMI Patches

This directory contains patches for tinker to support GMI/Slime checkpoint and training features.

## Patch Versions

### latest/
- **Base commit**: `9ba155a` (Sync contents)
- **Target commit**: `e7a43affeb3cc037acd0b5bd969e4d5c0f50f852`
- **Repository**: https://github.com/thinking-machines-lab/tinker.git

## Patch Contents

The `tinker_gmi.patch` modifies `src/tinker/lib/public_interfaces/service_client.py`:

- Adds `debug_train_only`, `checkpoint_path`, `user_metadata` parameters to `_create_model_submit()`
- Adds same parameters to `create_lora_training_client()` and `create_lora_training_client_async()`
- Updates `resume_training_run()` and `resume_training_run_async()` to load checkpoints during model creation (Slime/Megatron pattern) instead of separate `load_state()` call

## Generating a New Patch

To generate a patch for specific files only:

```bash
cd /path/to/tinker_gmi
git diff <base_commit>..<target_commit> -- src/tinker/lib/public_interfaces/service_client.py > ../tinkercloud/docker/patch/latest/tinker_gmi.patch
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
