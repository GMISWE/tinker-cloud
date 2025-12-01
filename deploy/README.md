# Deployment Guide

This directory contains the Kubernetes manifests and helper scripts used to run
the Miles Ray cluster plus the kgateway/OpenTinker training API inside the
`miles-gmi-tinker` namespace. The layout assumes a single GPU node (e.g.
`hgxcn24`) with a hostPath volume at `/mnt/miles-data-tinker` that holds models,
datasets, and checkpoints shared between the Miles and kgateway pods.

```
└── deploy
    ├── 00-namespace.yaml          # Namespace definition
    ├── 01-secrets.yaml            # (Optional) image pull or API secrets
    ├── 03-miles-statefulset.yaml  # Miles Ray head pod + init container to prep weights
    ├── 04-kgateway-deployment.yaml# FastAPI training bridge pointing at Ray head
    ├── 06-services.yaml           # Headless service for Ray + ClusterIP for Miles dashboard
    └── setup-data.sh              # Helper to copy existing model/data assets into hostPath
```

## Prerequisites

- Kubernetes cluster with at least one multi-GPU worker (manifests pin to `kubernetes.io/hostname=hgxcn24`; adjust if needed).
- Container registry credentials for pulling `us-west1-docker.pkg.dev/...` images (create the `gcp-secret` or edit `imagePullSecrets`).
- SSH access to the worker node so `setup-data.sh` can copy models/datasets.
- HF Hub access (init container downloads base checkpoints unless the hostPath already contains them).

## Quick Start

1. **Set up the namespace and secrets**
   ```bash
   kubectl apply -f deploy/00-namespace.yaml
   # Either edit deploy/01-secrets.yaml or create secrets manually, e.g.:
   kubectl --namespace miles-gmi-tinker \
     create secret docker-registry gcp-secret \
     --docker-server=us-west1-docker.pkg.dev \
     --docker-username=_json_key \
     --docker-password="$(cat key.json)" \
     --docker-email=dev@example.com
   ```

2. **Pre-populate the hostPath (optional but recommended)**
   ```bash
   cd deploy
   SOURCE_PATH=/mnt/slime-data \
   DEST_PATH=/mnt/miles-data-tinker \
   NODE=hgxcn24 \
   ./setup-data.sh
   ```
   This copies existing `models/`, `datasets/`, `checkpoints/`, etc. into the mount
   location. If you skip this step the init container in the StatefulSet will
   download the referenced Hugging Face model/dataset on first boot.

3. **Deploy Miles (Ray head + storage)**
   ```bash
   kubectl apply -f deploy/03-miles-statefulset.yaml
   kubectl apply -f deploy/06-services.yaml
   ```
   The StatefulSet:
   - Mounts `/mnt/miles-data-tinker` into `/data`
   - Runs an `init-model-weights` container that downloads `Qwen2.5-0.5B-Instruct` and GSM8k if missing
   - Starts Ray with dashboard (`:8265`) and Ray Client (`:10001`) ports exposed via the services declared in `06-services.yaml`

   Verify the pod:
   ```bash
   kubectl get pods -n miles-gmi-tinker
   kubectl logs miles-training-0 -n miles-gmi-tinker -c miles | tail
   ```

4. **Deploy kgateway/OpenTinker bridge**
   ```bash
   kubectl apply -f deploy/04-kgateway-deployment.yaml
   ```
   This deployment:
   - Mounts the same hostPath (`/data`) so checkpoints/models are shared
   - Points `RAY_ADDRESS` at the headless service (`ray://miles-headless.miles-gmi-tinker:10001`)
   - Exposes `TRAINING_PORT=8000` inside the cluster via the `kgateway-training` service

5. **Smoke test**
   ```bash
   kubectl port-forward -n miles-gmi-tinker svc/kgateway-training 8000:8000
   curl http://localhost:8000/health
   ```
   You should see `{"status":"healthy","version":"3.1.0","ray_initialized":true,...}`.

## Customization Tips

- **Node Selectors / Tolerations**: Both manifests target `kubernetes.io/hostname: hgxcn24`.
  Modify `spec.template.spec.nodeSelector` and tolerations if your GPU nodes use different labels.
- **Model bootstrap**: Edit the `init-model-weights` script inside `03-miles-statefulset.yaml` to download or convert different checkpoints.
- **Environment variables**: The kgateway deployment sets `TINKER_API_KEY`, `LOG_LEVEL`, and `RUN_MODE`. Override or extend as needed.
- **Storage**: If you prefer PVCs instead of hostPath, swap the `volumes` and `volumeMounts` blocks in both manifests accordingly.

## Teardown

```bash
kubectl delete -f deploy/04-kgateway-deployment.yaml
kubectl delete -f deploy/03-miles-statefulset.yaml
kubectl delete -f deploy/06-services.yaml
kubectl delete namespace miles-gmi-tinker
```

Remember to clean up `/mnt/miles-data-tinker` on the worker node if you no longer
need the cached models/checkpoints.

---

For additional context on how the FastAPI training service is structured, see
the root `README.md`. This file is focused solely on operational deployment.
