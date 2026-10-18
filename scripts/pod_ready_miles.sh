#!/usr/bin/env bash
# Idempotent readiness check + repair for the tinkercloud-miles pod.
# Run after deploy_tinkercloud.sh --profile miles (or at the start of any
# session): every step verifies before it acts, so a healthy pod passes in
# seconds and a fresh pod is fully staged.
#
#   pod_ready_miles.sh [model-basename]     # default Qwen2.5-0.5B
#
# Covers what deploy_tinkercloud.sh does not:
#   - HF weights staged + miles' FLAT model dir (not hub layout)
#   - torch_dist conversion (/data is an emptyDir — dies with the pod)
#   - Megatron-Bridge hide_adapters patch (lands in pod site-packages,
#     lost on every rebuild/re-pull — see scripts/patch_bridge_hide_adapters.py)
#   - /data service dirs
#   - code-hash spot-check of the deployed code vs the local source trees
set -euo pipefail

MODEL="${1:-Qwen2.5-0.5B}"
HF_ID="${HF_ID:-Qwen/$MODEL}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, DEFAULT_NODE
POD="${POD:-tinkercloud-miles}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# deploy_tinkercloud.sh --source dev bundles the sibling checkouts of
# tinker-cookbook and tinker_gmi next to this repo; the hash check below
# compares the deployed copies against the same siblings.
SRC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EX() { kubectl -n "$NS" exec "$POD" -- bash -c "$1"; }

echo "==> [1/6] pod alive + GPUs visible"
EX "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"

echo "==> [2/6] /data service dirs"
EX "mkdir -p /data/metadata /data/checkpoints /data/conv /data/trajectories"

echo "==> [3/6] HF weights + flat dir ($MODEL)"
if ! EX "test -f /data/.cache/huggingface/$MODEL/config.json"; then
  EX "export HF_HOME=/data/.cache/huggingface HF_TOKEN=\$(cat /tmp/hf_token.txt)
      python3 -c \"from huggingface_hub import snapshot_download; print(snapshot_download('$HF_ID'))\" > /tmp/snap_path.txt
      ln -sfn \$(cat /tmp/snap_path.txt) /data/.cache/huggingface/$MODEL
      test -f /data/.cache/huggingface/$MODEL/config.json"
  echo "    staged + flat-linked"
else
  echo "    present"
fi

echo "==> [4/6] torch_dist conversion"
if ! EX "ls /data/.cache/huggingface/${MODEL}_torch_dist/iter_* >/dev/null 2>&1 || test -f /data/.cache/huggingface/${MODEL}_torch_dist/latest_checkpointed_iteration.txt"; then
  # scripts/models/ names keep the size capitalized (qwen3-8B.sh) and have
  # no -Base suffix; full lowercasing never matched (latent until the first
  # fresh-pod 8B conversion, 2026-08-25).
  MODEL_SH=$(echo "$MODEL" | sed 's/-Base$//; s/^Qwen/qwen/')
  EX "set -u; export HF_HOME=/data/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
      cd /root/miles && source scripts/models/${MODEL_SH}.sh
      PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 1 \
        tools/convert_hf_to_torch_dist.py \"\${MODEL_ARGS[@]}\" \
        --hf-checkpoint /data/.cache/huggingface/$MODEL \
        --save /data/.cache/huggingface/${MODEL}_torch_dist"
  echo "    converted"
else
  echo "    present"
fi

echo "==> [5/6] hide_adapters patch"
kubectl -n "$NS" cp "$SCRIPT_DIR/patch_bridge_hide_adapters.py" "$POD:/data/patch_bridge_hide_adapters.py"
EX "python3 /data/patch_bridge_hide_adapters.py"

echo "==> [6/6] code-hash spot-check vs local source trees"
FAIL=0
for pair in \
  "tinker-cloud/training/backends/miles/backend.py /app/training/backends/miles/backend.py" \
  "tinker-cookbook/tinker_cookbook/supervised/train.py /work/tinker-cookbook/tinker_cookbook/supervised/train.py" \
  "tinker_gmi/src/tinker/__init__.py /work/tinker_gmi/src/tinker/__init__.py"; do
  L=$(echo "$pair" | cut -d' ' -f1); R=$(echo "$pair" | cut -d' ' -f2)
  LH=$(md5sum "$SRC_ROOT/$L" | cut -d' ' -f1)
  RH=$(EX "md5sum $R" | cut -d' ' -f1)
  if [ "$LH" != "$RH" ]; then echo "    DRIFT: $R != local $L"; FAIL=1; fi
done
if [ "$FAIL" = 1 ]; then
  echo "    pod code drifted — rerun: deploy_tinkercloud.sh --profile miles --code-only"
  exit 1
fi
echo "==> READY: $POD staged for $MODEL"
