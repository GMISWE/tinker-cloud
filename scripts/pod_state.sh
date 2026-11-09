#!/usr/bin/env bash
# What the cluster holds right now, in one exec per pod: GPU-holding pods on
# the node, and for each tinkercloud pod its GPU memory, API server, Ray usage
# and live actors. Run it before any plan that assumes a pod or a free GPU;
# handoffs and memories record how to check, never the state (pod-ops).
#
#   scripts/pod_state.sh            # every pod in $NS whose name starts with tinkercloud-
#   scripts/pod_state.sh <pod>...   # these pods only
set -uo pipefail
# shellcheck source=scripts/lib/cluster.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, kx, pod_cmd, ray_head
require_tunnel
export KX_TIMEOUT="${KX_TIMEOUT:-60}"

echo "== GPU-holding pods on $DEFAULT_NODE (allocatable: $(kx get node "$DEFAULT_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo '?')) =="
kx get pods -A -o json 2>/dev/null | python3 -c '
import json, sys
for p in json.load(sys.stdin)["items"]:
    g = sum(int(c["resources"].get("limits", {}).get("nvidia.com/gpu", 0)) for c in p["spec"]["containers"])
    if g:
        print(f'"'"'  {p["metadata"]["namespace"]}/{p["metadata"]["name"]}  {p["status"]["phase"]}  gpus={g}  age={p["metadata"]["creationTimestamp"]}'"'"')
' || echo "  (pod list unavailable: control plane dropped)"

if [ $# -gt 0 ]; then PODS=("$@"); else
  mapfile -t PODS < <(kx get pods -o name 2>/dev/null | sed 's#^pod/##' | grep '^tinkercloud-' || true)
fi
[ ${#PODS[@]} -gt 0 ] || { echo "== no tinkercloud- pod in $NS =="; exit 0; }

# the deployed tree carries no git metadata: compare content hashes of training/*.py
LOCAL_TREE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && find training -name '*.py' | sort | xargs sha256sum | cut -d' ' -f1 | sha256sum | cut -c1-12)
for pod in "${PODS[@]}"; do
  echo "== $pod =="
  # shellcheck disable=SC2016  # the string runs on the pod; its $ expand there
  kx exec "$pod" -- bash -c "$(pod_cmd '
    echo "  gpu MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr "\n" " ")"
    echo "  server:  $(curl -s -m 5 localhost:8000/health || echo DOWN)"
    ps -eo pid,etime,args | grep "m trainin[g]$" | sed "s/^/  proc:    /"
    echo "  ray:     $(ray status 2>/dev/null | grep -Eo "[0-9.]+/[0-9.]+ GPU" | head -1 || echo "no head")"
    echo "  actors:  $(ray list actors --filter state=ALIVE 2>/dev/null | grep -c ALIVE) alive"
    echo "  pypi:    $(python3 -c "import importlib.metadata as m,sys; print(*(p+\"=\"+m.version(p) for p in sys.argv[1:]))" uvicorn fastapi pydantic httpx 2>/dev/null)"
    echo "  /app:    training/ tree $(cd /app && find training -name "*.py" 2>/dev/null | sort | xargs sha256sum 2>/dev/null | cut -d" " -f1 | sha256sum | cut -c1-12) (local: '"$LOCAL_TREE"')"
  ')" 2>/dev/null || echo "  (exec dropped: state unknown)"
  echo "  head:    $(kx exec "$pod" -- bash -c "$(pod_cmd "$(ray_head state)")" 2>/dev/null || echo "unknown (exec dropped)")"
done
