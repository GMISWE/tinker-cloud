# shellcheck shell=bash
# Cluster defaults shared by every script that drives a pod. One place to
# change when the cluster moves; each value is overridable from the env.
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/h1.config}"
NS="${NS:-tinkercloud-nemorl}"
DEFAULT_NODE="${DEFAULT_NODE:-hgxcn13}"

# The API server sits behind an ssh tunnel. A refused connection means the
# tunnel is down and no retry can help; any other failure (TLS or i/o timeout)
# is a control-plane drop that the caller's retries are for.
require_tunnel() {
  local out
  out=$(kubectl version --request-timeout=5s 2>&1) && return 0
  case "$out" in
    *refused*) echo "kubectl: connection refused under $KUBECONFIG; restore the tunnel first: ssh h1" >&2; exit 2 ;;
  esac
  return 0
}

# kubectl helpers every pod script shares (pod-ops: every exec is droppable).
# kx: any kubectl verb, retried on control-plane drops; each try is bounded by
# KX_TIMEOUT seconds (default 600). EX: one exec of a bash string on $POD, no
# retry (a launch retried on rc double-launches); EX_TIMEOUT bounds that exec
# and treats the timeout as no verdict. Callers set POD before the first call.
kx() {
  local i; for i in 1 2 3 4; do
    if timeout "${KX_TIMEOUT:-600}" kubectl -n "$NS" "$@"; then return 0; fi
    echo "  kubectl retry $i" >&2; sleep 5
  done; return 1
}
EX() { kubectl -n "$NS" exec "$POD" -- bash -c "$1"; }
EX_TIMEOUT() { timeout "$1" kubectl -n "$NS" exec "$POD" -- bash -c "$2" || true; }

# require_server_env KEY=VAL ...: refuse unless the running API server on $POD has every
# KEY=VAL in its own environment (read from /proc, not from the boot command).
require_server_env() {
  local envs kv
  envs=$(EX "p=\$(pgrep -f '^python3 -m trainin[g]' | head -1); [ -n \"\$p\" ] && tr '\0' '\n' < /proc/\$p/environ") \
    || { echo "ERROR: no API server running on $POD" >&2; exit 1; }
  for kv in "$@"; do
    grep -qx -- "$kv" <<< "$envs" || {
      echo "ERROR: the server on $POD lacks $kv (has: $(grep -- "^${kv%%=*}=" <<< "$envs" || echo unset))" >&2
      echo "  rotate it with that setting: scripts/pod_rotate_server.sh <log> $kv ..." >&2
      exit 1; }
  done
}

# refuse_if_gpus_busy: refuse when any GPU on $POD holds memory (a probe or model is resident);
# FORCE=1 overrides. A test sharing GPUs with a run skews its memory readings or OOMs it.
refuse_if_gpus_busy() {
  local used
  [ "${FORCE:-0}" = 1 ] && return 0
  used=$(EX "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1>0' | wc -l") \
    || { echo "ERROR: could not read GPU memory on $POD" >&2; exit 1; }
  if [ "$used" != 0 ]; then
    echo "ERROR: $used GPU(s) on $POD hold memory; a run is resident (FORCE=1 to share them):" >&2
    EX "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -4" >&2
    exit 1
  fi
}
