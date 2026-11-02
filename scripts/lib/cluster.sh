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
