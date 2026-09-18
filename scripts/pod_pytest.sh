#!/usr/bin/env bash
# Run the CPU-only pytest suites (tests/protocol on the fake backend, unit
# tests) on a cluster pod against the LOCAL working tree, without touching
# the deployed /app.
#
#   scripts/pod_pytest.sh [pod] [pytest args...]
#   scripts/pod_pytest.sh tinkercloud-nemorl tests/protocol tests/test_ordering.py -q
#
# The tree is tarred (training/ tests/ gates/ pyproject.toml), pushed with
# sha256 verified on both sides, unpacked under /tmp/pytest-<user>/app and
# aliased as the `tinkercloud` package the tests import. Default args run
# the suites that need no GPU.
set -euo pipefail
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/ns.config}"
NS="${NS:-tinkercloud-nemorl}"
POD="${1:-tinkercloud-nemorl}"; shift || true
if [ $# -eq 0 ]; then
  set -- tests/protocol tests/test_ordering.py tests/test_backend_interface.py \
         tests/test_loss_registry.py tests/test_e0_registry.py tests/test_checkpoint_interchange.py \
         tests/test_checkpoint_store.py tests/test_miles_resume_args.py \
         tests/test_validators.py tests/test_miles_rl_layout.py tests/test_optim_metrics_seam.py \
         tests/test_miles_padding.py tests/test_futures_storage.py tests/test_result_validation.py tests/test_backend_config.py tests/test_miles_pack_length.py tests/test_miles_dp_reduction.py \
         tests/test_routing.py tests/test_sglang_client_pool.py -q -p no:cacheprovider
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="/tmp/pytest-${USER:-u}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

kx() { # retry kubectl on control-plane drops (pod-ops §12)
  local i; for i in 1 2 3 4; do
    if timeout "${KX_TIMEOUT:-600}" kubectl -n "$NS" "$@"; then return 0; fi
    echo "  kubectl retry $i" >&2; sleep 5
  done; return 1
}

tar -C "$ROOT" -czf "$TMP/code.tgz" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    training tests gates pyproject.toml
SHA="$(sha256sum "$TMP/code.tgz" | cut -d' ' -f1)"
for try in 1 2 3; do
  kx cp "$TMP/code.tgz" "$POD:$REMOTE.tgz"
  RSHA="$(kx exec "$POD" -- sha256sum "$REMOTE.tgz" | cut -d' ' -f1)"
  [ "$RSHA" = "$SHA" ] && break
  echo "  sha mismatch on push (try $try)" >&2; [ "$try" = 3 ] && exit 1
done
kx exec "$POD" -- bash -c "rm -rf '$REMOTE/app' && mkdir -p '$REMOTE/app' && tar -C '$REMOTE/app' -xzf '$REMOTE.tgz' && ln -sfn '$REMOTE/app' '$REMOTE/tinkercloud'"
echo "==> $POD:$REMOTE/app (sha $SHA)"
# only the fake backend and CPU suites run here; keep the pod's real backend env out
# not retried: a non-zero exit here is pytest's verdict, not a dropped connection
timeout "${KX_TIMEOUT:-600}" kubectl -n "$NS" exec "$POD" -- bash -c "cd '$REMOTE/app' && env -u TINKERCLOUD_BACKEND -u RAY_ADDRESS PYTHONPATH='$REMOTE' python -m pytest $(printf '%q ' "$@")"
