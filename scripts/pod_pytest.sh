#!/usr/bin/env bash
# Run the CPU-only pytest suites (tests/unit, and tests/conformance on the fake
# backend) on a cluster pod against the LOCAL working tree, without touching the
# deployed /app.
#
#   scripts/pod_pytest.sh [pod] [pytest args...]
#   scripts/pod_pytest.sh tinkercloud-nemorl tests/conformance tests/unit/core -q
#
# The tree is tarred (training/ tests/ gates/ pyproject.toml), pushed with
# sha256 verified on both sides, unpacked under /tmp/pytest-<user>/app and
# aliased as the `tinkercloud` package the tests import. Default args run
# the suites that need no GPU.
#
# GPU suites (tests/gpu/...) take minutes per test and pytest prints the
# tracebacks only at the end, so a tests/gpu path adds -x unless -x or
# --maxfail is given; rerun one test with -k <name>. Their kubectl timeout
# defaults to 1800 s instead of 600 (KX_TIMEOUT overrides). They start their own Ray
# instance, which Ray refuses while a head runs on the pod, so the run stops
# before the push when one does.
#
# Every run appends `date pod args | verdict` to ~/.cache/tinkercloud/pod_pytest.log,
# also one that stops before pytest (tunnel down, push failed, Ray head up).
set -euo pipefail
# shellcheck source=scripts/lib/cluster.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, DEFAULT_NODE
POD="${1:-tinkercloud-nemorl}"; shift || true
if [ $# -eq 0 ]; then
  set -- tests/unit tests/conformance -q -p no:cacheprovider
fi
GPU_RUN=0
case " $* " in
  *" tests/gpu"*) GPU_RUN=1; case " $* " in *" -x "*|*" --maxfail"*) ;; *) set -- "$@" -x ;; esac ;;
esac
# kubectl calls time out at KX_TIMEOUT seconds; a GPU suite needs the long default
KX_TIMEOUT="${KX_TIMEOUT:-$([ "$GPU_RUN" = 1 ] && echo 1800 || echo 600)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="/tmp/pytest-${USER:-u}"
LOG="$HOME/.cache/tinkercloud/pod_pytest.log"; mkdir -p "$(dirname "$LOG")"
LOGGED=0; ARGS="$*"   # the trap may run inside a function, where $* is that function's
TMP="$(mktemp -d)"
# every invocation leaves one log line, also one that dies before pytest (tunnel down, push failed, Ray guard)
# shellcheck disable=SC2329  # invoked by the trap below
on_exit() {
  local rc=$?
  rm -rf "$TMP"
  [ "$LOGGED" = 1 ] || printf '%s %s %s | no run (exit %s before pytest)\n' "$(date -u +%FT%TZ)" "$POD" "$ARGS" "$rc" >> "$LOG"
}
trap on_exit EXIT
require_tunnel

kx() { # retry kubectl on control-plane drops (pod-ops §12)
  local i; for i in 1 2 3 4; do
    if timeout "$KX_TIMEOUT" kubectl -n "$NS" "$@"; then return 0; fi
    echo "  kubectl retry $i" >&2; sleep 5
  done; return 1
}

# one shot, not kx: a dropped connection here means "unknown" and the run goes on.
# `ray stop` leaves <defunct> raylets behind, so only a non-zombie state counts.
if [ "$GPU_RUN" = 1 ] && timeout 30 kubectl -n "$NS" exec "$POD" -- bash -c "ps -C raylet -o stat= | grep -q '^[^Z]'" >/dev/null 2>&1; then
  echo "a Ray head is running on $POD; GPU suites start their own. Stop the server and the head first:" >&2
  echo "  kubectl -n $NS exec $POD -- bash -c \"pkill -TERM -f 'python3 -m trainin[g]'; sleep 6; ray stop --force\"" >&2
  exit 1
fi

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
# SDK_PYTHONPATH: a candidate tinker SDK (src dir, plus its deps) to shadow the pod's installed one
RC=0
timeout "$KX_TIMEOUT" kubectl -n "$NS" exec "$POD" -- bash -c "cd '$REMOTE/app' && env -u TINKERCLOUD_BACKEND -u RAY_ADDRESS PYTHONPATH='${SDK_PYTHONPATH:+$SDK_PYTHONPATH:}$REMOTE' python -m pytest $(printf '%q ' "$@")" \
  | tee "$TMP/run.log" || RC=$?
# the verdict is pytest's summary line ("N passed, M skipped in ...", "N tests collected", "no tests ran", "ERROR: ...")
VERDICT="$(grep -E '^(=+ )?([0-9]+ (passed|failed|errors?|skipped|tests? collected|deselected)|no tests ran|ERROR)' "$TMP/run.log" | tail -1 | tr -d '=' | sed 's/^ *//;s/ *$//')"
printf '%s %s %s | %s\n' "$(date -u +%FT%TZ)" "$POD" "$*" "${VERDICT:-no pytest summary (rc $RC)}" >> "$LOG"; LOGGED=1
exit $RC
