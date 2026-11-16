#!/usr/bin/env bash
# Run the CPU-only pytest suites (tests/unit, and tests/conformance on the fake
# backend) on a cluster pod against the LOCAL working tree, without touching the
# deployed /app.
#
#   scripts/pod_pytest.sh [pod] [pytest args...]
#   scripts/pod_pytest.sh tinkercloud-nemorl tests/conformance tests/unit/core -q
#
# The tree is tarred (training/ tests/ gates/ pyproject.toml, and the rotate script a test reads), pushed with
# sha256 verified on both sides, unpacked under /tmp/pytest-<user>-<run>/app (one directory
# per run, removed after it, so two runs on one pod never share a tree) and
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
# A run that passes while pytest skipped a file at collection exits 86 (outside pytest's 0 to 5 and
# timeout's 124): a skipped file is a run that did not happen. A -q run is not checked (pytest prints no
# count), and its log verdict says so; the default suites run with -q.
#
# Every run appends `date pod args | verdict` to ~/.cache/tinkercloud/pod_pytest.log,
# also one that stops before pytest (tunnel down, push failed, Ray head up).
set -euo pipefail
# shellcheck source=scripts/lib/cluster.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, kx, require_tunnel, refuse_if_gpus_busy, EX_TIMEOUT, ray_head, SERVER_STOP
POD="${1:-tinkercloud-nemorl}"; shift || true
if [ $# -eq 0 ]; then
  set -- tests/unit tests/conformance -q -p no:cacheprovider
fi
SKIPPED_FILES_EXIT=86
GPU_RUN=0
case " $* " in
  *" tests/gpu"*) GPU_RUN=1; case " $* " in *" -x "*|*" --maxfail"*) ;; *) set -- "$@" -x ;; esac ;;
esac
# a GPU suite on GPUs a run already holds skews the run or OOMs it
[ "$GPU_RUN" = 1 ] && refuse_if_gpus_busy
# kubectl calls time out at KX_TIMEOUT seconds; a GPU suite needs the long default
KX_TIMEOUT="${KX_TIMEOUT:-$([ "$GPU_RUN" = 1 ] && echo 1800 || echo 600)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HOME/.cache/tinkercloud/pod_pytest.log"; mkdir -p "$(dirname "$LOG")"
LOGGED=0; ARGS="$*"   # the trap may run inside a function, where $* is that function's
TMP="$(mktemp -d)"
REMOTE="/tmp/pytest-${USER:-u}-${TMP##*.}"   # mktemp's random suffix names this run
PUSHED=0
# every invocation leaves one log line, also one that dies before pytest (tunnel down, push failed, Ray guard)
# shellcheck disable=SC2329  # invoked by the trap below
on_exit() {
  local rc=$?
  rm -rf "$TMP"
  # A GPU run that fails or is cut off can leave its pytest alive on the GPUs (seen: one that waited
  # on a pass that had run out of memory until its own 25-minute timeout, while the next run was
  # refused). Its processes are the ones whose working directory is this run's tree. The head is
  # taken to be the run's own: the guard below saw none before the push, and one started since by
  # someone else is stopped with it.
  if [ "$PUSHED" = 1 ] && [ "$GPU_RUN" = 1 ] && [ "$rc" != 0 ]; then
    { kx exec "$POD" -- bash -c "for p in /proc/[0-9]*; do [ \"\$(readlink \$p/cwd 2>/dev/null)\" = '$REMOTE/app' ] && kill -9 \${p##*/}; done; true" \
        && EX_TIMEOUT 60 "$(ray_head stop)"; } >&2 || echo "  could not clean up the failed GPU run on $POD" >&2
  fi
  # a pod we never reached has nothing to remove; a failed removal leaves one tree in /tmp
  [ "$PUSHED" = 0 ] || kx exec "$POD" -- rm -rf "$REMOTE" "$REMOTE.tgz" || echo "  could not remove $POD:$REMOTE" >&2
  [ "$LOGGED" = 1 ] || printf '%s %s %s | no run (exit %s before pytest)\n' "$(date -u +%FT%TZ)" "$POD" "$ARGS" "$rc" >> "$LOG"
}
trap on_exit EXIT
require_tunnel

# GPU suites start their own head, so none may run; a state that cannot be read stops the run.
if [ "$GPU_RUN" = 1 ]; then
  HEAD_STATE=$(EX_TIMEOUT 30 "$(ray_head state)")
  case "$HEAD_STATE" in
    none) ;;
    "") echo "ERROR: could not read the Ray head's state on $POD (the exec gave no output)" >&2; exit 1 ;;
    *)
      echo "the Ray head on $POD is '$HEAD_STATE'; GPU suites start their own. Stop the server and the head first:" >&2
      printf '  kubectl -n %s exec %s -- bash -c %q\n' "$NS" "$POD" "$SERVER_STOP; ray stop --force" >&2
      exit 1 ;;
  esac
fi

tar -C "$ROOT" -czf "$TMP/code.tgz" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    training tests gates pyproject.toml scripts/pod_rotate_server.sh
SHA="$(sha256sum "$TMP/code.tgz" | cut -d' ' -f1)"
PUSHED=1
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
VERDICT="$(grep -E '^(=+ )?([0-9]+ (passed|failed|errors?|skipped|tests? collected|deselected)|no tests ran|ERROR)' "$TMP/run.log" | tail -1 | tr -d '=' | sed 's/^ *//;s/ *$//' || true)"
# A file skipped at collection (a module it imports is missing on the pod, or a module-level skip) did
# not run, and pytest still exits 0. Its count is on pytest's collection line, "collected 14 items / 5
# skipped", also behind "collecting ... " (-v) and "/ 3 deselected" (-k, -m): pytest's
# TerminalReporter.report_collect. -q drops that line, so a quiet run cannot be checked and says so.
COLLECTED="$(grep -E 'collected [0-9]+ items?' "$TMP/run.log" | tail -1 || true)"
if [ "$RC" = 0 ] && [ -z "$COLLECTED" ]; then
  VERDICT="${VERDICT:-no pytest summary}; files skipped at collection not checked (-q hides pytest's count)"
elif [ "$RC" = 0 ]; then
  NOT_RUN="$(sed -n -E 's/.*\/ ([0-9]+) skipped.*/\1/p' <<< "$COLLECTED")"
  if [ -n "$NOT_RUN" ]; then
    echo "ERROR: $NOT_RUN test file(s) were skipped at collection on $POD and did not run (exit $SKIPPED_FILES_EXIT)." >&2
    echo "  Add -rs to see each reason. Run them on a pod where they collect, or leave each out by name (--ignore=<file>)." >&2
    VERDICT="$NOT_RUN file(s) skipped at collection; ${VERDICT:-no pytest summary}"
    RC=$SKIPPED_FILES_EXIT
  fi
fi
printf '%s %s %s | %s\n' "$(date -u +%FT%TZ)" "$POD" "$*" "${VERDICT:-no pytest summary (rc $RC)}" >> "$LOG"; LOGGED=1
exit $RC
