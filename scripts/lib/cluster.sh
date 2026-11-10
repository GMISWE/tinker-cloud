# shellcheck shell=bash
# Cluster defaults shared by every script that drives a pod. One place to
# change when the cluster moves. KUBECONFIG, NS and DEFAULT_NODE are overridable from
# the env; every other value is fixed here.
export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/h1.config}"
NS="${NS:-tinkercloud-nemorl}"
DEFAULT_NODE="${DEFAULT_NODE:-hgxcn13}"
# Pod label naming a pod's trust domain: its NetworkPolicy admits, on every port, the pods
# that carry the same value. A single-pod deploy gives it the pod's own name.
# shellcheck disable=SC2034  # read by the scripts that source this file
TRUST_DOMAIN_LABEL=tinkercloud.trust-domain

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
EX() { kubectl -n "$NS" exec "$POD" -- bash -c "$(pod_cmd "$1")"; }
EX_TIMEOUT() { timeout "$1" kubectl -n "$NS" exec "$POD" -- bash -c "$(pod_cmd "$2")" || true; }

# Ray is members-only: every Ray process of a deployment runs in token mode with the one
# token file below (owner-only, never on a command line). This block is the one place that
# knows the file, the mode, how an exec helper's command gets them and how a head starts.
RAY_AUTH_TOKEN_FILE=/data/ray_auth_token
RAY_AUTH_EXPORT="export RAY_AUTH_MODE=token RAY_AUTH_TOKEN_PATH=$RAY_AUTH_TOKEN_FILE"
# Prefixed to every command the exec helpers here and in the deploy run: a pod that holds the
# token runs its Ray clients (ray status, the server, drivers) in token mode. Without the file
# they run without a token, which is right only while the head is absent or open (an earlier
# deploy's); a token-mode head without its file reads as `unusable` below and stops the deploy.
# pod_pytest.sh's runs carry no token: the CPU suites start no Ray, and the GPU suites run with
# the deployed head stopped.
RAY_AUTH_PRELUDE="[ -s $RAY_AUTH_TOKEN_FILE ] && $RAY_AUTH_EXPORT;"
# pod_cmd <command>: the command as every exec helper runs it on a pod.
pod_cmd() { printf '%s %s' "$RAY_AUTH_PRELUDE" "$1"; }
# The port `ray start` gives its Ray Client server when none is named.
RAY_CLIENT_PORT=10001
# The RAY_ADDRESS every backend's server takes: a driver on the pod's node, never a Ray Client.
RAY_DRIVER_ADDRESS=auto
# The API server's port: each boot exports it as TRAINING_PORT; the container, the NetworkPolicy
# and the health checks read it here.
API_PORT=8000
# The API server's process as `pgrep -f`/`pkill -f` (ERE) match it: a command line that starts with
# a python of any name or path, with single-token interpreter flags (`-u`, `-Xdev`), running
# `-m training` with or without arguments. The anchor leaves out the `bash -c "... python3 -m
# training ..."` launcher, which stays alive as the server's parent, and every pgrep, pkill or
# shell whose command line holds the pattern.
SERVER_PATTERN='^([^ ]*/)?python[0-9.]*( -[^ ]+)* -m training( |$)'
# The pod command that stops the API server, if one runs, and fails unless none runs after it.
# pkill's exit cannot tell the cases apart (1 is both "nothing matched" and "a match could not be
# signalled"), so the check is pgrep after the wait: 1 (nothing matches) passes; 0 (a server
# outlived the TERM) or 2 (a bad pattern) fails.
# shellcheck disable=SC2034  # read by the scripts that source this file
SERVER_STOP="pkill -f '$SERVER_PATTERN'; sleep 8; pgrep -f '$SERVER_PATTERN' > /dev/null; [ \$? -eq 1 ] || { echo 'ERROR: an API server still runs after pkill, or pgrep failed'; exit 1; }"
# The HF token file on a pod, and the pod command that fails unless it is non-empty. The deploy
# and the rotate run it before they stop the running server, since a refusal inside the boot exec
# cannot fail its caller (EX_TIMEOUT). The boot command runs it again, so a caller that skips
# that check still starts no server without a token.
HF_TOKEN_POD_FILE=/tmp/hf_token.txt
HF_TOKEN_CHECK="[ -s $HF_TOKEN_POD_FILE ] || { echo 'ERROR: $HF_TOKEN_POD_FILE is empty or missing'; exit 1; }"
# The variables every server boot exports, in order (server_boot's case table holds each value).
SERVER_BOOT_KEYS=" PYTHONPATH NUM_GPUS RAY_ADDRESS TRAINING_PORT HF_TOKEN HF_HOME TINKER_API_KEY TINKERCLOUD_BACKEND ALLOW_PARTIAL_BATCHES "
# The bound on the boot's exec: the command detaches the server and returns after `sleep 2`.
# shellcheck disable=SC2034  # read by the scripts that source this file
SERVER_BOOT_TIMEOUT=30
# server_boot <gpus> <backend> <log> [KEY=VAL ...]: prints the pod command that starts the API
# server detached, with SERVER_BOOT_KEYS set and then each KEY=VAL quoted with %q, so a value
# reaches the server unchanged. A KEY=VAL replaces a default (named on stderr), except
# TRAINING_PORT, which is API_PORT above. Refused before anything is printed: TRAINING_PORT=...,
# an argument that is not a plain KEY=VAL, and a <log> that is not an absolute path of plain
# [A-Za-z0-9_./-] characters (callers read the log by its bare path). Run it under EX_TIMEOUT
# $SERVER_BOOT_TIMEOUT and judge the launch by a fresh probe: EX_TIMEOUT drops the exec's exit.
server_boot() {
  local gpus=$1 backend=$2 log=$3 kv key val cmd="$HF_TOKEN_CHECK"$'\n'
  shift 3
  [[ "$log" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "ERROR: log path '$log' is not an absolute plain [A-Za-z0-9_./-] path" >&2; return 1; }
  for key in $SERVER_BOOT_KEYS; do
    case "$key" in
      PYTHONPATH) val=/app ;;
      NUM_GPUS) val=$(printf '%q' "$gpus") ;;
      RAY_ADDRESS) val=$RAY_DRIVER_ADDRESS ;;
      TRAINING_PORT) val=$API_PORT ;;
      HF_TOKEN) val="\$(cat $HF_TOKEN_POD_FILE)" ;;
      HF_HOME) val=/data/.cache/huggingface ;;
      TINKER_API_KEY) val=tml-dev-key ;;
      TINKERCLOUD_BACKEND) val=$(printf '%q' "$backend") ;;
      ALLOW_PARTIAL_BATCHES) val=true ;;
      *) echo "ERROR: SERVER_BOOT_KEYS names $key, which has no default" >&2; return 1 ;;
    esac
    cmd+="export $key=$val"$'\n'
  done
  for kv in "$@"; do
    key=${kv%%=*}
    if [[ "$kv" != *=* || ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "ERROR: '$kv' is not a plain KEY=VAL" >&2; return 1
    fi
    if [ "$key" = TRAINING_PORT ]; then
      echo "ERROR: '$kv': the port is API_PORT in lib/cluster.sh, not a KEY=VAL" >&2; return 1
    fi
    [[ "$SERVER_BOOT_KEYS" == *" $key "* ]] && echo "==> $key=... replaces the default $key" >&2
    cmd+="export $key=$(printf '%q' "${kv#*=}")"$'\n'
  done
  printf '%s\n' \
    "$cmd" \
    "cd /app && setsid nohup python3 -m training > $log 2>&1 < /dev/null &" \
    "sleep 2; echo LAUNCHED"
}
# The pod command that prints the API server's /health status code (curl prints 000 and fails
# when nothing answers).
# shellcheck disable=SC2034  # read by the scripts that source this file
SERVER_HEALTH="curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:$API_PORT/health"

# The pod side of this pod's Ray head: `ray_head state`, `ray_head ensure <gpus>` or
# `ray_head recycle <gpus>`.
# state prints one word: none (no live raylet), open (a head answers a client holding no token: an
# earlier deploy's), token (it refuses that client and answers one holding the token file), or a
# sentence starting `unusable` (a head runs that neither answers).
# ensure leaves a token head as it is, starts one where there is none, replaces an open one, and
# fails on anything else. recycle stops whatever head runs and starts a new one. Both fail
# unless the old head is gone after the stop.
# Starting makes the token when the pod has none, starts a token-mode head and stops its Ray
# Client server: `ray start` starts one whenever grpc imports, on port 10001 unless named, and
# refuses a port outside 1024-65535 (ray/_private/parameter.py, `RayParams._check_usage`), while the head starts it
# only for a non-zero port (ray/_private/node.py, `start_head_processes`). The deploy and rotate
# scripts make every backend's server a driver (RAY_DRIVER_ADDRESS), so none uses it (miles, verl
# and megatron_bridge are untested as drivers). A start fails unless the head started, the port is then
# closed, and the head answers a client holding the token.
read -r -d '' RAY_HEAD_SH <<'EOF' || true
set -uo pipefail
token_file=$1; auth_export=$2; client_port=$3; cmd=$4
with_token() ( eval "$auth_export"; "$@" )
fail() { echo "ERROR: $*"; exit 1; }
# only a raylet not in state Z counts: after `ray stop` on a pod whose PID 1 is `sleep infinity`,
# `ps -eo stat,comm` showed the stopped heads' raylet and gcs_server processes in state Z
live_raylet() { ps -C raylet -o stat= | grep -q '^[^Z]'; }
head_state() {
  local out
  if ! live_raylet; then echo none; return; fi
  if out=$(env -u RAY_AUTH_MODE -u RAY_AUTH_TOKEN_PATH -u RAY_AUTH_TOKEN ray status 2>&1); then echo open; return; fi
  if ! grep -qiE 'InvalidAuthToken|authentication token' <<< "$out"; then
    echo "unusable: ray status without the token failed for a reason other than auth: $(tail -1 <<< "$out")"; return
  fi
  if [ -s "$token_file" ] && with_token ray status > /dev/null 2>&1; then echo token
  else echo "unusable: a Ray head runs that refuses a client without the token, and $token_file does not open it"
  fi
}
stop_head() {
  local out
  out=$(ray stop --force 2>&1)
  for _ in 1 2 3 4 5 6 7 8 9 10; do live_raylet || { rm -rf /tmp/ray; return; }; sleep 1; done
  echo "$out" | tail -5; fail "a raylet still runs 10 s after ray stop"
}
start_head() {   # <gpus>
  local listen
  if [ ! -s "$token_file" ]; then
    (umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$token_file") || fail "could not write $token_file"
    [ -s "$token_file" ] || fail "$token_file is empty after writing it"
  fi
  chmod 600 "$token_file" || fail "could not make $token_file owner-only"
  with_token ray start --head --num-gpus "$1" --disable-usage-stats > /tmp/ray_start.log 2>&1 < /dev/null \
    || { tail -20 /tmp/ray_start.log; fail "ray start failed"; }
  sleep 5
  pkill -f 'ray.util.client.serve[r]'
  sleep 1
  listen=$(ss -ltnH) || fail "ss failed, so the Ray Client port cannot be checked"
  if grep -qE ":${client_port}[[:space:]]" <<< "$listen"; then fail "the Ray Client server still listens on :$client_port"; fi
  with_token ray status > /dev/null 2>&1 || fail "the new head does not answer a client holding the token"
  echo started
}
case $cmd in
  state) head_state ;;
  ensure)
    state=$(head_state)
    case $state in
      token) echo "the head runs with token auth" ;;
      none) start_head "$5" ;;
      open) echo "replacing a Ray head that runs without token auth"; stop_head; start_head "$5" ;;
      *) fail "$state" ;;
    esac ;;
  recycle) stop_head; start_head "$5" ;;
  *) fail "ray_head takes state, ensure <gpus> or recycle <gpus>, not '$cmd'" ;;
esac
EOF
# ray_head <command> [gpus]: the pod-side command line that runs RAY_HEAD_SH.
ray_head() { printf 'bash -c %q ray_head %q %q %q' "$RAY_HEAD_SH" "$RAY_AUTH_TOKEN_FILE" "$RAY_AUTH_EXPORT" "$RAY_CLIENT_PORT"; printf ' %q' "$@"; }

# require_server_env KEY=VAL ...: refuse unless the running API server on $POD has every
# KEY=VAL in its own environment (read from /proc, not from the boot command).
require_server_env() {
  local envs kv
  envs=$(EX "p=\$(pgrep -f '$SERVER_PATTERN' | head -1); [ -n \"\$p\" ] && tr '\0' '\n' < /proc/\$p/environ") \
    || { echo "ERROR: no API server running on $POD" >&2; exit 1; }
  for kv in "$@"; do
    grep -qx -- "$kv" <<< "$envs" || {
      echo "ERROR: the server on $POD lacks $kv (has: $(grep -- "^${kv%%=*}=" <<< "$envs" || echo unset))" >&2
      echo "  rotate it with that setting: scripts/pod_rotate_server.sh <log> $kv ..." >&2
      exit 1; }
  done
}

# pod_log_errors <log on $POD>: the last exception line and the frames of our own code in that log,
# with Ray's escaped "\n" undone; a full traceback is ~40 KB of mostly library frames. An exception line
# starts its line (after an optional Ray "(actor pid=N) " prefix): a logged training sample quoting one does not.
# worker_py <args>: the Megatron policy worker venv's python on $POD (where megatron.core/bridge import),
# stdout and stderr merged, minus the ~40 import-time warning lines (Apex, tvm_ffi, modelopt) every start prints;
# tracebacks pass through and the exit status is python's.
WORKER_PY="${WORKER_PY:-/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python}"
worker_py() {
  EX "PYTHONWARNINGS=ignore $WORKER_PY $* 2>&1 | grep -vE '^\\[WARNING\\] Field|registry\\.py:85|info = core\\._register|Tried to register an operator|Duplicate registration|^  warnings\\.warn'; exit \${PIPESTATUS[0]}"
}

pod_log_errors() {
  EX "python3 - '$1' <<'PYEOF'
import re, sys
s = open(sys.argv[1], errors='replace').read().replace('\\\\n', '\\n')
errs = re.findall(r'(?m)^(?:\([^)\\n]*\) )?[A-Za-z_.]*(?:Error|Exception): [^\\n]{0,300}', s)
if not errs:
    sys.exit()
ours = re.findall(r'File \"([^\"]*(?:/data/runs|tinker_cookbook|nemo_rl|megatron/bridge|/app/training)[^\"]*)\", line (\\d+), in (\\w+)', s)
seen = []
for f in ours:
    if f not in seen and 'site-packages' not in f[0]:
        seen.append(f)
for f, n, fn in seen[-8:]:
    print(f'  {f}:{n} {fn}')
print('ERROR', errs[-1])
PYEOF"
}

# refuse_if_gpus_busy: refuse when any GPU on $POD holds memory (a probe or model is resident);
# FORCE=1 overrides. A test sharing GPUs with a run skews its memory readings or OOMs it.
refuse_if_gpus_busy() {
  local used
  [ "${FORCE:-0}" = 1 ] && return 0
  # an idle H100 with no process on it read 4 MiB (observed 2026-11-06); pod_rotate_server.sh drains to < 500 MiB
  used=$(EX "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1>=500' | wc -l") \
    || { echo "ERROR: could not read GPU memory on $POD" >&2; exit 1; }
  if [ "$used" != 0 ]; then
    echo "ERROR: $used GPU(s) on $POD hold memory; a run is resident (FORCE=1 to share them):" >&2
    EX "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -4" >&2
    exit 1
  fi
}
