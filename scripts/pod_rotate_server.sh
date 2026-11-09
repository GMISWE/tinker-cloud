#!/usr/bin/env bash
# Rotate the TinkerCloud API server on a pod: kill server -> drain GPUs ->
# boot with the given env -> verify bind in the NEW server's own log.
#
#   pod_rotate_server.sh <server-log-basename> [KEY=VAL ...]
#   ENV_FILE=<file> pod_rotate_server.sh <log>   (KEY=VAL lines, # comments; arguments win)
#
# e.g. pod_rotate_server.sh q4b_S_guard_server \
#        TINKERCLOUD_MILES_MULTILORA_SLOTS=8 TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES=2048
#
# Hard-won rotation rules encoded: the shared SERVER_STOP pkill (lib/cluster.sh) in
# its own exec, server killed before Ray is touched, sglang stragglers
# killed explicitly, bind verified in the NEW server's own log (never by
# curl alone). Env: KUBECONFIG, NS, POD, GPUS overridable.
set -euo pipefail

LOG_NAME="${1:?usage: pod_rotate_server.sh <server-log-basename> [KEY=VAL ...]}"
shift
EXTRA_ENV=()
if [ -n "${ENV_FILE:-}" ]; then
  [ -f "$ENV_FILE" ] || { echo "ERROR: ENV_FILE $ENV_FILE does not exist"; exit 1; }
  mapfile -t EXTRA_ENV < <(grep -vE '^\s*(#|$)' "$ENV_FILE")
fi
EXTRA_ENV+=("$@")

# shellcheck source=scripts/lib/cluster.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, DEFAULT_NODE, EX, EX_TIMEOUT, ray_head, SERVER_STOP, API_PORT, HF_TOKEN_POD_FILE, HF_TOKEN_CHECK, server_boot, SERVER_BOOT_TIMEOUT, SERVER_HEALTH
require_tunnel
# no defaults: a default pod or backend boots the wrong server on the right pod
POD="${POD:?set POD (scripts/pod_state.sh lists the pods)}"
BACKEND="${BACKEND:?set BACKEND (the backend this pod serves, e.g. nemo_rl)}"
GPUS="${GPUS:-4}"
# Every backend's server boots as a Ray driver on the pod's node (RAY_DRIVER_ADDRESS); the head runs
# no Ray Client server (lib/cluster.sh RAY_HEAD_SH). miles, verl and megatron_bridge are untested so.
SERVER_LOG="/data/${LOG_NAME}.log"

# 0. refuse bad input while the running server still serves
BOOT_CMD=$(server_boot "$GPUS" "$BACKEND" "$SERVER_LOG" "${EXTRA_ENV[@]}")
EX "$HF_TOKEN_CHECK" || { echo "ERROR: no boot without a non-empty $HF_TOKEN_POD_FILE on $POD (or the exec failed)"; exit 1; }

# 1. kill the server (own exec)
echo "==> killing server"
EX "$SERVER_STOP"

# 2. drain GPUs; after 60s kill sglang stragglers (they survive server death
#    holding ~24GB and wedge the next create_model)
echo "==> draining GPUs"
DRAINED=0
for i in $(seq 1 24); do
  USED=$(EX "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=\$1} END {print s}'")
  if [ "${USED:-99999}" -lt 500 ]; then DRAINED=1; break; fi
  [ "$i" = 12 ] && EX "pkill -9 -f 'sglan[g]::' || true"
  sleep 5
done
if [ "$DRAINED" != 1 ]; then
  echo "ERROR: GPUs did not drain:"; EX "nvidia-smi --query-gpu=index,memory.used --format=csv"
  exit 1
fi

# 3. Ray head must show 0 GPU in use (leaked PGs = footgun 12) out of $GPUS
#    (a head started with another --num-gpus gives the server the wrong
#    layout with no error) — else recycle it
#    `ray_head ensure` keeps a token head, starts or replaces the others and fails otherwise.
EX "$(ray_head ensure "$GPUS")"
RAY_GPUS=$(EX "ray status 2>/dev/null | grep -Eo '[0-9.]+/[0-9.]+ GPU' | head -1") \
  || { echo "ERROR: could not read the Ray head's GPUs on $POD"; exit 1; }
if [ "$RAY_GPUS" != "0.0/$GPUS.0 GPU" ]; then
  echo "==> Ray head is not idle on $GPUS GPUs (${RAY_GPUS:-ray status printed no GPU line}); recycling"
  EX "$(ray_head recycle "$GPUS")"
fi

# 4. boot the new server
echo "==> booting server ($SERVER_LOG)"
EX "rm -f $SERVER_LOG"
EX_TIMEOUT "$SERVER_BOOT_TIMEOUT" "$BOOT_CMD"

# 5. verify in the NEW log (a curl-200 alone can be a zombie on :$API_PORT)
echo "==> verifying bind"
for _ in $(seq 1 20); do
  if EX "grep -qE 'Address already in use|error while attempting to bind' $SERVER_LOG 2>/dev/null"; then
    echo "ERROR: bind failed — a zombie server still owns :$API_PORT"; EX "tail -5 $SERVER_LOG"; exit 1
  fi
  # the server's log carries "Application startup complete" but no "Uvicorn running" line
  if EX "grep -q 'Application startup complete' $SERVER_LOG 2>/dev/null"; then
    CODE=$(EX "$SERVER_HEALTH") || true
    if [ "$CODE" = 200 ]; then echo "==> server healthy ($SERVER_LOG)"; exit 0; fi
  fi
  sleep 5
done
echo "ERROR: server did not become healthy; last log lines:"; EX "tail -20 $SERVER_LOG"
exit 1
