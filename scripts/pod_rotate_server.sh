#!/usr/bin/env bash
# Rotate the TinkerCloud API server on a pod: kill server -> drain GPUs ->
# boot with the given env -> verify bind in the NEW server's own log.
#
#   pod_rotate_server.sh <server-log-basename> [KEY=VAL ...]
#
# e.g. pod_rotate_server.sh q4b_S_guard_server \
#        TINKERCLOUD_MILES_MULTILORA_SLOTS=8 TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES=2048
#
# Hard-won rotation rules encoded: bracketed launch-flag-agnostic pkill in
# its own exec, server killed before Ray is touched, sglang stragglers
# killed explicitly, bind verified in the NEW server's own log (never by
# curl alone). Env: KUBECONFIG, NS, POD, GPUS overridable.
set -euo pipefail

LOG_NAME="${1:?usage: pod_rotate_server.sh <server-log-basename> [KEY=VAL ...]}"
shift
EXTRA_ENV=("$@")

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/cluster.sh"   # KUBECONFIG, NS, DEFAULT_NODE
POD="${POD:-tinkercloud-miles}"
GPUS="${GPUS:-4}"
BACKEND="${BACKEND:-miles}"
SERVER_LOG="/data/${LOG_NAME}.log"

EX() { kubectl -n "$NS" exec "$POD" -- bash -c "$1"; }
EX_TIMEOUT() { timeout "$1" kubectl -n "$NS" exec "$POD" -- bash -c "$2" || true; }

# 1. kill the server (own exec; launch-flag-agnostic bracketed pattern)
echo "==> killing server"
EX "pkill -f 'm trainin[g]' || true; sleep 8"

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

# 3. Ray head must show 0 GPU in use (leaked PGs = footgun 12) — else recycle it
RAY_OK=$(EX "ray status 2>/dev/null | grep -Eo '[0-9.]+/[0-9.]+ GPU' | head -1" || true)
case "$RAY_OK" in
  0.0/*) : ;;
  *)
    echo "==> Ray head unhealthy ($RAY_OK); recycling"
    EX "ray stop --force >/dev/null 2>&1 || true; rm -rf /tmp/ray; ray start --head --num-gpus $GPUS --disable-usage-stats > /tmp/ray_start.log 2>&1 < /dev/null; sleep 5" ;;
esac

# 4. boot the new server (detached; all fds redirected; timeout-wrapped exec)
echo "==> booting server ($SERVER_LOG)"
ENV_LINES=""
for kv in "${EXTRA_ENV[@]:-}"; do
  [ -n "$kv" ] && ENV_LINES+="export $kv"$'\n'
done
EX "rm -f $SERVER_LOG"
EX_TIMEOUT 30 "
export PYTHONPATH=/app NUM_GPUS=$GPUS RAY_ADDRESS=ray://localhost:10001
export HF_TOKEN=\$(cat /tmp/hf_token.txt) HF_HOME=/data/.cache/huggingface
export TINKER_API_KEY=tml-dev-key TINKERCLOUD_BACKEND=$BACKEND ALLOW_PARTIAL_BATCHES=true
$ENV_LINES
cd /app && setsid nohup python3 -m training > $SERVER_LOG 2>&1 < /dev/null &
sleep 2; echo LAUNCHED"

# 5. verify in the NEW log (a curl-200 alone can be a zombie on :8000)
echo "==> verifying bind"
for _ in $(seq 1 20); do
  if EX "grep -q 'error while attempting to bind' $SERVER_LOG 2>/dev/null"; then
    echo "ERROR: bind failed — a zombie server still owns :8000"; EX "tail -5 $SERVER_LOG"; exit 1
  fi
  if EX "grep -q 'Uvicorn running' $SERVER_LOG 2>/dev/null"; then
    CODE=$(EX "curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/health" || echo 000)
    if [ "$CODE" = 200 ]; then echo "==> server healthy ($SERVER_LOG)"; exit 0; fi
  fi
  sleep 5
done
echo "ERROR: server did not become healthy; last log lines:"; EX "tail -20 $SERVER_LOG"
exit 1
