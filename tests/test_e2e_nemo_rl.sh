#!/usr/bin/env bash
#
# E2E smoke test for TinkerCloud NeMo RL backend.
#
# Exercises the 7 core operations validated in T051:
#   1. health          - GET  /health
#   2. create_session  - POST /api/v1/create_session
#   3. create_model    - POST /api/v1/create_model  (async → poll)
#   4. forward_backward- POST /api/v1/forward_backward (async → poll)
#   5. optim_step      - POST /api/v1/optim_step    (async → poll)
#   6. save_checkpoint - POST /api/v1/save_weights   (async → poll)
#   7. delete_model    - POST /api/v1/delete_model
#
# Usage (inside container):
#   bash /workspace/tinkercloud/tests/test_e2e_nemo_rl.sh
#
# Usage (from host):
#   docker exec tinkercloud-nemo bash /workspace/tinkercloud/tests/test_e2e_nemo_rl.sh
#
# Environment variables:
#   BASE_URL         - TinkerCloud URL       (default: http://localhost:8000)
#   API_KEY          - Tinker API key         (default: slime-dev-key)
#   MODEL_PATH       - HuggingFace model path (default: Qwen/Qwen2.5-0.5B)
#   SEQ_LEN          - Sequence length        (default: 64)
#   POLL_INTERVAL    - Seconds between polls  (default: 2)
#   POLL_TIMEOUT     - Max seconds to poll    (default: 300)

set -euo pipefail

# ---------- configuration ----------
BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-slime-dev-key}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B}"
SEQ_LEN="${SEQ_LEN:-16}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
POLL_TIMEOUT="${POLL_TIMEOUT:-300}"

PASS=0
FAIL=0
SESSION_ID=""
MODEL_ID=""

# ---------- helpers ----------
log()  { echo "[$(date +%H:%M:%S)] $*"; }
pass() { PASS=$((PASS + 1)); log "PASS  $1"; }
fail() { FAIL=$((FAIL + 1)); log "FAIL  $1 — $2"; }

# POST JSON and return body. Sets $HTTP_CODE.
post() {
    local url="$1" body="$2"
    local tmp
    tmp=$(mktemp)
    HTTP_CODE=$(curl -s -o "$tmp" -w "%{http_code}" \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -H "x-api-key: ${API_KEY}" \
        -d "$body")
    RESPONSE=$(cat "$tmp")
    rm -f "$tmp"
}

# GET and return body. Sets $HTTP_CODE.
get() {
    local url="$1"
    local tmp
    tmp=$(mktemp)
    HTTP_CODE=$(curl -s -o "$tmp" -w "%{http_code}" \
        -X GET "$url" \
        -H "x-api-key: ${API_KEY}")
    RESPONSE=$(cat "$tmp")
    rm -f "$tmp"
}

# Extract JSON field (requires jq).
jval() { echo "$RESPONSE" | jq -r "$1"; }

# Poll retrieve_future until completed or timeout.
poll_future() {
    local request_id="$1" op_name="$2"
    local elapsed=0
    while [ "$elapsed" -lt "$POLL_TIMEOUT" ]; do
        post "${BASE_URL}/api/v1/retrieve_future/${request_id}" '{}'
        if [ "$HTTP_CODE" = "200" ]; then
            return 0
        elif [ "$HTTP_CODE" = "408" ]; then
            sleep "$POLL_INTERVAL"
            elapsed=$((elapsed + POLL_INTERVAL))
        else
            fail "$op_name" "retrieve_future returned HTTP ${HTTP_CODE}: ${RESPONSE}"
            return 1
        fi
    done
    fail "$op_name" "timed out after ${POLL_TIMEOUT}s"
    return 1
}

# Generate a fake TensorData blob of given length (all zeros).
tensor_data() {
    local len="$1"
    local data
    data=$(python3 -c "print(','.join(['0.0']*${len}))")
    echo "{\"data\":[${data}],\"shape\":[${len}],\"dtype\":\"float32\"}"
}

# Generate a fake token list of given length (all token-id 1).
token_list() {
    local len="$1"
    python3 -c "print(','.join(['1']*${len}))"
}

# ---------- banner ----------
echo "============================================"
echo " TinkerCloud NeMo RL — E2E Smoke Test"
echo "============================================"
log "BASE_URL   = ${BASE_URL}"
log "MODEL_PATH = ${MODEL_PATH}"
log "SEQ_LEN    = ${SEQ_LEN}"
echo ""

# ========== 1. Health ==========
log "--- 1/7 Health ---"
get "${BASE_URL}/health"
if [ "$HTTP_CODE" = "200" ]; then
    pass "health"
else
    fail "health" "HTTP ${HTTP_CODE}"
fi

# ========== 2. Create Session ==========
log "--- 2/7 Create Session ---"
post "${BASE_URL}/api/v1/create_session" \
    '{"tags":["e2e-test"],"sdk_version":"test-1.0"}'
if [ "$HTTP_CODE" = "200" ]; then
    SESSION_ID=$(jval '.session_id')
    pass "create_session (session_id=${SESSION_ID})"
else
    fail "create_session" "HTTP ${HTTP_CODE}: ${RESPONSE}"
    echo "Cannot continue without session. Aborting."
    exit 1
fi

# ========== 3. Create Model (async) ==========
log "--- 3/7 Create Model ---"
post "${BASE_URL}/api/v1/create_model" "$(cat <<EOF
{
    "session_id": "${SESSION_ID}",
    "model_seq_id": 1,
    "base_model": "${MODEL_PATH}",
    "debug_train_only": true,
    "parallelism_config": {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "num_gpus": 4
    },
    "max_seq_len": ${SEQ_LEN}
}
EOF
)"
if [ "$HTTP_CODE" = "200" ]; then
    REQUEST_ID=$(jval '.request_id')
    MODEL_ID=$(jval '.model_id')
    log "  request_id=${REQUEST_ID}, model_id=${MODEL_ID}"
    if poll_future "$REQUEST_ID" "create_model"; then
        pass "create_model (model_id=${MODEL_ID})"
    fi
else
    fail "create_model" "HTTP ${HTTP_CODE}: ${RESPONSE}"
    echo "Cannot continue without model. Aborting."
    exit 1
fi

# ========== 4. Forward-Backward (async) ==========
log "--- 4/7 Forward-Backward ---"

# Build the datum payload
TOKENS=$(token_list "$SEQ_LEN")
ADVANTAGES=$(tensor_data "$SEQ_LEN")
LOGPROBS=$(tensor_data "$SEQ_LEN")
MASK=$(tensor_data "$SEQ_LEN")
TARGET_TOKENS="{\"data\":[${TOKENS}],\"shape\":[${SEQ_LEN}],\"dtype\":\"int64\"}"

post "${BASE_URL}/api/v1/forward_backward" "$(cat <<EOF
{
    "model_id": "${MODEL_ID}",
    "forward_backward_input": {
        "data": [
            {
                "model_input": {"tokens": [${TOKENS}]},
                "loss_fn_inputs": {
                    "target_tokens": ${TARGET_TOKENS},
                    "logprobs": ${LOGPROBS},
                    "advantages": ${ADVANTAGES},
                    "mask": ${MASK}
                }
            }
        ],
        "loss_fn": "ppo_loss"
    }
}
EOF
)"
if [ "$HTTP_CODE" = "200" ]; then
    REQUEST_ID=$(jval '.request_id')
    log "  request_id=${REQUEST_ID}"
    if poll_future "$REQUEST_ID" "forward_backward"; then
        DEFERRED=$(jval '.deferred // empty')
        log "  result: deferred=${DEFERRED:-false}"
        pass "forward_backward"
    fi
else
    fail "forward_backward" "HTTP ${HTTP_CODE}: ${RESPONSE}"
fi

# ========== 5. Optim Step (async) ==========
log "--- 5/7 Optim Step ---"
post "${BASE_URL}/api/v1/optim_step" "$(cat <<EOF
{
    "model_id": "${MODEL_ID}"
}
EOF
)"
if [ "$HTTP_CODE" = "200" ]; then
    REQUEST_ID=$(jval '.request_id')
    log "  request_id=${REQUEST_ID}"
    if poll_future "$REQUEST_ID" "optim_step"; then
        LOSS=$(jval '.metrics.total_loss // .total_loss // .loss // "N/A"')
        GRAD=$(jval '.metrics.grad_norm // .grad_norm // "N/A"')
        log "  result: loss=${LOSS}, grad_norm=${GRAD}"
        pass "optim_step"
    fi
else
    fail "optim_step" "HTTP ${HTTP_CODE}: ${RESPONSE}"
fi

# ========== 6. Save Checkpoint (async) ==========
log "--- 6/7 Save Checkpoint ---"
post "${BASE_URL}/api/v1/save_weights" "$(cat <<EOF
{
    "model_id": "${MODEL_ID}",
    "path": "e2e-test-checkpoint"
}
EOF
)"
if [ "$HTTP_CODE" = "200" ]; then
    REQUEST_ID=$(jval '.request_id')
    log "  request_id=${REQUEST_ID}"
    if poll_future "$REQUEST_ID" "save_checkpoint"; then
        pass "save_checkpoint"
    fi
else
    fail "save_checkpoint" "HTTP ${HTTP_CODE}: ${RESPONSE}"
fi

# ========== 7. Delete Model ==========
log "--- 7/7 Delete Model ---"
post "${BASE_URL}/api/v1/delete_model" "$(cat <<EOF
{
    "model_id": "${MODEL_ID}"
}
EOF
)"
if [ "$HTTP_CODE" = "200" ]; then
    pass "delete_model"
else
    fail "delete_model" "HTTP ${HTTP_CODE}: ${RESPONSE}"
fi

# ========== Summary ==========
echo ""
echo "============================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "============================================"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
