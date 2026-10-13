#!/bin/bash
# Automodel image smoke: the training env writes a LoRA adapter, the vLLM env
# serves the base model, hot-loads the adapter under a versioned name, and
# generates with both. PASS needs the adapter output to differ from the base.
#   bash smoke_vllm_lora.sh [model] [gpu]   (output dir: $OUT, default /data/smoke_automodel)
set -euo pipefail
M=${1:-Qwen/Qwen2.5-0.5B-Instruct}
GPU=${2:-0}
OUT=${OUT:-/data/smoke_automodel}
PORT=${PORT:-8100}
URL=http://127.0.0.1:$PORT
export HF_HOME=${HF_HOME:-/data/.cache/huggingface}
mkdir -p "$OUT"

# 1. training env: write a PEFT LoRA adapter with non-zero B
/opt/venv/bin/python - "$M" "$OUT/adapter_v1" <<'PY'
import sys
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(snapshot_download(sys.argv[1]), dtype=torch.bfloat16)
peft = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
torch.manual_seed(0)
for name, w in peft.named_parameters():
    if "lora_B" in name:
        torch.nn.init.normal_(w, std=0.02)
peft.save_pretrained(sys.argv[2])
print("ADAPTER_OK")
PY

# 2. vLLM env: serve the base with runtime LoRA
CUDA_VISIBLE_DEVICES=$GPU /opt/vllm-env/bin/vllm-api-server --model "$M" --port "$PORT" \
  --enable-lora --max-lora-rank 16 --max-loras 2 --max-cpu-loras 4 \
  --gpu-memory-utilization 0.5 --max-model-len 4096 > "$OUT/vllm.log" 2>&1 < /dev/null &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf "$URL/health" > /dev/null && break; sleep 5; done
curl -sf "$URL/health" > /dev/null || { echo "SERVER_FAIL (see $OUT/vllm.log)"; exit 1; }
echo SERVER_OK

# 3. hot-load the adapter as <model_id>@v<k>, then generate with base and adapter
curl -sf -X POST "$URL/v1/load_lora_adapter" -H 'Content-Type: application/json' \
  -d "{\"lora_name\": \"m1@v1\", \"lora_path\": \"$OUT/adapter_v1\"}"
echo
gen() {
  curl -sf "$URL/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$1\", \"prompt\": \"The capital of France is\", \"max_tokens\": 12, \"temperature\": 0}" \
    | /opt/vllm-env/bin/python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}
BASE=$(gen "$M")
LORA=$(gen m1@v1)
echo "GEN base $(printf %q "$BASE")"
echo "GEN m1@v1 $(printf %q "$LORA")"
[ "$BASE" != "$LORA" ] || { echo "FAIL: adapter output equals base output"; exit 1; }
echo PASS
