# Training API Test Data

This directory contains test datasets used for validating the kgateway Training API integration with Slime.

## gsm8k_rl.jsonl

**Purpose**: Server-side fallback dataset for RL training initialization when using tinker-cookbook.

**Format**: JSONL with GSM8K-style math problems
```json
{"prompt": "math problem text", "response": "step-by-step solution with answer"}
```

**Usage**:
- Slime's RolloutManager requires a dataset path during initialization
- This dataset serves as a fallback when `debug_train_only=False`
- In production, tinker-cookbook uses client-side datasets (HuggingFace)
- Only `tinker.Datum` objects are sent to the server for training

**Deployment**:
1. Copy to slime pod: `/data/datasets/gsm8k_rl.jsonl`
2. Configured in `api.py:403` via `args.prompt_data`

**Test Command**:
```bash
kubectl exec -n slime-gmi slime-training-0 -- bash -c \
  "export TINKER_API_KEY=tml-dev-key && \
   cd /tmp/tinker-cookbook && \
   python3 -u tinker_cookbook/recipes/rl_basic.py \
     base_url=http://kgateway-training.slime-gmi:8000 \
     model_name=/data/models/Qwen2.5-0.5B-Instruct_torch_dist \
     max_tokens=256 \
     log_path=/tmp/tinker-examples/rl_basic_\$(date +%Y%m%d_%H%M%S)"
```

**Dataset Strategy**:
- **Primary**: Client-side datasets (HuggingFace) - used by tinker-cookbook recipes
- **Fallback**: Server-side datasets (JSONL) - required by Slime's RolloutManager initialization
- The server-side dataset is only for initialization; actual training uses client-provided data

**Contents**: 5 GSM8K math problems with step-by-step solutions
