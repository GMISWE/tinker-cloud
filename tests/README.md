# opentinker-miles Tests

Integration tests for opentinker-miles server running in Docker.

## Prerequisites

1. **opentinker-miles server running:**
   ```bash
   cd /root/gavin/opentinker-miles
   ALLOW_PARTIAL_BATCHES=true \
   PYTHONPATH=/root/gavin/opentinker-miles:/root/Megatron-LM:/root/miles:$PYTHONPATH \
   python -m uvicorn training.api:app --host 0.0.0.0 --port 8000
   ```

2. **Model available:**
   ```bash
   # Download Qwen2.5-0.5B-Instruct
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
       --local-dir /data/models/Qwen2.5-0.5B-Instruct

   # Convert to torch_dist format (if needed for other tests)
   python /root/gavin/miles/tools/convert_hf_to_torch_dist.py \
       --model /data/models/Qwen2.5-0.5B-Instruct \
       --output /data/models/Qwen2.5-0.5B-Instruct_torch_dist \
       --model-args "--swiglu --num-layers 24 --hidden-size 896 ..."
   ```

3. **tinker-cookbook installed:**
   ```bash
   cd /root/gavin/tinker-cookbook && pip install -e .
   ```

## Running Tests

### Cleanup (always run before tests)
```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=slime-dev-key \
    python tests/cleanup_test_env.py
```

### DPO Tests

**Shell script (quick):**
```bash
# Run 3 steps (default)
./tests/test_dpo_reduced.sh

# Run specific number of steps
./tests/test_dpo_reduced.sh 5
```

**Python (pytest compatible):**
```bash
# Run all DPO tests
pytest tests/test_dpo.py -v

# Run specific test
pytest tests/test_dpo.py::test_dpo_reduced -v

# Run directly
python tests/test_dpo.py --test reduced
python tests/test_dpo.py --test all
```

### Other Tests

```bash
# Health check
pytest tests/test_health.py -v

# Model creation
pytest tests/test_model_creation.py -v

# Full HTTP API test
pytest tests/test_gmi_http.py -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINKER_BASE_URL` | `http://localhost:8000` | opentinker-miles server URL |
| `TINKER_API_KEY` | `slime-dev-key` | API key for authentication |
| `TEST_MODEL_PATH` | `/data/models/Qwen2.5-0.5B-Instruct` | Path to HF model |

## Test Files

| File | Description |
|------|-------------|
| `cleanup_test_env.py` | Cleanup script to free GPUs before tests |
| `test_dpo.py` | DPO training integration tests |
| `test_dpo_reduced.sh` | Shell script for quick DPO test |
| `test_health.py` | Server health check tests |
| `test_model_creation.py` | Model loading tests |
| `test_gmi_http.py` | Full HTTP API tests |
| `test_advantage_computation.py` | Advantage calculation unit tests |
| `test_kgateway_training.py` | Training flow tests |
