# Data Preparation Scripts

Scripts for preparing models and datasets for RLVE training on the OpenTinker-Miles stack.

## Quick Start

```bash
# Full setup: download models, convert to Megatron format, download datasets
./setup_all.sh

# Or step by step:
./prepare_weights.sh      # Download HuggingFace models
./convert_weights.sh      # Convert to Megatron torch_dist format
./prepare_datasets.sh     # Download training datasets
```

## Scripts

### prepare_weights.sh

Download HuggingFace models to `/data/models/`.

```bash
# Download all default models
./prepare_weights.sh

# Download specific model
./prepare_weights.sh Qwen/Qwen2.5-7B-Instruct
./prepare_weights.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

**Default models:**
- Qwen/Qwen2.5-0.5B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen3-4B-Instruct-2507
- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- meta-llama/Llama-3.2-1B
- Qwen/Qwen3-8B-Base

### convert_weights.sh

Convert HuggingFace models to Megatron torch_dist format (required for Miles training).

```bash
# Convert all unconverted models
./convert_weights.sh

# Convert specific model
./convert_weights.sh Qwen2.5-7B-Instruct
./convert_weights.sh /data/models/Qwen2.5-7B-Instruct
```

**Supported architectures:**
- Qwen 2.5 family (0.5B, 1.5B, 3B, 7B, 32B)
- Qwen 3 family (0.6B, 1.7B, 4B, 8B, 14B, 32B)
- DeepSeek-R1-Distill-Qwen-1.5B
- Llama 3.2 family (1B, 3B)
- Llama 3.1 (8B)

### prepare_datasets.sh

Download training datasets to `/data/datasets/`.

```bash
# Download all default datasets
./prepare_datasets.sh

# Download specific dataset
./prepare_datasets.sh gsm8k
./prepare_datasets.sh math
```

**Available datasets:**
- gsm8k (openai/gsm8k)
- math (hendrycks/competition_math)
- metamath (meta-math/MetaMathQA)
- openmath (nvidia/OpenMathInstruct-2)

### setup_all.sh

Combined setup script with options.

```bash
./setup_all.sh                  # Full setup
./setup_all.sh --models-only    # Only models
./setup_all.sh --datasets-only  # Only datasets
./setup_all.sh --skip-convert   # Download without conversion
```

## Directory Structure

```
/data/
├── models/
│   ├── Qwen2.5-7B-Instruct/           # HuggingFace format
│   ├── Qwen2.5-7B-Instruct_torch_dist/ # Megatron format
│   └── ...
├── datasets/
│   ├── gsm8k/
│   ├── math/
│   └── ...
├── checkpoints/                        # Training checkpoints
└── logs/                               # Training logs
```

## Environment Variables

- `MILES_DIR` - Miles installation path (default: /root/gavin/miles)
- `MEGATRON_DIR` - Megatron-LM path (default: /root/Megatron-LM)
- `HF_TOKEN` - HuggingFace token for gated models

## Adding New Models

To add support for a new model architecture, edit `convert_weights.sh`:

1. Add a model script in `/root/gavin/miles/scripts/models/` with `MODEL_ARGS`
2. Add entry to `MODEL_CONFIGS` array:
   ```bash
   "ModelPattern|script-name.sh|extra-args"
   ```

Example for a new Qwen model:
```bash
"Qwen2.5-14B|qwen2.5-14B.sh|--untie-embeddings-and-output-weights"
```
