---
layout: default
title: Quickstart
nav_order: 3
---

# Quickstart: Your First Training Run

This guide walks you through a complete training workflow using tinkercloud.

## Prerequisites

- tinkercloud server running on `http://localhost:8000`
- `tinker_gmi` client installed
- Model checkpoint available (e.g., `Qwen2.5-0.5B-Instruct_torch_dist`)

## Step 1: Create a Session

Every training run starts with creating a session:

```python
import tinker

# Create service client
client = tinker.ServiceClient(
    base_url="http://localhost:8000",
    api_key="slime-dev-key"  # or set TINKER_API_KEY env var
)

# Session is created automatically when you create a training client
```

## Step 2: Create a Training Client

This allocates GPUs and initializes the model:

```python
# Create LoRA training client
training_client = await client.create_lora_training_client_async(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    rank=32,                    # LoRA rank
    train_mlp=True,            # Train MLP layers
    train_attn=True,           # Train attention layers
    train_unembed=True,        # Train unembedding layer
)

# Get tokenizer for data preparation
tokenizer = training_client.get_tokenizer()
```

**Note:** Model creation takes 1-2 minutes as it:
1. Builds Megatron arguments from HF config
2. Creates Ray placement groups
3. Initializes RayTrainGroup actors
4. Starts SGLang inference engines

## Step 3: Prepare Training Data

### For Supervised Learning

```python
# Create a training datum
tokens = tokenizer.encode("Hello, how are you?")
target_tokens = tokenizer.encode("I'm doing well, thank you!")

datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData.from_numpy(np.array(target_tokens)),
        "weights": tinker.TensorData.from_numpy(np.ones(len(target_tokens))),
    }
)
```

### For Reinforcement Learning

```python
import torch

# Prepare RL data with advantages
datum = tinker.Datum(
    model_input=tinker.ModelInput.from_ints(full_tokens),
    loss_fn_inputs={
        "target_tokens": tinker.TensorData.from_torch(torch.tensor(response_tokens)),
        "logprobs": tinker.TensorData.from_torch(sampling_logprobs),
        "advantages": tinker.TensorData.from_torch(advantages),
    }
)
```

## Step 4: Run Forward-Backward

Compute gradients on a batch of data:

```python
# Submit forward-backward (non-blocking)
fwd_bwd_future = await training_client.forward_backward_async(
    data=[datum],              # List of Datum objects
    loss_fn="cross_entropy"    # or "importance_sampling" for RL
)

# Wait for result
result = await fwd_bwd_future.result_async()

# Extract metrics
print(f"Loss: {result.metrics.get('loss:sum', 0)}")

# For RL, extract training logprobs
if result.loss_fn_outputs:
    training_logprobs = result.loss_fn_outputs[0]["logprobs"].to_torch()
```

## Step 5: Update Weights

Apply accumulated gradients:

```python
# Submit optimizer step (non-blocking)
optim_future = await training_client.optim_step_async(
    tinker.AdamParams(
        learning_rate=1e-6,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8
    )
)

# Wait for completion
optim_result = await optim_future.result_async()
print(f"Learning rates: {optim_result.learning_rates}")
```

## Step 6: Sample from Updated Model

Get a sampling client with the new weights:

```python
# Save weights and get sampling client
sampling_client = await training_client.save_weights_and_get_sampling_client_async()

# Generate text
prompt = tinker.ModelInput.from_ints(tokenizer.encode("Once upon a time"))
response = await sampling_client.sample_async(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.SamplingParams(
        max_tokens=100,
        temperature=0.7,
        stop=["<|endoftext|>"]
    )
)

# Decode response
generated_tokens = response.sequences[0].tokens
generated_text = tokenizer.decode(generated_tokens)
print(generated_text)
```

## Complete Example: Supervised Learning Loop

```python
import asyncio
import numpy as np
import tinker

async def train():
    # Setup
    client = tinker.ServiceClient(base_url="http://localhost:8000")
    training_client = await client.create_lora_training_client_async(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        rank=32
    )
    tokenizer = training_client.get_tokenizer()

    # Training data
    examples = [
        ("What is 2+2?", "The answer is 4."),
        ("What is the capital of France?", "The capital of France is Paris."),
    ]

    # Training loop
    for epoch in range(3):
        for prompt, response in examples:
            # Prepare data
            input_tokens = tokenizer.encode(prompt)
            target_tokens = tokenizer.encode(response)
            all_tokens = input_tokens + target_tokens

            datum = tinker.Datum(
                model_input=tinker.ModelInput.from_ints(all_tokens),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(
                        np.array(target_tokens, dtype=np.int64)
                    ),
                    "weights": tinker.TensorData.from_numpy(
                        np.ones(len(target_tokens), dtype=np.float32)
                    ),
                }
            )

            # Forward-backward
            fwd_bwd = await training_client.forward_backward_async([datum], "cross_entropy")
            result = await fwd_bwd.result_async()
            print(f"Epoch {epoch}, Loss: {result.metrics.get('loss:sum', 0):.4f}")

            # Optimizer step
            await (await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=1e-5)
            )).result_async()

    # Test generation
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    prompt = tinker.ModelInput.from_ints(tokenizer.encode("What is 2+2?"))
    response = await sampling_client.sample_async(
        prompt, 1, tinker.SamplingParams(max_tokens=50, temperature=0.1)
    )
    print(f"\nGenerated: {tokenizer.decode(response.sequences[0].tokens)}")

if __name__ == "__main__":
    asyncio.run(train())
```

## Performance Tips

### Pipeline Forward-Backward and Optim Step

Submit both operations without waiting:

```python
# Submit forward-backward
fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn)

# Immediately submit optim step (don't wait for fwd_bwd)
optim_future = await training_client.optim_step_async(adam_params)

# Now wait for both
fwd_bwd_result = await fwd_bwd_future.result_async()
optim_result = await optim_future.result_async()
```

### Batch Data Efficiently

Group samples to maximize GPU utilization:

```python
# Good: batch multiple samples
await training_client.forward_backward_async(
    data=[datum1, datum2, datum3, datum4],  # Batch of 4
    loss_fn="cross_entropy"
)

# Avoid: single samples (inefficient)
for datum in [datum1, datum2, datum3, datum4]:
    await training_client.forward_backward_async([datum], loss_fn)
```

## Common Issues

### "Session not found"
- Server was restarted - create a new training client
- Session timed out - call `session_heartbeat()` periodically

### "Model creation timed out"
- Check GPU availability: `nvidia-smi`
- Check Ray status: `ray status`
- View logs: `tail -f /data/logs/tinkercloud.log`

### "CUDA out of memory"
- Reduce batch size
- Use smaller LoRA rank
- Clean up existing sessions before starting

## Next Steps

- [Architecture](architecture.md) - Understand the system design
- [RL Training](cookbook/rl-training.md) - RLVE training guide
- [API Reference](api/endpoints.md) - Full endpoint documentation
