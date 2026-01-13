import asyncio
from tinker_cookbook import model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train

model_name = 'Qwen/Qwen2.5-7B-Instruct'
renderer_name = model_info.get_recommended_renderer_name(model_name)

# Batch sizes optimized for GB200/H200 (based on rlve-4xGB200.sh)
# GB200 script uses: rollout_batch_size=128, n_samples_per_prompt=16
# For tinker-cookbook: batch_size * group_size = samples per step
# Start with batch_size=16, group_size=8 = 128 samples (conservative)
builder = Gsm8kDatasetBuilder(
    batch_size=16,    # number of prompts per batch (increased from 2)
    group_size=8,     # samples per prompt (increased from 4)
    renderer_name=renderer_name,
    model_name_for_tokenizer=model_name,
)

config = train.Config(
    model_name=model_name,
    log_path='/tmp/rlve-test',
    dataset_builder=builder,
    learning_rate=1e-5,
    max_tokens=512,   # max response length for sampling
    eval_every=0,
    save_every=0,
    base_url='http://localhost:8000',
    wandb_project='rlve-kgateway',
    wandb_name='qwen2.5-7b-gsm8k-batch128',
)

asyncio.run(train.main(config))
