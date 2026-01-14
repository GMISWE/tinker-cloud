# RLVE Training Cheatsheet

## Docker Container Setup

```bash
# Start tinkercloud container
docker stop tinkercloud-rlve 2>/dev/null; docker rm tinkercloud-rlve 2>/dev/null
docker run -d --name tinkercloud-rlve --gpus all \
  -v /data:/data --network host --shm-size=16g \
  -e ALLOW_PARTIAL_BATCHES=true \
  gmicloudai/tinkercloud:dev-local
```

## RLVE Training Command (Production Parameters)

```bash
docker exec tinkercloud-rlve bash -c 'cd /workspace/tinker-cookbook && \
TINKER_API_KEY=slime-dev-key \
WANDB_API_KEY=your_wandb_key \
python -m tinker_cookbook.recipes.rlve.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    base_url=http://localhost:8000 \
    environment_list="Division,EuclidGame,Multiplication,Sorting" \
    groups_per_batch=64 \
    group_size=16 \
    max_tokens=4096 \
    learning_rate=1e-6 \
    loss_fn=ppo \
    advantage_estimator=grpo \
    grpo_reward_center=True \
    grpo_std_normalization=True \
    kl_penalty_coef=0 \
    log_path=/data/logs/rlve-tinker \
    wandb_project=rlve-qwen \
    wandb_name=tinker-7B-4xH200 \
    behavior_if_log_dir_exists=delete \
    eval_every=0 \
    save_every=0 \
    n_batches=10'
```

## Quick Test Command (Smaller Batch)

```bash
docker exec tinkercloud-rlve bash -c 'cd /workspace/tinker-cookbook && \
TINKER_API_KEY=slime-dev-key \
WANDB_API_KEY=your_wandb_key \
python -m tinker_cookbook.recipes.rlve.train \
    model_name=Qwen/Qwen2.5-7B-Instruct \
    base_url=http://localhost:8000 \
    environment_list="Division,Multiplication" \
    groups_per_batch=4 \
    group_size=4 \
    max_tokens=2048 \
    n_batches=5 \
    compute_post_kl=True \
    eval_every=0 \
    save_every=0 \
    log_path=/data/logs/rlve-test \
    behavior_if_log_dir_exists=delete \
    wandb_project=rlve-qwen \
    wandb_name=test-7b'
```

## Key Parameters

| Parameter | Production | Quick Test | Description |
|-----------|------------|------------|-------------|
| `groups_per_batch` | 64 | 4 | Problems per batch |
| `group_size` | 16 | 4 | Samples per problem |
| `max_tokens` | 4096 | 2048 | Max response length |
| `learning_rate` | 1e-6 | 1e-6 | Learning rate |
| `n_batches` | 1000 | 5-10 | Training steps |
| `loss_fn` | ppo | ppo | Loss function |
| `advantage_estimator` | grpo | grpo | Advantage method |

**Total samples per batch** = `groups_per_batch` × `group_size`
- Production: 64 × 16 = 1024 samples (~12 min/batch)
- Quick test: 4 × 4 = 16 samples (~15 sec/batch)

## Available Environments

```
Division, EuclidGame, Multiplication, Sorting,
GCDOne_Counting, HamiltonianPath, LampChanging, LargestConvexPolygon,
PCPPermutation, Path_NoGoingBack_Counting, SAT, ShortestPath,
SpiralMatrix, SubsequenceReversalLNDS, UndamagedSubmatrixCounting, WYRLevelingGround
```

## Available Models

| Model | Path |
|-------|------|
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct` |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

## Monitoring

```bash
# View logs
docker logs -f tinkercloud-rlve

# Check metrics
cat /data/logs/rlve-tinker/metrics.jsonl | jq -r '[.["progress/batch"], .["env/all/correct"], .["env/all/format"]] | @tsv'

# Check difficulty progression
cat /data/logs/rlve-tinker/metrics.jsonl | jq -r '[.["progress/batch"], .["RLVE/Division/difficulty"], .["RLVE/EuclidGame/difficulty"], .["RLVE/Multiplication/difficulty"], .["RLVE/Sorting/difficulty"]] | @tsv'

# Rebuild image after code changes
cd /root/.work/gavin/tinkercloud && ./docker/build_dev.sh gmicloudai/tinkercloud:dev-local
```

## Example Results (10 batches, 7B model)

| Metric | Start | End |
|--------|-------|-----|
| Overall Correct | 82.2% | 85.9% |
| Format | 99.9% | 99.9% |
| Division difficulty | 0 | 1 |
| Sorting difficulty | 1 | 5 |
| EuclidGame accuracy | 32% | 71% |

