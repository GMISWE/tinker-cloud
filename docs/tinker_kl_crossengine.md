# KL Divergence Metrics: Cross-Engine vs Backend

## Background

tinker-cookbook logs two sets of KL-related metrics during RL training. They measure different things and come from different sources. On wandb dashboards, use the `backend/` prefixed metrics and ignore the `optim/kl_sample_train_*` ones.

## Metric Sources

### 1. Cross-Engine KL (client-side, tinker-cookbook)

Source: `compute_kl_sample_train()` in `tinker_cookbook/rl/metrics.py`

| Key | Description |
|-----|-------------|
| `optim/kl_sample_train_v1` | Mean(log p_sampling - log p_training) per action token |
| `optim/kl_sample_train_v2` | 0.5 * Mean((log p_sampling - log p_training)^2) |
| `optim/entropy` | -Mean(log p_sampling) over action tokens |

Compares `datum.loss_fn_inputs["logprobs"]` (from sampling engine, e.g. vLLM/SGLang, on old weights) against `training_logprobs_D` (from training engine, e.g. Megatron, on new weights).

**Problem**: Cross-engine comparison. The sampling engine (vLLM/SGLang) and training engine (Megatron) use different implementations, precision, and kernels. Numerical differences between engines create noise that doesn't reflect real policy drift.

**NeMo RL deferred mode**: In deferred mode, `training_logprobs_D` comes from `optim_result.loss_fn_outputs` which are reference logprobs on pre-step weights (same policy as sampling). So `kl_sample_train_v1` approaches zero and measures only cross-engine numerical noise.

Kept for upstream compatibility. Not used for training decisions.

### 2. Backend KL (server-side, Megatron-vs-Megatron)

Source: `optim_result.metrics` from Miles/NeMo RL backend, routed through `train_step(metrics=metrics)` and prefixed with `backend/`.

| Key | Description |
|-----|-------------|
| `backend/ppo_kl:sum` | KL(pi_old \|\| pi_new) computed entirely within Megatron |
| `backend/pg_clipfrac:mean` | Fraction of tokens where PPO ratio was clipped |
| `backend/pg_loss:sum` | Policy gradient loss |
| `backend/entropy_loss:sum` | Entropy bonus loss |
| `backend/total_loss:sum` | Combined loss |

The training backend computes KL between old and new policy weights using the same engine for both. No cross-engine contamination. This is the reliable metric.

## Which to Trust

| Scenario | Use | Ignore |
|----------|-----|--------|
| Monitoring policy drift | `backend/ppo_kl:sum` | `optim/kl_sample_train_*` |
| Checking clip behavior | `backend/pg_clipfrac:mean` | - |
| Entropy monitoring | `backend/entropy_loss:sum` | `optim/entropy` |

## Code Path

```
train_step()  (tinker_cookbook/rl/train.py)
  -> forward_backward_async() -> fwd_bwd_result (logprobs for cross-engine KL)
  -> optim_step_async() -> optim_result.metrics (backend KL)
  -> metrics.update({f"backend/{k}": v for k, v in optim_result.metrics.items()})

compute_full_batch_metrics_and_get_sampling_client()
  -> compute_kl_sample_train(data_D, training_logprobs_D)  # cross-engine, logged only
  -> metrics dict already contains backend/ prefixed metrics from train_step
```

## Why Not Remove Cross-Engine KL

- Upstream tinker-cookbook uses it; removing breaks compatibility
- It doesn't affect training (logged metric only, never feeds back into loss or advantages)
- GRPO advantage computation and PPO ratio clipping happen independently of this metric
