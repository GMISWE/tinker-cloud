# TinkerCloud Training API

FastAPI server that bridges the Tinker client SDK to pluggable training backends.

```
Tinker Client --> FastAPI --> BackendFactory --> Miles or NeMo RL --> Ray + Megatron GPUs
```

## Quickstart (NeMo RL)

### 1. Deploy

One command via `scripts/deploy_tinkercloud.sh` — clones the four repos
(tinker-cloud, RL, tinker_gmi, tinker-cookbook) at pinned refs, creates the
pod/container, installs the code, starts Ray and the API server, and
health-checks it:

```bash
# Onto a Kubernetes GPU cluster
HF_TOKEN_FILE=~/.hf_token ./scripts/deploy_tinkercloud.sh --source git --target k8s

# Onto a docker GPU box
HF_TOKEN_FILE=~/.hf_token ./scripts/deploy_tinkercloud.sh --source git --target docker

# Redeploy code + restart the server on an existing pod/container
./scripts/deploy_tinkercloud.sh --source git --code-only
```

Both targets run the same in-container setup (`scripts/lib/setup_container.sh`),
so k8s and docker deployments cannot drift. (Developers working in the
tinker-nemorl monorepo can pass `--source dev` to bundle their local working
trees, uncommitted changes included.)

Common knobs (env vars):

| Var | Meaning |
|---|---|
| `HF_TOKEN_FILE` | Local file with the HuggingFace token (required on first deploy) |
| `GPUS` | GPUs to request / declare to Ray (default 4) |
| `IMAGE` | Base image (default `nvcr.io/nvidia/nemo-rl:v0.5.0`) |
| `TINKER_CLOUD_REF`, `RL_REF`, `TINKER_GMI_REF`, `COOKBOOK_REF` | Pinned git refs, default `main` (`*_REPO` to override URLs) |
| `KUBECONFIG`, `NS`, `POD`, `NODE` | k8s target placement: kubeconfig path, namespace, pod name, node name |
| `CONTAINER`, `DATA_DIR`, `DOCKER` | docker target: container name, host data dir mounted at `/data`, docker command (`DOCKER="sg docker -c"` if your shell lacks the docker group) |

k8s note: `GPUS` must fit the node's *unallocated* `nvidia.com/gpu` (idle pods
still hold their allocation), or kubelet rejects the pod.

### 2. Run a recipe

```bash
# docker target (k8s: same command via kubectl -n <namespace> exec <pod> -- bash -c '...')
docker exec -d tinkercloud-nemorl bash -c '
export TINKER_API_KEY=tml-dev-key TINKER_BASE_URL=http://localhost:8000
python -m tinker_cookbook.recipes.math_rl.train \
  model_name=meta-llama/Llama-3.2-1B \
  group_size=16 \
  groups_per_batch=8 \
  learning_rate=1e-4 \
  log_path=/data/trajectories/math_rl
'
```

## Verified tinker-cookbook Recipes (NeMo RL backend)

Validated end-to-end against this server with each recipe's README hyperparameters.

| Recipe | Type | Model | Verdict |
|---|---|---|---|
| math_rl (arithmetic) | RL/GRPO | Llama-3.2-1B | **Pass** — correct 0.74 → 1.00 in 4 batches, stable through 21 |
| rl_basic | RL | Llama-3.2-1B | **Pass** — loss/reward healthy from batch 0 |
| rl_loop | RL (sync) | Qwen2.5-0.5B | **Pass** — 130+ batches, reward 0.0 → 0.5 |
| multiplayer/guess_number | multi-turn RL | Qwen3-4B-Instruct | **Pass** — reward 0.32 → 0.43 over 6 batches |
| preference/shorter | pairwise-preference RL | Qwen3-4B-Instruct | **Pass** — ac_tokens/turn 63.5 → 36.6 by step 22 (target: significant drop by 40) |
| sl_basic | SFT | Llama-3.2-1B | **Pass** — 13+ steps, loss decreasing, grad_norm stable |
| sl_loop | SFT (sync) | Qwen2.5-0.5B | **Pass** — 27+ steps, checkpointing OK |
| chat_sl (no_robots) | SFT (async) | Qwen3-8B-Base | **Pass** — test/nll 1.871 → 1.800 by step 20 (recipe ref: 1.788 @ 140) |
| preference/dpo | DPO | Llama-3.2-1B | **Pass** (mechanics) — 34 steps error-free, gradients flowing; loss ~ln(2) flat in 34 steps at documented lr=1e-5, longer run needed for convergence verdict |

Full status for all 23 recipe entries, bugs, and gaps: `specs/002-nemorl-recipe-testing/compatibility-matrix.md` in the monorepo.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TINKERCLOUD_BACKEND` | `miles` or `nemo_rl` | `miles` |
| `TRAINING_HOST` / `TRAINING_PORT` | Bind address | `0.0.0.0` / `8000` |
| `TINKER_API_KEY` | API auth key | `tml-dev-key` |
| `RAY_ADDRESS` | Ray cluster endpoint | auto (local head) |
| `ALLOW_PARTIAL_BATCHES` | Pad batches smaller than DP size | `false` |
| `HF_HOME` | HuggingFace model cache | `/data/models` |


