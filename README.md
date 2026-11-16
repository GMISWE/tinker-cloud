<p align="center"><img src="docs/tinkercloud-mark.svg" alt="tinkercloud" width="640"></p>

An open training server for the [Tinker](https://thinkingmachines.ai/tinker/) API. Training loops
written against the Tinker SDK (RL, SFT, DPO, distillation) run unchanged on your own GPUs, on the
training engine you choose.

```
tinker SDK / tinker-cookbook ──HTTP──▶ TinkerCloud ──▶ one backend ──▶ Ray actors on your GPUs
                                                        Miles · NeMo RL · Automodel · verl · Megatron-Bridge
```

## Why TinkerCloud

- **One API, any engine.** The backend is chosen when the server starts (`--backend <name>`). Client
  code never changes, whether the engine is Megatron or FSDP and whether it trains adapters or full
  models.
- **The Tinker primitives, kept exact.** `forward_backward`, `optim_step`, `sample` and the
  checkpoint calls keep their documented meaning on every backend. A model's requests run in
  program order, a retried request is idempotent, and every result is validated against its
  response model before a client can read it.
- **Shared GPUs.** Many LoRA tenants train on one model replica set and sample from one vLLM
  (Automodel). Several full models, a student and its teachers, share the same GPUs (verl).
- **Checked on real recipes.** The tinker-cookbook recipes below run end to end with their published
  hyperparameters. The unit suites run on every pull request, and a conformance suite checks the
  protocol against a CPU backend.
- **Deployable as it stands.** One script deploys to Kubernetes or docker. Ray runs with token
  authentication, the pod ships with a NetworkPolicy that admits only the API port, model
  repository code runs only for models the operator lists, and `ENV=production` refuses the public
  dev key.
- **Portable checkpoints.** Adapters are saved in PEFT layout and can be exported through signed
  download URLs.

## Backends

One server process serves one backend. Each implements the same `TrainingBackend` interface
(`training/backends/`).

| Backend | Engine | Serves | Notes |
|---|---|---|---|
| `nemo_rl` | NeMo RL on Megatron-Bridge | RL, SFT and DPO recipes | The verified recipes below |
| `automodel` | NeMo Automodel, FSDP2 | LoRA: many tenants per training group | AdamW and a Muon-family optimizer; one vLLM per base model |
| `verl` | verl, FSDP | Full fine-tuning; per-candidate `cross_entropy` | Multi-teacher distillation: see [verl backend](#verl-backend) |
| `miles` | Miles on Megatron | `cross_entropy`, `importance_sampling`, `ppo` | A pool of LoRA adapters per base model |
| `megatron_bridge` | Megatron-Bridge | `cross_entropy`, `classification_ce` (sequence classification) | Experimental |
| `fake` | none (CPU) | Deterministic stand-in | Runs the protocol suite without a GPU |

## Quickstart (NeMo RL)

### 1. Deploy


```bash
# Onto a Kubernetes GPU cluster
HF_TOKEN_FILE=~/.hf_token ./scripts/deploy_tinkercloud.sh --source git --target k8s

# Onto a docker GPU box
HF_TOKEN_FILE=~/.hf_token ./scripts/deploy_tinkercloud.sh --source git --target docker

# Redeploy code + restart the server on an existing pod/container
./scripts/deploy_tinkercloud.sh --source git --code-only
```

The server itself is `python -m training [--backend miles|nemo_rl|...]` (or
`uvicorn training.api:app`); the deploy script runs it behind a Ray head.

Common knobs:

| Var | Meaning |
|---|---|
| `HF_TOKEN_FILE` | Local file with the HuggingFace token (required on first deploy) |
| `API_KEY_FILE` | Local file with the server's API key: one line without whitespace (a comma list gives several keys; a client sends one of them). Unset, the target keeps the key it has; a target with none runs on the public `tml-dev-key`. The key lives under the target's `/data`: a target recreated on an empty `/data` needs `API_KEY_FILE` again. Also read by `scripts/pod_rotate_server.sh` |
| `GPUS` | GPUs to request / declare to Ray (default 4) |
| `IMAGE` | Base image (default `nvcr.io/nvidia/nemo-rl:v0.5.0`) |
| `TINKER_CLOUD_REF`, `RL_REF`, `TINKER_GMI_REF`, `COOKBOOK_REF` | Pinned git refs, default `main` (`*_REPO` to override URLs) |
| `KUBECONFIG`, `NS`, `POD`, `NODE` | k8s target placement: kubeconfig path, namespace, pod name, node name |
| `CONTAINER`, `DATA_DIR`, `DOCKER` | docker target: container name, host data dir mounted at `/data`, docker command (`DOCKER="sg docker -c"` if your shell lacks the docker group) |

k8s note: `GPUS` must fit the node's *unallocated* `nvidia.com/gpu` (idle pods
still hold their allocation), or kubelet rejects the pod.

### 2. Run a recipe

```bash
# a target deployed with API_KEY_FILE: TINKER_API_KEY=$(cut -d, -f1 /data/tinkercloud_api_key) instead
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
| preference/shorter | pairwise-preference RL | Qwen3-4B-Instruct | **Pass** — ac_tokens/turn 63.5 → 24.7 by step 31 (target: significant drop by 40) |
| sl_basic | SFT | Llama-3.2-1B | **Pass** — 13+ steps, loss decreasing, grad_norm stable |
| sl_loop | SFT (sync) | Qwen2.5-0.5B | **Pass** — 27+ steps, checkpointing OK |
| chat_sl (no_robots) | SFT (async) | Qwen3-8B-Base | **Pass** — test/nll 1.871 → 1.800 by step 20 (recipe ref: 1.788 @ 140) |
| preference/dpo | DPO | Llama-3.2-1B | **Pass** (mechanics) — 34 steps error-free, gradients flowing; loss ~ln(2) flat in 34 steps at documented lr=1e-5, longer run needed for convergence verdict |


## verl backend

The `verl` backend serves full fine-tuning on [verl](https://github.com/verl-project/verl)'s FSDP
engine, and the primitives that distillation from several teachers needs:

- **Full models, not only adapters.** Each model is its own worker group; `optim_step` applies
  the step's AdamW parameters and its `grad_clip_norm`.
- **Per-candidate cross-entropy.** `cross_entropy` takes `(N, K)` targets and signed weights and
  returns `(N, K)` logprobs, so a loss over the K best tokens of each position is one call.
- **Teachers beside the student.** A teacher is a forward-only model on the student's GPUs
  (`TINKERCLOUD_VERL_MODELS_PER_GPU`). One pass runs at a time, so models never compete for memory.
- **Top-K sampling.** `sample` returns the K best logprobs at each position. The vLLM sampler
  sleeps while any model trains and wakes for the next sample.

The tinker-cookbook recipe `recipes/distillation/mopd` uses all four: a 3B student, three teachers
and one sampler on four GPUs. Start the server with `python -m training --backend verl` inside a
verl environment; the deploy script has no verl profile yet.

## Configuration

Server-wide settings (`training/config.py`):

| Variable | Purpose | Default |
|---|---|---|
| `TINKERCLOUD_BACKEND` | `miles`, `nemo_rl`, `automodel`, `verl`, `megatron_bridge`, `fake` (also `python -m training --backend B`) | `miles` |
| `TINKERCLOUD_BACKEND_OVERRIDES` | JSON object of backend config fields, applied over the environment | unset |
| `TRAINING_HOST` / `TRAINING_PORT` | Bind address | `0.0.0.0` / `8000` |
| `TINKER_API_KEY` | API auth key (comma-separated for several); `ENV=production` refuses to start when the list is empty or holds `tml-dev-key`. The deploy and rotate scripts set it from the key file on the target (given once as `API_KEY_FILE`), else to `tml-dev-key`; never from a KEY=VAL | `tml-dev-key` |
| `RAY_ADDRESS` / `RAY_NAMESPACE` | Ray cluster endpoint / namespace | auto (local head) / `default` |
| `RAY_AUTH_MODE` | `token`: every Ray process must hold the deployment's token; `ENV=production` refuses to start with anything else. The deploy and rotate scripts set it | unset |
| `RAY_AUTH_TOKEN_PATH` | The token file Ray reads in token mode (read by Ray and the deploy scripts, not by `config.py`); the deploy writes one per pod at `/data/ray_auth_token` (mode 600) | unset |
| `METADATA_DIR` | Futures DB, sessions DB, checkpoint metadata | `/data/metadata` |
| `TINKERCLOUD_CHECKPOINT_BASE` | Root of every model's checkpoints (`<model>/{weights,sampler_weights,external_weights}/<name>/`) and native area (`<model>/native/`) | `/data/checkpoints` |
| `TINKERCLOUD_URL_SIGNING_KEY` | HMAC key for the signed per-file download URLs of `save_weights_external` exports; unset = random per process (URLs die with it); required when `ENV=production` | unset |
| `TINKERCLOUD_EXTERNAL_URL_TTL_S` | Lifetime of a signed download URL, seconds | `3600` |
| `TINKERCLOUD_PUBLIC_BASE_URL` | Origin written into signed URLs; unset = the request's own base URL | unset |
| `SESSION_TIMEOUT_S` / `SESSION_REAP_INTERVAL_S` | Expire silent sessions and free their models (`-1` disables) | `600` / `60` |
| `ALLOW_PARTIAL_BATCHES` | Pad batches smaller than the DP size | `false` |
| `KGATEWAY_LOG_LEVEL` / `KGATEWAY_ACCESS_LOG` | Log level / HTTP access log | `INFO` / `false` |
| `HF_HOME` | HuggingFace model cache | `/data/models` |
| `TINKERCLOUD_TRUST_REMOTE_CODE_MODELS` | Comma-separated model ids / paths whose repository code transformers may import (`trust_remote_code`), each as the backend loads it (miles and verl may resolve an id to a local directory); a model whose config, tokenizer or processor config declares custom code (an `auto_map` at any depth) is refused unless listed, as is one the hub cannot answer for or that names no repository the server can read; with the hub library in offline mode (`HF_HUB_OFFLINE`) the cached snapshot is what is checked. The automodel backend applies the list to its tokenizer loads only: its model load and its vLLM server pass no `trust_remote_code` | unset (none) |

Per-backend knobs are declared fields with typed defaults in
`training/backends/miles/config.py` (`SLIME_*`, `TINKERCLOUD_MILES_*`),
`training/backends/nemo_rl/config.py` (`NEMORL_*`, `NRL_*`),
`training/backends/verl/config.py` (`TINKERCLOUD_VERL_*`) and
`training/backends/megatron_bridge/config.py` (`MEGATRON_BRIDGE_*`); the server
logs the effective values and their sources at startup. What the SDK used to
declare per model now lives here: debug mode (`NEMORL_DEBUG_TRAIN_ONLY`),
the refit-elision bound (`NEMORL_STALENESS_K`), the sequence length
(`NEMORL_MAX_SEQ_LEN`, default = the model's context capped by
`TINKERCLOUD_MAX_SEQ_LEN_CAP`) and the Evo2 recipe paths. The classification
head comes from the model's own `config.json` (`architectures` plus an explicit
`num_labels` or `id2label`).

## Tests

CPU-only suites run against the `fake` backend; see `tests/README.md`
(`scripts/pod_pytest.sh` runs them on a cluster pod against your working tree).
