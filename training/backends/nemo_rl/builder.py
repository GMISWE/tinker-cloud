"""
NeMo RL argument builder — maps HuggingFace model config + Tinker
create_model args to NeMo RL PolicyConfig dict.

Returns (config_dict, hf_path) similar to MilesArgumentBuilder.
"""
import logging
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder
from .config import NemoRLConfig
from ...utils.model_config import detect_num_gpus
from .loss_config import NO_GRAD_CLIP, TINKER_PG_LOSS_DEFAULTS

logger = logging.getLogger(__name__)

# Default upper bound on the model-derived sequence length. Sizing the policy
# and vLLM engine to a model's *full* native context (often 128K+) would waste
# KV-cache/activation memory, so we cap the auto-derived value here. Operators
# can raise it (Explicit Configuration) for long-context runs.

# HF config attributes that report a model's max context window, in priority order.
_MAX_POSITIONS_ATTRS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "model_max_length",
)


def _read_model_max_positions(cfg: Any) -> Optional[int]:
    """Best-effort read of a model's native context window from its HF config.

    Falls back to a nested text_config (VLMs / composite configs). Returns None
    when no sane positive value is found.
    """
    candidates = [cfg]
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg is not None:
        candidates.append(text_cfg)
    for c in candidates:
        for attr in _MAX_POSITIONS_ATTRS:
            val = getattr(c, attr, None)
            if isinstance(val, int) and 0 < val < 10_000_000:
                return val
    return None


class NemoRLArgumentBuilder(ArgumentBuilder):
    """Builds NeMo RL PolicyConfig + loss config from Tinker API parameters."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None,
                 config: Optional[NemoRLConfig] = None):
        self.overrides = overrides or {}  # raw MasterConfig deep-merge overrides
        self.cfg = config or NemoRLConfig.from_env()

    def build_args(
        self,
        base_model: str,
        num_gpus: int = 0,  # 0 = auto-detect
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Build NeMo RL config from Tinker API parameters.

        Returns:
            Tuple of (config_dict, hf_path) where config_dict contains:
            - "policy": PolicyConfig dict
            - "loss_fn": ClippedPGLossConfig dict
            - "cluster": cluster config (bundle_ct_per_node_list, etc.)
        """
        if num_gpus <= 0:
            num_gpus = detect_num_gpus()
            logger.info("Auto-detected %d GPUs for NeMo RL config", num_gpus)

        debug_train_only = kwargs.get("debug_train_only", False)
        max_batch_size = kwargs.get("max_batch_size", 4096)
        # The client sends max_seq_len=2048 by default (upstream Tinker never has
        # the training script declare a context length). Treat it as a floor and
        # size up to the model's native context below, so long-context recipes
        # (e.g. harbor_rl, 32K trajectories) work with no cookbook/SDK change.
        requested_seq_len = kwargs.get("max_seq_len", 2048)
        seq_len_cap = self.cfg.max_seq_len_cap
        max_seq_len = requested_seq_len
        rlve_config = kwargs.get("rlve_config")
        wandb_config = kwargs.get("wandb_config")

        hf_path = base_model

        # Detect VLM + read the model's native context window (cheap — reads config.json)
        is_vlm = False
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
            is_vlm = (
                hasattr(cfg, "vision_config")
                or "VL" in cfg.__class__.__name__
                or "Vision" in cfg.__class__.__name__
            )
            if is_vlm:
                logger.info("VLM detected: %s — enforcing sequence_packing=False, cp_size=1", cfg.__class__.__name__)

            model_ctx = _read_model_max_positions(cfg)
            if model_ctx:
                derived = min(model_ctx, seq_len_cap)
                max_seq_len = max(requested_seq_len, derived)
                if max_seq_len != requested_seq_len:
                    logger.info(
                        "max_seq_len raised %d -> %d (model context %d, cap %d)",
                        requested_seq_len, max_seq_len, model_ctx, seq_len_cap,
                    )
        except Exception as e:
            logger.debug("model config read skipped (VLM/seq-len detection): %s", e)

        if rlve_config and rlve_config.get("enabled", False):
            miles_only_keys = [
                "custom_prompt_preprocessor", "answer_marker_type",
                "difficulty_sliding_window_size", "min_metric_to_increase_difficulty",
                "min_prompts_before_difficulty_check", "over_sampling_batch_size",
                "use_dynamic_sampling_filter", "partial_rollout", "balance_data",
            ]
            unsupported = [k for k in miles_only_keys if k in rlve_config]
            if unsupported:
                logger.warning(
                    "RLVE server-side args ignored on NeMo RL backend (Miles-only): %s. "
                    "RLVE in Tinker mode is client-driven — these settings have no effect.",
                    unsupported,
                )

        tp_size = 1
        pp_size = 1
        cp_size = 1
        dp_size = num_gpus  # default: all GPUs for data parallelism
        if parallelism:
            # ParallelismConfig (models/requests.py) sends *_size keys; the
            # bare spellings are kept for backward compat. Reading only the
            # bare ones silently dropped every declared TP/PP (specs/009).
            tp_size = parallelism.get(
                "tensor_parallel_size", parallelism.get("tensor_parallel", 1)
            )
            pp_size = parallelism.get(
                "pipeline_parallel_size", parallelism.get("pipeline_parallel", 1)
            )
            cp_size = parallelism.get(
                "context_parallel_size", parallelism.get("context_parallel", 1)
            )
        # Explicit env override, mirroring miles' SLIME_DEFAULT_TP.
        env_tp = self.cfg.default_tp
        if env_tp:
            tp_size = env_tp
            logger.info("Using configured TP override: TP=%d", tp_size)
        if parallelism or env_tp:
            if is_vlm and cp_size > 1:
                logger.warning(
                    "VLM models require cp_size=1 (NeMo RL workers assert empty "
                    "multimodal_kwargs when CP > 1). Overriding cp_size=%d -> 1.",
                    cp_size,
                )
                cp_size = 1
            model_parallel = tp_size * pp_size * cp_size
            dp_size = max(1, num_gpus // model_parallel)

        # GBS = samples per train() call; grad-accum steps = GBS / (MBS * DP)
        # bridge the gap. MBS=1 (the old default) is a measured perf bug at
        # scale (specs/013 A5: 128 sequential single-sequence micro-batches
        # at global-pad width, 5.6x padding inflation, 103 s/step at 8B/TP2).
        # Default is NeMo RL dynamic batching (length-sorted, token-budgeted,
        # per-micro-batch length trim). The budget is a MEMORY constant, not
        # max_seq_len: activations run ~9 MB/token at 8B/TP2 without
        # checkpointing, so a budget that tracks a 32k context packs 32k-token
        # micro-batches and OOMs an H200 (measured, A5 sweep). 8192 tokens
        # (~74 GB at 8B/TP2) is the validated default; the backend raises it
        # per-batch to the longest actual sample before each train/logprob
        # call (backend._ensure_dyn_mb_budget), so a long sample degrades to
        # exactly the old MBS=1 worst case instead of tripping NeMo RL's
        # sample-exceeds-budget assert.
        # NEMORL_TRAIN_MB_TOKENS: override base budget; 0 disables dynamic
        # batching; NEMORL_TRAIN_MBS then sets the static micro-batch size.
        train_global_batch_size = max_batch_size
        train_micro_batch_size = self.cfg.train_mbs
        dyn_env = self.cfg.train_mb_tokens
        # VLMs keep the static path: dynamic batching's slice/truncate is
        # unvalidated against multimodal kwargs (same conservatism as the
        # sequence_packing/cp guards above).
        dyn_default = 0 if is_vlm else min(max_seq_len, 8192)
        dyn_mb_tokens = dyn_env if dyn_env is not None else dyn_default
        dynamic_batching_cfg = (
            {
                "enabled": True,
                "train_mb_tokens": dyn_mb_tokens,
                "logprob_mb_tokens": dyn_mb_tokens,
                "sequence_length_round": 64,
            }
            if dyn_mb_tokens > 0
            else {"enabled": False}
        )

        # Policy config (maps to NeMo RL PolicyConfig TypedDict)
        policy_config = {
            "model_name": base_model,
            "tokenizer": {
                "name": base_model,
            },
            "train_global_batch_size": train_global_batch_size,
            "train_micro_batch_size": train_micro_batch_size,
            "logprob_batch_size": train_micro_batch_size,
            "precision": "bfloat16",
            "max_total_sequence_length": max_seq_len,
            "make_sequence_length_divisible_by": 1,
            # AdamParams.grad_clip_norm defaults to 0 (off); NeMo RL bakes the
            # clip at model creation, so the default must be the API's default.
            # A bound that cannot bind, not None: with None the worker skips the
            # norm entirely and reports grad_norm 0.0, an observation key.
            "max_grad_norm": NO_GRAD_CLIP,
            "offload_optimizer_for_logprob": False,
            # Use DTensor V2 backend (recommended by NeMo RL)
            "dtensor_cfg": {
                "enabled": True,
                "_v2": True,
                "cpu_offload": False,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "tensor_parallel_size": tp_size,
                "context_parallel_size": cp_size,
                "custom_parallel_plan": None,
            },
            "megatron_cfg": {
                "enabled": False,
            },
            "dynamic_batching": dynamic_batching_cfg,
            "sequence_packing": {
                "enabled": False,
            },
            # Generation config (colocated by default)
            "generation": {
                "backend": "vllm",
                "max_new_tokens": max_seq_len,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "stop_token_ids": None,
                "stop_strings": None,
                "vllm_cfg": {
                    # Per-request generation (specs/019): generate_async on
                    # AsyncLLM. Incompatible with vLLM-internal DP, which we
                    # never enable (DP is across worker groups).
                    "async_engine": True,
                    "precision": "bfloat16",
                    "kv_cache_dtype": "auto",
                    "tensor_parallel_size": tp_size,
                    "pipeline_parallel_size": pp_size,
                    "expert_parallel_size": 1,
                    "gpu_memory_utilization": 0.6,
                    "max_model_len": max_seq_len,
                    "enforce_eager": False,
                    "use_deep_gemm": False,
                    "num_last_layers_in_bf16": 0,
                    "num_first_layers_in_bf16": 0,
                    "enable_vllm_metrics_logger": False,
                    "vllm_metrics_logger_interval": 0.5,
                },
                "vllm_kwargs": {},
                "colocated": {
                    "enabled": True,
                },
            },
            # Optimizer. Betas/eps match the Tinker AdamParams contract defaults
            # (beta2=0.95) — the client sends AdamParams per optim_step but the
            # worker only exposes a learning-rate setter, so betas/eps are fixed
            # at creation (P4; mismatches warned in backend.apply_optimizer_step).
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": 5.0e-6,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.95],
                    "eps": 1.0e-8,
                    "foreach": False,
                    "fused": False,
                },
            },
        }

        # The client's declared seed. The DTensor worker builds the model and
        # applies LoRA under torch's process-global RNG and never seeds it, so
        # without this two creations at the same seed start from different
        # weights -- measured: pre-step sum-logprobs -887.7514 / -887.3088 /
        # -887.2706 for one seed (specs/014-gate-suite §THE nemo_rl PEER PROBE).
        # Carried as config rather than an env var so it is per-model and does
        # not race between concurrent creates. The worker must read it; see
        # probes/fix_nemorl_seed.py for the engine-side half.
        if lora_config and lora_config.get("seed") is not None:
            policy_config["seed"] = int(lora_config["seed"])

        if lora_config and lora_config.get("rank", 0) > 0:
            # All-linear coverage (attention + MLP) to match hosted Tinker
            # semantics — per the Tinker LoRA primer, attention-only LoRA
            # under-performs even at matched parameter count (BUG-015 residual
            # gap). Client may override via lora_config["target_modules"].
            target_modules = lora_config.get("target_modules") or [
                "*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj",
                "*.gate_proj", "*.up_proj", "*.down_proj",
            ]
            policy_config["dtensor_cfg"]["lora_cfg"] = {
                "enabled": True,
                "dim": lora_config.get("rank", 8),
                "alpha": lora_config.get("alpha") or lora_config.get("rank", 8),
                "dropout": lora_config.get("dropout", 0.0),
                "dropout_position": "pre",
                "target_modules": target_modules,
                "exclude_modules": [],
                "lora_A_init": "kaiming",
                # Read unconditionally by the TP>1 worker path
                # (automodel/setup.py asserts `not use_triton`). False is
                # PeftConfig.from_dict's default for a missing key, so this
                # pins the kernel choice every banked run already used;
                # triton LoRA is TP-incompatible upstream anyway.
                "use_triton": False,
            }
        else:
            policy_config["dtensor_cfg"]["lora_cfg"] = {"enabled": False}

        # E2 arm (specs/013 round-6): lower the same declaration onto NeMo RL's
        # Megatron path — LoRA attaches to the FUSED linear_qkv/linear_fc1
        # modules, the other side of the lowering partition. Explicit opt-in
        # (NEMORL_MEGATRON=1); default path is unchanged. The Megatron worker
        # has no set_learning_rate, so the client's per-step LR is reproduced
        # server-side by the scheduler (linear lr0*(1-t/T), matching
        # q5_conv_migration.lr_at); verify delivered lr in train metrics.
        if self.cfg.megatron:
            meg_lr = self.cfg.megatron_lr
            meg_iters = self.cfg.megatron_lr_decay_iters
            policy_config["dtensor_cfg"] = {"enabled": False}
            # The bridge scheduler converts lr_decay_iters to SAMPLE units by
            # multiplying with config train_global_batch_size at init, while
            # the worker steps it by the ACTUAL per-call batch. Pin the config
            # gbs to the actual per-step batch so decay is exact (measured:
            # 4096 vs 128 slowed decay 32x — delivered lr 1.99972e-4 vs
            # declared 1.99099e-4 at step 1).
            policy_config["train_global_batch_size"] = self.cfg.megatron_gbs
            if lora_config and lora_config.get("rank", 0) > 0:
                meg_peft = {
                    "enabled": True,
                    # Megatron module names: qkv and gate/up are fused — the
                    # lowering under test. Coverage mirrors all-linear.
                    "target_modules": ["linear_qkv", "linear_proj",
                                       "linear_fc1", "linear_fc2"],
                    "exclude_modules": [],
                    "dim": lora_config.get("rank", 8),
                    "alpha": lora_config.get("alpha") or lora_config.get("rank", 8),
                    "dropout": lora_config.get("dropout", 0.0),
                    "dropout_position": "post",
                    "lora_A_init_method": self.cfg.megatron_a_init,
                    "lora_B_init_method": "zero",
                    "a2a_experimental": False,
                    "lora_dtype": None,
                }
            else:
                meg_peft = {"enabled": False}
            policy_config["megatron_cfg"] = {
                "enabled": True,
                "empty_unused_memory_level": 1,
                "activation_checkpointing": False,
                "converter_type": "Qwen2ForCausalLM",
                "tensor_model_parallel_size": tp_size,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "pipeline_model_parallel_size": pp_size,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "context_parallel_size": cp_size,
                "pipeline_dtype": "bfloat16",
                "sequence_parallel": False,
                "freeze_moe_router": False,  # ValueError with PEFT; model is dense
                "moe_router_dtype": "fp64",
                "moe_router_load_balancing_type": "none",
                "moe_router_bias_update_rate": 0.0,
                "moe_permute_fusion": False,
                "moe_enable_deepep": False,
                "moe_token_dispatcher_type": "allgather",
                "moe_shared_expert_overlap": False,
                "moe_per_layer_logging": False,
                "apply_rope_fusion": True,
                "bias_activation_fusion": True,
                "defer_fp32_logits": False,
                "train_iters": meg_iters,
                "peft": meg_peft,
                "optimizer": {
                    "optimizer": "adam",
                    "lr": meg_lr,
                    "min_lr": 0.0,
                    "weight_decay": 0.0,
                    "bf16": True,
                    "fp16": False,
                    "params_dtype": "float32",
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "adam_eps": 1.0e-8,
                    "sgd_momentum": 0.9,
                    "use_distributed_optimizer": True,
                    "use_precision_aware_optimizer": self.cfg.megatron_precision_aware,
                    # Read unconditionally by the pod's nemo-rl rev
                    # (validate_and_set_config), absent from the older
                    # RL submodule checkout — supply both.
                    "optimizer_cpu_offload": False,
                    "optimizer_offload_fraction": 0.0,
                    "clip_grad": 1.0,
                },
                "scheduler": {
                    "start_weight_decay": 0.0,
                    "end_weight_decay": 0.0,
                    "weight_decay_incr_style": "constant",
                    "lr_decay_style": "linear",
                    "lr_decay_iters": meg_iters,
                    "lr_warmup_iters": 0,
                    "lr_warmup_init": 0.0,
                },
                "distributed_data_parallel_config": {
                    "grad_reduce_in_fp32": False,
                    "overlap_grad_reduce": True,
                    "overlap_param_gather": True,
                    "use_custom_fsdp": False,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                },
            }

        # Loss function config: Tinker's importance_sampling (no clip, no KL);
        # the ppo path swaps in the client's clip bounds per call.
        loss_fn_config = dict(TINKER_PG_LOSS_DEFAULTS)

        if rl_config:
            if "kl_penalty_coef" in rl_config:
                loss_fn_config["reference_policy_kl_penalty"] = rl_config["kl_penalty_coef"]
            if "eps_clip" in rl_config:
                loss_fn_config["ratio_clip_min"] = rl_config["eps_clip"]
                loss_fn_config["ratio_clip_max"] = rl_config["eps_clip"]

        cluster_config = {
            "bundle_ct_per_node_list": [num_gpus],
            "num_gpus_per_node": num_gpus,
            "max_colocated_worker_groups": 2,  # policy + generation
        }

        # Checkpointing config (passed to Policy.save_checkpoint)
        checkpointing_config = {
            "model_save_format": "safetensors",
            "save_consolidated": False,
        }
        if checkpoint_config:
            checkpointing_config.update(checkpoint_config)

        config_dict = {
            "policy": policy_config,
            "loss_fn": loss_fn_config,
            "cluster": cluster_config,
            "checkpointing": checkpointing_config,
            "dp_size": dp_size,
            "debug_train_only": debug_train_only,
            "wandb_config": wandb_config,
            "rlve_config": rlve_config,
        }

        if self.overrides:
            _deep_merge(config_dict, self.overrides)

        logger.info(
            "NeMo RL config built: model=%s, num_gpus=%d, tp=%d, dp=%d, gbs=%d, mbs=%d",
            base_model, num_gpus, tp_size, dp_size,
            train_global_batch_size, train_micro_batch_size,
        )

        return config_dict, hf_path


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
