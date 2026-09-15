"""
Miles argument builder: a Tinker create_model request + MilesConfig + the
model's HF config -> the Megatron/Slime argparse namespace Miles boots from.
"""
import logging
import sys
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

from ..base import ArgumentBuilder
from ...utils.model_config import detect_num_gpus, load_model_config
from .config import MilesConfig
from .model_setup import (
    auto_detect_all_parallelism,
    compute_sglang_mem_fraction,
    detect_torch_dist_path,
)

logger = logging.getLogger(__name__)


class MilesArgumentBuilder(ArgumentBuilder):
    """Builds the Slime/Megatron argument namespace for one Miles model."""

    def __init__(self, default_save_dir: str = "/data/checkpoints/tinker",
                 config: Optional[MilesConfig] = None):
        """
        Args:
            default_save_dir: Default directory for saving checkpoints
            config: Miles knobs (SLIME_* / TINKERCLOUD_MILES_*); read from the
                environment when not given
        """
        self.default_save_dir = default_save_dir
        self.cfg = config or MilesConfig.from_env()

    def build_args(
        self,
        base_model: str,
        lora_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False,
        load_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        parallelism_config: Optional[Dict] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        num_gpus: int = 0,
        parallelism: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ) -> Tuple[Namespace, str]:
        """
        Build Slime training arguments.

        Args:
            base_model: Path to model (can be torch_dist format)
            lora_config: LoRA configuration dict
            debug_train_only: If True, skip update_weights() to avoid SGLang cache flush
            load_dir: Megatron directory to load weights (only) from at create
            save_dir: Megatron --save root for this model (its native area);
                default_save_dir when not given
            parallelism_config: Optional parallelism config (TP, PP, num_gpus)
            max_batch_size: Max batch size for forward_backward (avoids gradient accumulation)
            rlve_config: Optional RLVE configuration (enables server-side RLVE)
            wandb_config: Optional Wandb configuration for logging

        Returns:
            Tuple of (args namespace, hf_model_path)
        """
        lora_config = lora_config or {}
        if parallelism_config is None:
            parallelism_config = parallelism  # ArgumentBuilder ABC spelling
        # Auto-detect torch_dist path
        megatron_checkpoint_path, hf_model_path = detect_torch_dist_path(base_model)

        # Load model config
        model_config = load_model_config(hf_model_path)
        logger.info(f"HF path: {hf_model_path}, Megatron path: {megatron_checkpoint_path}")
        logger.info(f"Loaded model config: {model_config}")

        # Determine parallelism - use new unified auto-detection
        rlve_enabled = bool(rlve_config and rlve_config.get("enabled", False))
        num_gpus = num_gpus or detect_num_gpus()
        if parallelism_config:
            num_gpus = parallelism_config.get("num_gpus", num_gpus)

        # Multi-LoRA pool mode (explicit opt-in, Constitution P3). Requires a
        # LoRA model; boots disaggregated (train/rollout GPUs split — upstream
        # bans colocate+offload with multi-LoRA).
        multi_lora_slots = self.cfg.multilora_slots
        if multi_lora_slots > 0 and not (lora_config and lora_config.get("rank", 0) > 0):
            logger.info("Multi-LoRA pool disabled for this model: no LoRA rank in lora_config")
            multi_lora_slots = 0
        if multi_lora_slots > 0 and rlve_enabled:
            raise ValueError("Multi-LoRA pool mode does not support RLVE")
        rollout_gpus = 0
        if multi_lora_slots > 0:
            train_gpus = self.cfg.train_gpus or max(num_gpus // 2, 1)
            rollout_gpus = num_gpus - train_gpus
            assert rollout_gpus > 0, (
                f"Multi-LoRA needs disaggregated rollout GPUs: total={num_gpus}, train={train_gpus}"
            )
            num_gpus = train_gpus  # parallelism + actor sizing below see train GPUs only

        # Get max sequence length for CP decision
        # Use parameter value (from tinker-cookbook), but allow rlve_config to override for RLVE mode
        rlve_max_response_len = rlve_config.get('rollout_max_response_len') if rlve_config else None
        if rlve_max_response_len:
            max_seq_len = int(rlve_max_response_len)

        # Auto-detect all parallelism dimensions
        parallel = auto_detect_all_parallelism(
            model_config,
            num_gpus,
            max_seq_len=max_seq_len,
            rlve_enabled=rlve_enabled,
            model_name=base_model,
            default_tp=self.cfg.default_tp,
            default_cp=self.cfg.default_cp,
            rlve_tp=self.cfg.rlve_tp,
            rlve_cp=self.cfg.rlve_cp,
        )
        tp_size = parallel['tp']
        pp_size = parallel['pp']
        cp_size = parallel['cp']
        # Attribution-arm override (specs/013 stack-residual): explicit env,
        # default = auto-detect.
        if self.cfg.tp:
            tp_size = self.cfg.tp

        # Build parallel_config dict for compatibility
        parallel_config = {
            'tensor_parallel_size': tp_size,
            'pipeline_parallel_size': pp_size,
            'context_parallel_size': cp_size,
            'num_gpus': num_gpus,
        }

        logger.info(f"Parallelism: TP={tp_size}, PP={pp_size}, CP={cp_size}, GPUs={num_gpus}")

        # Build minimal args for parse_args
        minimal_args = self._build_minimal_args(
            hf_model_path, model_config, tp_size, pp_size, cp_size, megatron_checkpoint_path, max_batch_size,
            num_gpus=num_gpus,
            rlve_config=rlve_config,
            multi_lora_slots=multi_lora_slots,
            lora_config=lora_config,
            save_dir=save_dir,
        )

        # Parse args to get Slime defaults
        args = self._parse_slime_args(minimal_args)

        # Configure model-specific settings
        args = self._configure_model_args(
            args,
            base_model,
            megatron_checkpoint_path,
            lora_config,
            debug_train_only,
            load_dir,
            model_config,
            parallel_config,
            save_dir=save_dir,
            rlve_config=rlve_config,
            wandb_config=wandb_config,
            multi_lora_slots=multi_lora_slots,
            rollout_gpus=rollout_gpus,
            max_seq_len=max_seq_len,
        )

        return args, hf_model_path

    def _build_minimal_args(
        self,
        hf_model_path: str,
        model_config: Dict[str, Any],
        tp_size: int,
        pp_size: int,
        cp_size: int,
        megatron_checkpoint_path: str,
        max_batch_size: int = 4096,
        num_gpus: int = 4,
        rlve_config: Optional[Dict[str, Any]] = None,
        multi_lora_slots: int = 0,
        lora_config: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
    ) -> list:
        """Build minimal CLI arguments for Slime's parse_args."""
        # Check if RLVE mode is enabled
        rlve_enabled = bool(rlve_config and rlve_config.get("enabled", False))

        # Batch size configuration - satisfies Slime's assertion:
        # rollout_batch_size * n_samples_per_prompt % global_batch_size == 0
        if rlve_enabled:
            # RLVE mode: use RLVE-specific rollout settings
            rollout_batch_size = rlve_config.get("rollout_batch_size", 32)
            n_samples_per_prompt = rlve_config.get("n_samples_per_prompt", 8)
            global_batch_size = rollout_batch_size * n_samples_per_prompt
            logger.info(f"RLVE mode: rollout_batch_size={rollout_batch_size}, n_samples_per_prompt={n_samples_per_prompt}")
        else:
            # Standard mode: use max_batch_size as global_batch_size to avoid gradient accumulation
            # (gradient accumulation causes logprobs to be lost for later microbatches)
            rollout_batch_size = max_batch_size
            n_samples_per_prompt = 1
            global_batch_size = max_batch_size
            logger.info(f"Using max_batch_size={max_batch_size} as global_batch_size (avoids gradient accumulation)")

        minimal_args = [
            '--train-backend', 'megatron',
            '--hf-checkpoint', hf_model_path,
            '--rollout-batch-size', str(rollout_batch_size),
            '--n-samples-per-prompt', str(n_samples_per_prompt),
            '--num-rollout', '1',
            # Model parameters from config
            '--num-layers', str(model_config['num_layers']),
            '--hidden-size', str(model_config['hidden_size']),
            '--ffn-hidden-size', str(model_config['ffn_hidden_size']),
            '--num-attention-heads', str(model_config['num_attention_heads']),
            '--num-query-groups', str(model_config['num_query_groups']),
            '--vocab-size', str(model_config['vocab_size']),
            '--norm-epsilon', str(model_config['norm_epsilon']),
            '--rotary-base', str(int(model_config['rotary_base'])),
            '--disable-bias-linear',
            # Training config
            '--seq-length', '512',
            '--micro-batch-size', '1',
            # Pack length must clear the kernel's batch-invariance threshold.
            # get_batch rounds each microbatch's packed stream up to a multiple
            # of tp_size * this value. linear_fc2 (K=4864 -> N=896, the only
            # large-K GEMM) picks a different cuBLAS K-reduction below M=352,
            # so at the stock 128 a client's segmentation selects the kernel:
            # fb(3) packs to 256 (one kernel), fb(5) to 384 (another), and a
            # datum's own returned logprobs move up to 0.38 nats with how the
            # client chunked its round. 512 is the smallest power of two above
            # the threshold and divides the 8192-token dynamic budget exactly,
            # so production-sized packs gain no padding at all. Measured:
            # 5.169e-02 -> 2.882e-07 on SPLIT_INV (specs/014-gate-suite
            # INVESTIGATION-miles-split.md). 352 is a property of this GEMM on
            # this GPU and cuBLAS, not a constant — re-measure with
            # probes/bi_fc2_bisect.py on a new model shape or GPU.
            '--data-pad-size-multiplier',
            str(self.cfg.data_pad_size_multiplier),
            # The client's declared seed, which decides the LoRA initializer.
            # Megatron's own default is 1234, so omitting this was reproducible
            # by accident while silently ignoring whatever the client asked for
            # -- the API accepted the field and nothing read it. On the CLI, not
            # the post-parse namespace: that never reaches the Ray actors.
            '--seed', str((lora_config or {}).get('seed') or 1234),
            '--global-batch-size', str(global_batch_size),
            # RL algorithm
            '--advantage-estimator', self.cfg.advantage_estimator,
            # Note: KL/TIS settings are added conditionally below based on RLVE mode
            # IS ratio clamp, fixed at boot. Default cannot bind (Tinker importance_sampling);
            # a ppo deployment sets SLIME_EPS_CLIP* (see backend._check_miles_clip_config).
            '--eps-clip', str(self.cfg.eps_clip),
            '--eps-clip-high', str(self.cfg.eps_clip_high),
            # Entropy coefficient (0 = no entropy bonus, matches Miles native)
            '--entropy-coef', str(self.cfg.entropy_coef),
        ]
        # Per-request sampling seeds are honoured by SGLang only under
        # deterministic inference (a boot-time engine mode with a throughput cost).
        if self.cfg.sglang_deterministic:
            minimal_args += ['--sglang-enable-deterministic-inference']
        minimal_args += [
            # Weight decay - must be in minimal_args so Megatron's parse_args sees it
            # (Megatron defaults to 0.01; set to 0 for RL where weight decay fights policy updates)
            '--weight-decay', str(self.cfg.weight_decay),
            # Note: --normalize-advantages defaults to False, which is correct
            # (tinker-cookbook already centers advantages within groups)
            # Parallelism — actor-num-gpus-per-node must be set here so parse_args()
            # computes correct world_size and data_parallel_size (Miles defaults to 8)
            '--actor-num-gpus-per-node', str(num_gpus),
            '--tensor-model-parallel-size', str(tp_size),
            '--pipeline-model-parallel-size', str(pp_size),
            '--context-parallel-size', str(cp_size),
            # Checkpoint paths - these are needed during parse_args() so that
            # the fallback logic can set args.load = args.ref_load when no
            # checkpoint resume is specified (see miles/utils/arguments.py:1452-1460)
            '--ref-load', megatron_checkpoint_path,
            '--save', save_dir or self.default_save_dir,
            # Pool mode: save_due_adapter_checkpoints only writes adapters at a
            # save-interval multiple, and the interval lives on the ACTORS' args
            # copy — interval 1 makes the client's save_state the trigger (saves
            # are skipped when the step dir already exists). This is what
            # publishes the cross-backend adapter (specs/007).
            # Dedicated mode: ~100 batches (each batch ~200 microbatches).
            '--save-interval', '1' if multi_lora_slots > 0 else '20000',
        ]

        if multi_lora_slots > 0:
            # Multi-LoRA pool: validate_multi_lora_args runs inside parse_args
            # and needs the LoRA surface at parse time; it also bans
            # colocate/offload (disaggregated boot) and forces the multi-LoRA
            # rollout fn / data source / dynamic GBS.
            rank = lora_config.get("rank", 32) if lora_config else 32
            alpha = (lora_config.get("alpha") or rank) if lora_config else rank
            minimal_args.extend([
                '--multi-lora-n-adapters', str(multi_lora_slots),
                '--lora-rank', str(rank),
                '--lora-alpha', str(alpha),
                '--target-modules', 'all-linear',
                '--qkv-format', 'thd',
            ])
        else:
            # Memory management: colocate SGLang with Megatron, enable offload
            # These MUST be set here because parse_args() sets defaults based on them
            # Attribution-arm override (specs/013 stack-residual): SLIME_NO_OFFLOAD=1
            # drops both flags; default unchanged.
            if not self.cfg.no_offload:
                minimal_args.extend([
                    '--colocate',
                    '--offload',  # Equivalent to --offload-train + --offload-rollout
                ])

        # Add kv-channels if model has explicit head_dim (e.g., Qwen3)
        if model_config.get('kv_channels'):
            minimal_args.extend(['--kv-channels', str(model_config['kv_channels'])])

        # KL loss configuration:
        # - RLVE mode: no KL loss (TIS disabled for testing)
        # - Standard mode: no KL loss by default because with_ref=False means
        #   ref_log_probs come from SGLang sampling, creating a spurious
        #   cross-engine KL gradient (Megatron vs SGLang numerical differences)
        #   that dominates pg_loss ~500:1 and causes training collapse.
        #   Enable via SLIME_USE_KL_LOSS=1 only when a real Megatron reference
        #   model is loaded.
        if self.cfg.use_kl_loss:
            minimal_args.extend([
                '--use-kl-loss',
                '--kl-loss-coef', str(self.cfg.kl_loss_coef),
                '--kl-loss-type', 'low_var_kl',
            ])

        # Add untie-embeddings flag if needed
        if not model_config['tie_word_embeddings']:
            minimal_args.append('--untie-embeddings-and-output-weights')

        # Add RLVE-specific CLI arguments when enabled
        if rlve_enabled:
            environment_list = rlve_config.get("environment_list", [])
            if not environment_list:
                raise ValueError("RLVE enabled but environment_list is empty")

            minimal_args.extend([
                '--rlve',
                '--environment-list', *environment_list,
                '--custom-prompt-preprocessor', rlve_config.get("custom_prompt_preprocessor", "TinyZero"),
                '--answer-marker-type', rlve_config.get("answer_marker_type", "<answer></answer>"),
                '--initial-difficulty', str(rlve_config.get("initial_difficulty", 0)),
                '--difficulty-sliding-window-size', str(rlve_config.get("difficulty_sliding_window_size", 4)),
                '--min-metric-to-increase-difficulty', str(rlve_config.get("min_metric_to_increase_difficulty", 0.9)),
                '--min-prompts-before-difficulty-check', str(rlve_config.get("min_prompts_before_difficulty_check", 8)),
                '--rm-type', 'rlve',  # Required for RLVE reward routing
                '--reward-key', 'reward',
                '--disable-rollout-global-dataset',  # RLVE uses procedural generation, not global dataset
                '--rollout-max-response-len', str(rlve_config.get("rollout_max_response_len", 4096)),
                '--rollout-temperature', str(rlve_config.get("rollout_temperature", 1.0)),
                # GB200-specific args
                '--num-rollout', str(rlve_config.get("num_rollout", 500)),
                '--over-sampling-batch-size', str(rlve_config.get("over_sampling_batch_size", 384)),
            ])

            # Add conditional boolean flags
            if rlve_config.get("balance_data", True):
                minimal_args.append('--balance-data')
            if rlve_config.get("partial_rollout", True):
                minimal_args.append('--partial-rollout')
            if rlve_config.get("use_dynamic_sampling_filter", True):
                minimal_args.extend([
                    '--dynamic-sampling-filter-path',
                    'miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std',
                ])

            logger.info(f"RLVE enabled with {len(environment_list)} environments: {environment_list[:3]}...")

        return minimal_args

    def _parse_slime_args(self, minimal_args: list) -> Namespace:
        """Parse Slime arguments using Slime's parse_args."""
        original_argv = sys.argv
        try:
            sys.argv = ['gmi_wrapper'] + minimal_args
            from miles.utils.arguments import parse_args
            args = parse_args()
            return args
        finally:
            sys.argv = original_argv

    def _configure_model_args(
        self,
        args: Namespace,
        base_model: str,
        megatron_checkpoint_path: str,
        lora_config: Dict,
        debug_train_only: bool,
        load_dir: Optional[str],
        model_config: Dict[str, Any],
        parallel_config: Dict[str, int],
        save_dir: Optional[str] = None,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        multi_lora_slots: int = 0,
        rollout_gpus: int = 0,
        max_seq_len: int = 2048,
    ) -> Namespace:
        """Configure model-specific argument overrides."""
        # Check if RLVE mode is enabled
        rlve_enabled = bool(rlve_config and rlve_config.get("enabled", False))
        # Model architecture flags
        args.swiglu = True
        args.use_rotary_position_embeddings = True
        args.disable_bias_linear = True
        args.add_qkv_bias = True
        args.normalization = "RMSNorm"
        args.group_query_attention = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0

        # Checkpoint paths
        args.pretrained_checkpoint = megatron_checkpoint_path
        args.ref_load = megatron_checkpoint_path
        # every model saves under its own root (the store's native area for
        # it), so counter-numbered iter_* dirs of different models never collide
        args.save = save_dir or self.default_save_dir

        # create_model(resume_from) is a weights-only load by contract: a
        # fresh optimizer, RNG and iteration count. Without these Megatron's
        # --load is a full resume. load_weights(optimizer=true) is the resume path.
        if load_dir:
            args.load = load_dir
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True

        # LoRA configuration. alpha defaults to rank per the API schema
        # (requests.py LoraConfig); alpha=0 zeroes LoRA scaling and gradients
        # entirely (the historical grad_norm=0 constant-loss bug).
        args.lora_rank = lora_config.get("rank", 0) if lora_config else 0
        args.lora_alpha = (lora_config.get("alpha") or args.lora_rank) if lora_config else 0
        args.lora_dropout = lora_config.get("dropout", 0.0) if lora_config else 0.0
        # Megatron module targets from Tinker's train_attn/train_mlp flags
        # (upstream requires target_modules when LoRA is enabled; unembed LoRA
        # is not supported by miles' injector).
        target_modules = []
        if not lora_config or lora_config.get("train_attn", True):
            target_modules += ["linear_qkv", "linear_proj"]
        if not lora_config or lora_config.get("train_mlp", True):
            target_modules += ["linear_fc1", "linear_fc2"]
        args.target_modules = target_modules
        # Attribution-arm override (specs/013 stack-residual): miles' bridge
        # fork defaults lora_A_init_method="xavier" (multi_lora_utils/
        # lora_utils getattr fallback); expose it so the init term measured on
        # the NeMo megatron path (0.076) can be tested on this stack.
        if self.cfg.lora_a_init:
            args.lora_A_init_method = self.cfg.lora_a_init
        # Upstream only injects megatron-side LoRA adapters on the bridge path
        # (model.py: is_lora_enabled and megatron_to_hf_mode == "bridge");
        # without this the model silently builds as full-finetune and LoRA
        # weight sync to SGLang fails (no lora_A/lora_B params).
        if args.lora_rank > 0:
            args.megatron_to_hf_mode = "bridge"

        # Parallelism settings - use values from parallel_config (already auto-detected in build_args)
        tp_size = parallel_config['tensor_parallel_size']
        pp_size = parallel_config['pipeline_parallel_size']
        cp_size = parallel_config['context_parallel_size']
        num_gpus = parallel_config['num_gpus']
        dp_size = num_gpus // (tp_size * pp_size * cp_size)

        args.tensor_model_parallel_size = tp_size
        args.pipeline_model_parallel_size = pp_size
        args.context_parallel_size = cp_size
        args.virtual_pipeline_model_parallel_size = None
        args.sequence_parallel = cp_size > 1  # Enable sequence parallel with CP>1
        # use_distributed_optimizer is deliberately NOT set here. miles derives
        # the DDP config's copy of it independently (bridge_lora_helpers.py:
        # "muon" not in args.optimizer), so pinning it False made the two
        # disagree: DDP reduce-scattered the grad buffer while the optimizer was
        # the non-distributed one, which needs a full all-reduce. Each rank then
        # held the summed gradient only on its own 1/dp shard and its own
        # UNREDUCED gradient everywhere else, and stepped the whole parameter set
        # with it -- so the ranks diverged and the gradient depended on which
        # rank a datum landed on and on the DP width. Letting the engine own the
        # flag keeps its derivations from drifting apart.
        # specs/014-gate-suite/INVESTIGATION-miles-split.md §DEFECT 2.
        args.num_gpus_per_node = num_gpus
        args.actor_num_gpus_per_node = num_gpus
        args.actor_num_nodes = 1

        logger.info(
            f"Parallelism config: TP={tp_size}, PP={pp_size}, "
            f"CP={cp_size}, DP={dp_size} (RLVE={rlve_enabled})"
        )

        # Optimizer settings
        # Note: Initial LR is set here, but can be overridden at each optim_step
        # via adam_params.learning_rate (Tinker API pattern)
        args.optimizer = "adam"
        args.lr = 1e-6
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.98
        args.adam_eps = 1e-8
        wd = self.cfg.weight_decay
        args.weight_decay = wd
        args.start_weight_decay = wd
        args.end_weight_decay = wd

        # LR scheduler
        args.lr_decay_style = "constant"
        args.lr_warmup_iters = 0
        args.lr_decay_iters = 100
        args.min_lr = 1e-6

        # Attention and precision
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.accumulate_allreduce_grads_in_fp32 = True
        args.attention_softmax_in_fp32 = True
        args.attention_backend = "flash"
        args.use_flash_attn = True
        args.use_cpu_initialization = False
        args.bf16 = True
        args.fp16 = False

        # Tokenizer
        args.tokenizer_type = "HuggingFaceTokenizer"
        args.model_name = "qwen2.5"

        # Dynamic batch size: ON by default. miles' regrouping is order-safe
        # (gather_log_data un-permutes by micro_batch_indices; DP merge keys on
        # partition_indices), verified bit-level in specs/013 M-fix: per-token
        # logprobs identical across groupings, 30-step endpoint within 2x the
        # rerun envelope, 27.3 -> 5.3 s/step at 8B/TP2DP2. Budget is the
        # client-declared max_seq_len (floor 8192): packed peak activation ==
        # the old mbs=1 worst case (one full-length sample), so no new OOM
        # surface; not the model's native context (specs/013 A5 lesson).
        args.use_dynamic_batch_size = self.cfg.dyn_batch
        args.max_tokens_per_gpu = self.cfg.max_tokens_per_gpu or max(8192, max_seq_len)

        # Features
        args.colocate = multi_lora_slots == 0
        args.move_rl_fields_to_gpu = True

        # Rollout/SGLang configuration
        args.rollout_num_gpus = rollout_gpus if multi_lora_slots > 0 else 4
        args.rollout_num_gpus_per_engine = 1
        args.sglang_router_ip = None
        args.sglang_router_port = None
        # Must be 1.0: miles' loss hub divides training logits by this arg
        # (logit_processors.get_responses, an RL importance-sampling
        # convention), so any other value silently trains and reports
        # log_softmax(z/T) for every loss fn. Tinker training logprobs are
        # untempered; sampling temperature is per-request, never this arg.
        args.rollout_temperature = 1.0
        args.rollout_top_p = 0.9
        args.rollout_top_k = 50
        args.rollout_max_response_len = 256
        args.rollout_stop = []
        args.rollout_stop_token_ids = None
        args.rollout_skip_special_tokens = True
        args.use_slime_router = False
        args.rollout_external = False
        args.debug_rollout_only = False
        args.debug_train_only = debug_train_only
        args.sglang_mem_fraction_static = compute_sglang_mem_fraction(model_config, base_model)

        # Rollout function paths. Pool mode keeps the multi-LoRA pair that
        # validate_multi_lora_args swapped in during parse (never driven by
        # TinkerCloud, but reset here would break reconcile-time wiring).
        if multi_lora_slots == 0:
            args.rollout_function_path = "miles.rollout.sglang_rollout.generate_rollout"
            args.eval_function_path = "miles.rollout.sglang_rollout.generate_rollout"

        # No server-side prompt dataset: Tinker clients drive all training and
        # sampling data, so RolloutManager must not load one (its data source
        # raises if the file is missing). (Pool mode: validate forces True —
        # the per-adapter data source is controller-global; it loads nothing
        # for Tinker adapters.)
        if multi_lora_slots == 0:
            args.rollout_global_dataset = False
        args.prompt_data = "/data/datasets/gsm8k_rl.jsonl"
        args.rollout_shuffle = False
        args.rollout_max_prompt_len = 2048
        args.input_key = "prompt"
        args.label_key = "response"
        args.metadata_key = "metadata"
        args.tool_key = None
        args.apply_chat_template = False

        # Observability: Wandb logging configuration
        # Priority: wandb_config > SLIME_ENABLE_WANDB env var
        if wandb_config and wandb_config.get("enabled"):
            args.use_wandb = True
            args.wandb_project = wandb_config.get("project", "rlve")
            args.wandb_group = wandb_config.get("group")
            args.wandb_mode = "online"  # Required for real-time logging
            args.wandb_dir = "/data/wandb"
            # Set wandb API key if provided (otherwise uses WANDB_API_KEY env var)
            if wandb_config.get("api_key"):
                args.wandb_key = wandb_config.get("api_key")
            logger.info(f"Wandb logging enabled: project={args.wandb_project}")
        else:
            enable_wandb = self.cfg.enable_wandb
            if enable_wandb:
                logger.warning(
                    "Slime WandB logging is ENABLED (via SLIME_ENABLE_WANDB env var). "
                    "This is NOT recommended for production."
                )
            args.use_wandb = enable_wandb
        args.use_tensorboard = False

        # Environment variables to propagate to Ray workers
        # Ray workers need PYTHONPATH to import megatron.training
        # (megatron-core only installs megatron.core, not megatron.training)
        if rlve_enabled:
            # RLVE mode: include RLVE Gym environments path
            args.train_env_vars = {
                "PYTHONPATH": "/root/Megatron-LM:/root/miles:/root/miles/examples/RLVE",
            }
            # RLVE-specific dataset settings (override defaults)
            args.rollout_global_dataset = False  # Use procedural generation
            args.rlve = True
            args.environment_list = rlve_config.get("environment_list", [])
            args.rm_type = "rlve"
            logger.info(f"RLVE mode configured with {len(args.environment_list)} environments")
        else:
            args.train_env_vars = {
                "PYTHONPATH": "/root/Megatron-LM:/root/miles",
            }

        # Disable offload_train for simpler GPU memory management
        # When offload_train=False, must also set train_memory_margin_bytes=0 to avoid assert.
        # Also reset flags parse_args DERIVED from the minimal-args --offload flag,
        # or actors allocate grad buffers in a torch_memory_saver region without
        # LD_PRELOAD (only set for offload actors) and crash at init.
        args.offload_train = False
        args.offload_rollout = False
        args.train_memory_margin_bytes = 0
        args.disable_grad_buffers_cpu_backup = False
        args.disable_param_buffers_cpu_backup = False

        # Tinker seam (miles tinker-seam branch):
        # per-request batch sizes ride the dynamic_global_batch_size rollout key
        # (its assert couples this arg to the key's presence), and each actor
        # splits the fanned-out batch by DP rank locally.
        args.use_dynamic_global_batch_size = True
        args.delay_split_train_data_by_dp = True

        return args
