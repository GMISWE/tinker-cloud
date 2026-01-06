"""
Slime argument builder for training configuration.

This module handles the construction of Slime training arguments,
integrating model configuration, parallelism settings, and LoRA configuration.
"""
import logging
import os
import sys
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

from ..utils.model_config import (
    load_model_config,
    get_parallelism_config,
    auto_detect_all_parallelism,
    detect_torch_dist_path,
    parse_checkpoint_uri,
    compute_sglang_mem_fraction,
)

logger = logging.getLogger(__name__)


class SlimeArgumentBuilder:
    """
    Builds Slime training arguments from configuration.

    This class encapsulates the logic for constructing Slime's argument
    namespace, handling model discovery, parallelism configuration, and
    all the various training settings required by Slime.
    """

    def __init__(self, default_save_dir: str = "/data/checkpoints/tinker"):
        """
        Initialize the argument builder.

        Args:
            default_save_dir: Default directory for saving checkpoints
        """
        self.default_save_dir = default_save_dir

    def build_args(
        self,
        base_model: str,
        lora_config: Dict[str, Any],
        debug_train_only: bool = False,
        checkpoint_path: Optional[str] = None,
        parallelism_config: Optional[Dict] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Namespace, str]:
        """
        Build Slime training arguments.

        Args:
            base_model: Path to model (can be torch_dist format)
            lora_config: LoRA configuration dict
            debug_train_only: If True, skip update_weights() to avoid SGLang cache flush
            checkpoint_path: If provided, load from this checkpoint
            parallelism_config: Optional parallelism config (TP, PP, num_gpus)
            max_batch_size: Max batch size for forward_backward (avoids gradient accumulation)
            rlve_config: Optional RLVE configuration (enables server-side RLVE)
            wandb_config: Optional Wandb configuration for logging

        Returns:
            Tuple of (args namespace, hf_model_path)
        """
        # Auto-detect torch_dist path
        megatron_checkpoint_path, hf_model_path = detect_torch_dist_path(base_model)

        # Load model config
        model_config = load_model_config(hf_model_path)
        logger.info(f"HF path: {hf_model_path}, Megatron path: {megatron_checkpoint_path}")
        logger.info(f"Loaded model config: {model_config}")

        # Determine parallelism - use new unified auto-detection
        rlve_enabled = rlve_config and rlve_config.get("enabled", False)
        num_gpus = int(os.environ.get("SLIME_NUM_GPUS", "4"))
        if parallelism_config:
            num_gpus = parallelism_config.get("num_gpus", num_gpus)

        # Get max sequence length for CP decision
        # Use parameter value (from tinker-cookbook), but allow rlve_config to override for RLVE mode
        if rlve_config and rlve_config.get('rollout_max_response_len'):
            max_seq_len = rlve_config.get('rollout_max_response_len')

        # Auto-detect all parallelism dimensions
        parallel = auto_detect_all_parallelism(
            model_config,
            num_gpus,
            max_seq_len=max_seq_len,
            rlve_enabled=rlve_enabled,
            model_name=base_model
        )
        tp_size = parallel['tp']
        pp_size = parallel['pp']
        cp_size = parallel['cp']

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
            rlve_config=rlve_config
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
            checkpoint_path,
            model_config,
            parallel_config,
            rlve_config=rlve_config,
            wandb_config=wandb_config
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
        rlve_config: Optional[Dict[str, Any]] = None
    ) -> list:
        """Build minimal CLI arguments for Slime's parse_args."""
        # Check if RLVE mode is enabled
        rlve_enabled = rlve_config and rlve_config.get("enabled", False)

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
            '--global-batch-size', str(global_batch_size),
            # RL algorithm
            '--advantage-estimator', os.environ.get('SLIME_ADVANTAGE_ESTIMATOR', 'grpo'),
            # Note: KL/TIS settings are added conditionally below based on RLVE mode
            # PPO clipping - asymmetric clip for importance ratios (matches Miles native)
            '--eps-clip', os.environ.get('SLIME_EPS_CLIP', '0.2'),
            '--eps-clip-high', os.environ.get('SLIME_EPS_CLIP_HIGH', '0.28'),
            # Entropy coefficient (0 = no entropy bonus, matches Miles native)
            '--entropy-coef', os.environ.get('SLIME_ENTROPY_COEF', '0.00'),
            # Note: --normalize-advantages defaults to False, which is correct
            # (tinker-cookbook already centers advantages within groups)
            # Parallelism
            '--tensor-model-parallel-size', str(tp_size),
            '--pipeline-model-parallel-size', str(pp_size),
            '--context-parallel-size', str(cp_size),
            # Checkpoint paths - these are needed during parse_args() so that
            # the fallback logic can set args.load = args.ref_load when no
            # checkpoint resume is specified (see miles/utils/arguments.py:1452-1460)
            '--ref-load', megatron_checkpoint_path,
            '--save', self.default_save_dir,
            '--save-interval', '20000',  # ~100 batches (each batch ~200 microbatches)
            # Memory management: colocate SGLang with Megatron, enable offload
            # These MUST be set here because parse_args() sets defaults based on them
            '--colocate',
            '--offload',  # Equivalent to --offload-train + --offload-rollout
        ]

        # Add kv-channels if model has explicit head_dim (e.g., Qwen3)
        if model_config.get('kv_channels'):
            minimal_args.extend(['--kv-channels', str(model_config['kv_channels'])])

        # Add TIS/KL settings conditionally based on RLVE mode
        if rlve_enabled:
            # RLVE mode: TIS disabled for testing (was: minimal_args.append('--use-tis'))
            pass
        else:
            # Standard mode: Use KL loss for stability
            minimal_args.extend([
                '--use-kl-loss',
                '--kl-loss-coef', os.environ.get('SLIME_KL_LOSS_COEF', '0.1'),
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
        checkpoint_path: Optional[str],
        model_config: Dict[str, Any],
        parallel_config: Dict[str, int],
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """Configure model-specific argument overrides."""
        # Check if RLVE mode is enabled
        rlve_enabled = rlve_config and rlve_config.get("enabled", False)
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
        args.save = self.default_save_dir

        # Handle checkpoint resume
        if checkpoint_path:
            args.load = parse_checkpoint_uri(checkpoint_path, args.save)

        # LoRA configuration
        args.lora_rank = lora_config.get("rank", 0) if lora_config else 0
        args.lora_alpha = lora_config.get("alpha", 0) if lora_config else 0
        args.lora_dropout = lora_config.get("dropout", 0.0) if lora_config else 0.0

        # Parallelism settings - use values from parallel_config (already auto-detected in build_args)
        tp_size = parallel_config.get('tensor_parallel_size', 2)
        pp_size = parallel_config.get('pipeline_parallel_size', 1)
        cp_size = parallel_config.get('context_parallel_size', 1)
        num_gpus = parallel_config.get('num_gpus', 4)
        dp_size = num_gpus // (tp_size * pp_size * cp_size)

        args.tensor_model_parallel_size = tp_size
        args.pipeline_model_parallel_size = pp_size
        args.context_parallel_size = cp_size
        args.virtual_pipeline_model_parallel_size = None
        args.sequence_parallel = cp_size > 1  # Enable sequence parallel with CP>1
        args.use_distributed_optimizer = False
        args.num_gpus_per_node = num_gpus
        args.actor_num_gpus_per_node = num_gpus
        args.actor_num_nodes = 1

        logger.info(
            f"Parallelism config: TP={tp_size}, PP={pp_size}, "
            f"CP={cp_size}, DP={dp_size} (RLVE={rlve_enabled})"
        )

        # Optimizer settings
        args.optimizer = "adam"
        args.lr = 1e-6
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.98
        args.adam_eps = 1e-8
        args.weight_decay = 0.1

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

        # Dynamic batch size - disabled for simplicity to avoid reordering complexity
        args.use_dynamic_batch_size = False
        args.max_tokens_per_gpu = 4096

        # Features
        args.colocate = True
        args.move_rl_fields_to_gpu = True

        # Rollout/SGLang configuration
        args.rollout_num_gpus = 4
        args.rollout_num_gpus_per_engine = 1
        args.sglang_router_ip = None
        args.sglang_router_port = None
        args.rollout_temperature = 0.7
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

        # Rollout function paths
        args.rollout_function_path = "miles.rollout.sglang_rollout.generate_rollout"
        args.eval_function_path = "miles.rollout.sglang_rollout.generate_rollout"

        # Dataset configuration (fallback for testing)
        args.rollout_global_dataset = True
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
            enable_wandb = os.getenv("SLIME_ENABLE_WANDB", "0") == "1"
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
        # When offload_train=False, must also set train_memory_margin_bytes=0 to avoid assert
        args.offload_train = False
        args.offload_rollout = False
        args.train_memory_margin_bytes = 0

        return args
