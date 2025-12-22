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
    detect_torch_dist_path,
    parse_checkpoint_uri,
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
        parallelism_config: Optional[Dict] = None
    ) -> Tuple[Namespace, str]:
        """
        Build Slime training arguments.

        Args:
            base_model: Path to model (can be torch_dist format)
            lora_config: LoRA configuration dict
            debug_train_only: If True, skip update_weights() to avoid SGLang cache flush
            checkpoint_path: If provided, load from this checkpoint
            parallelism_config: Optional parallelism config (TP, PP, num_gpus)

        Returns:
            Tuple of (args namespace, hf_model_path)
        """
        # Auto-detect torch_dist path
        megatron_checkpoint_path, hf_model_path = detect_torch_dist_path(base_model)

        # Load model config
        model_config = load_model_config(hf_model_path)
        logger.info(f"HF path: {hf_model_path}, Megatron path: {megatron_checkpoint_path}")
        logger.info(f"Loaded model config: {model_config}")

        # Determine parallelism
        parallel_config = get_parallelism_config(
            model_config, parallelism_config, base_model
        )
        tp_size = parallel_config['tensor_parallel_size']
        pp_size = parallel_config['pipeline_parallel_size']
        num_gpus = parallel_config['num_gpus']

        logger.info(f"Parallelism: TP={tp_size}, PP={pp_size}, GPUs={num_gpus}")

        # Build minimal args for parse_args
        minimal_args = self._build_minimal_args(
            hf_model_path, model_config, tp_size, pp_size, megatron_checkpoint_path
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
            parallel_config
        )

        return args, hf_model_path

    def _build_minimal_args(
        self,
        hf_model_path: str,
        model_config: Dict[str, Any],
        tp_size: int,
        pp_size: int,
        megatron_checkpoint_path: str
    ) -> list:
        """Build minimal CLI arguments for Slime's parse_args."""
        # Batch size configuration - satisfies Slime's assertion:
        # rollout_batch_size * n_samples_per_prompt % global_batch_size == 0
        rollout_batch_size = 8
        n_samples_per_prompt = 1
        global_batch_size = rollout_batch_size * n_samples_per_prompt

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
            # Parallelism
            '--tensor-model-parallel-size', str(tp_size),
            '--pipeline-model-parallel-size', str(pp_size),
            # Checkpoint paths - these are needed during parse_args() so that
            # the fallback logic can set args.load = args.ref_load when no
            # checkpoint resume is specified (see miles/utils/arguments.py:1452-1460)
            '--ref-load', megatron_checkpoint_path,
            '--save', self.default_save_dir,
            '--save-interval', '100',
        ]

        # Add untie-embeddings flag if needed
        if not model_config['tie_word_embeddings']:
            minimal_args.append('--untie-embeddings-and-output-weights')

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
        parallel_config: Dict[str, int]
    ) -> Namespace:
        """Configure model-specific argument overrides."""
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
        args.lora_rank = lora_config.get("rank", 0)
        args.lora_alpha = lora_config.get("alpha", 0)
        args.lora_dropout = lora_config.get("dropout", 0.0)

        # Parallelism settings
        num_gpus = parallel_config.get('num_gpus', 4)
        args.virtual_pipeline_model_parallel_size = None
        args.context_parallel_size = 1
        args.sequence_parallel = False
        args.use_distributed_optimizer = False
        args.num_gpus_per_node = num_gpus
        args.actor_num_gpus_per_node = num_gpus
        args.actor_num_nodes = 1

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

        # Dynamic batch size
        args.use_dynamic_batch_size = True
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
        args.sglang_mem_fraction_static = 0.8

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

        # Observability: Disable Slime's built-in logging backends
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
        args.train_env_vars = {
            "PYTHONPATH": "/root/Megatron-LM:/root/miles",
        }

        # Enable offload_train and offload_rollout for memory management
        # offload_train: Required when train_memory_margin_bytes > 0 (default is 1GB)
        args.offload_train = True
        args.offload_rollout = True

        return args
