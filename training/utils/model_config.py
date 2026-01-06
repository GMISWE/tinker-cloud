"""
Model configuration utilities for HuggingFace models.

This module provides utilities for loading and analyzing model configurations,
including parameter estimation and parallelism recommendations.
"""
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_model_config(base_model: str) -> Dict[str, Any]:
    """
    Load HuggingFace model config from base_model path.

    Args:
        base_model: Path to model (can be torch_dist format)

    Returns:
        Dict with model config parameters

    Raises:
        Exception: If config cannot be loaded
    """
    from transformers import AutoConfig

    # Derive HF path (remove _torch_dist suffix if present)
    hf_model_path = base_model.replace('_torch_dist', '')

    try:
        config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

        return {
            'num_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'ffn_hidden_size': config.intermediate_size,
            'num_attention_heads': config.num_attention_heads,
            'num_query_groups': getattr(
                config, 'num_key_value_heads', config.num_attention_heads
            ),
            'kv_channels': getattr(config, 'head_dim', None),  # Qwen3 explicit head_dim
            'vocab_size': config.vocab_size,
            'norm_epsilon': getattr(
                config, 'rms_norm_eps',
                getattr(config, 'layer_norm_eps', 1e-6)
            ),
            'rotary_base': getattr(config, 'rope_theta', 10000),
            'tie_word_embeddings': getattr(config, 'tie_word_embeddings', False),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 2048),
        }
    except Exception as e:
        logger.error(f"Failed to load model config from {hf_model_path}: {e}")
        raise


def estimate_model_params(
    model_config: Dict[str, Any],
    model_name: str = ""
) -> float:
    """
    Estimate model parameters in billions.

    Uses two strategies:
    1. Extract from model name (e.g., "Llama-3.1-8B" → 8.0B)
    2. Estimate from hidden_size as proxy

    Args:
        model_config: Model configuration dict from load_model_config()
        model_name: Optional model name/path

    Returns:
        Estimated parameters in billions
    """
    # Strategy 1: Extract from model name (most reliable)
    # Examples: "Llama-3.1-8B", "Qwen2.5-0.5B", "Llama-2-70B"
    if model_name:
        match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
        if match:
            size_b = float(match.group(1))
            logger.info(f"Extracted {size_b}B from model name: {model_name}")
            return size_b

    # Strategy 2: Use hidden_size as proxy (fallback)
    # Correlation across architectures:
    # - hidden_size ~= 896: 0.5B (Qwen2.5-0.5B)
    # - hidden_size ~= 2048: 1-2B
    # - hidden_size ~= 4096: 7-8B (Llama-3.1-8B, Llama-2-7B)
    # - hidden_size ~= 5120: 13B (Llama-2-13B)
    # - hidden_size ~= 8192: 70B (Llama-2-70B)
    hidden_size = model_config['hidden_size']

    if hidden_size < 1536:  # < 1.5K → 0.5B range
        return 0.5
    elif hidden_size < 3072:  # 1.5K-3K → 1-2B range
        return 1.5
    elif hidden_size < 4608:  # 3K-4.6K → 7-8B range
        return 8.0
    elif hidden_size < 6144:  # 4.6K-6K → 13B range
        return 13.0
    elif hidden_size < 10240:  # 6K-10K → 30-70B range
        return 30.0
    else:  # > 10K
        return 70.0


def get_parallelism_config(
    model_config: Dict[str, Any],
    user_config: Optional[Dict] = None,
    model_name: str = ""
) -> Dict[str, int]:
    """
    Determine parallelism configuration using hybrid approach:
    1. Environment defaults (deployment-level config)
    2. Auto-detection based on model size (heuristic)
    3. User override (request-level config)

    Args:
        model_config: Model configuration dict
        user_config: Optional user-provided parallelism config
        model_name: Optional model name/path for extracting size

    Returns:
        Dict with tensor_parallel_size, pipeline_parallel_size, num_gpus
    """
    # 1. Environment defaults
    default_tp = int(os.getenv("SLIME_DEFAULT_TP", "1"))
    default_pp = int(os.getenv("SLIME_DEFAULT_PP", "1"))
    default_num_gpus = int(os.getenv("SLIME_NUM_GPUS", "4"))

    # 2. Auto-detect based on model size (if env not explicitly set)
    if default_tp == 1:
        total_params = estimate_model_params(model_config, model_name)
        logger.info(f"Estimated model params: {total_params:.2f}B")

        # For 4 GPUs available: use TP*PP=4
        # Llama-3.1-8B converted with TP=2, PP=2
        if total_params >= 30:    # >= 30B params: TP=8, PP=1 (requires 8 GPUs)
            default_tp = 8
            default_pp = 1
        elif total_params >= 10:  # 10B-30B params: TP=4, PP=1 (requires 4 GPUs)
            default_tp = 4
            default_pp = 1
        elif total_params >= 2:   # 2B-10B params: TP=2, PP=1, CP=2 (requires 4 GPUs)
            default_tp = 2
            default_pp = 1  # Use CP=2 instead of PP=2 for better RLVE alignment
        # else: < 2B params: TP=1, PP=1 (requires 1 GPU)

        logger.info(
            f"Auto-detected TP={default_tp}, PP={default_pp} "
            f"for {total_params:.2f}B params"
        )

    # 3. User override (optional)
    if user_config:
        tp = user_config.get("tensor_parallel_size", default_tp)
        pp = user_config.get("pipeline_parallel_size", default_pp)
        num_gpus = user_config.get("num_gpus", default_num_gpus)
        logger.info(f"User override: TP={tp}, PP={pp}, GPUs={num_gpus}")
    else:
        tp, pp, num_gpus = default_tp, default_pp, default_num_gpus

    return {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "num_gpus": num_gpus
    }


def auto_detect_all_parallelism(
    model_config: Dict[str, Any],
    num_gpus: int,
    max_seq_len: int = 2048,
    rlve_enabled: bool = False,
    model_name: str = ""
) -> Dict[str, int]:
    """
    Auto-detect all parallelism dimensions (TP, PP, CP, DP).

    Two modes:
    1. RLVE mode: Fixed parallelism optimized for long sequences (TP=2, CP=2)
    2. Standard mode: Full auto-detection based on model size and sequence length

    Args:
        model_config: HuggingFace model config dict
        num_gpus: Available GPUs
        max_seq_len: Maximum sequence length (affects CP decision)
        rlve_enabled: If True, use RLVE-optimized fixed config
        model_name: Optional model name for parameter estimation

    Returns:
        Dict with tp, pp, cp, dp values
    """
    if rlve_enabled:
        # RLVE MODE: Fixed parallelism for long sequences
        # TP=2, CP=2 → DP=1 on 4 GPUs (all GPUs work together on same batch)
        tp = int(os.environ.get('SLIME_RLVE_TP', '2'))
        pp = 1  # No pipeline parallel for RLVE (simpler, less latency)
        cp = int(os.environ.get('SLIME_RLVE_CP', '2'))
        dp = num_gpus // (tp * pp * cp)
        logger.info(f"RLVE parallelism: TP={tp}, PP={pp}, CP={cp}, DP={max(1, dp)}")
        return {'tp': tp, 'pp': pp, 'cp': cp, 'dp': max(1, dp)}

    # STANDARD MODE: Full auto-detection
    num_params = estimate_model_params(model_config, model_name)
    logger.info(f"Auto-detecting parallelism for {num_params:.2f}B params, {num_gpus} GPUs, max_seq_len={max_seq_len}")

    # Check for environment override (for debugging/testing)
    env_tp = os.environ.get('SLIME_DEFAULT_TP')
    if env_tp:
        tp = int(env_tp)
        logger.info(f"Using SLIME_DEFAULT_TP override: TP={tp}")
    # TP: based on model size
    elif num_params < 2.0:  # <2B params
        tp = 1
    elif num_params < 10.0:  # 2-10B params
        tp = min(2, num_gpus)
    elif num_params < 30.0:  # 10-30B params
        tp = min(4, num_gpus)
    else:  # >30B params
        tp = min(8, num_gpus)

    # PP: for very large models that don't fit with TP alone
    if num_params >= 30.0:
        pp = 2
    else:
        pp = 1

    # CP: based on sequence length requirements
    env_cp = os.environ.get('SLIME_DEFAULT_CP')
    if env_cp:
        cp = int(env_cp)
        logger.info(f"Using SLIME_DEFAULT_CP override: CP={cp}")
    elif max_seq_len > 2048:
        cp = 2  # Long sequences benefit from context parallel
    else:
        cp = 1  # Short sequences don't need CP

    # DP: computed from remaining resources
    dp = num_gpus // (tp * pp * cp)

    # Fallback: ensure at least DP=1 by reducing CP/PP
    while dp < 1 and cp > 1:
        cp = 1
        dp = num_gpus // (tp * pp * cp)
        logger.warning(f"Reduced CP to 1 to ensure DP >= 1")
    while dp < 1 and pp > 1:
        pp = 1
        dp = num_gpus // (tp * pp * cp)
        logger.warning(f"Reduced PP to 1 to ensure DP >= 1")

    logger.info(f"Auto-detected parallelism: TP={tp}, PP={pp}, CP={cp}, DP={max(1, dp)}")
    return {'tp': tp, 'pp': pp, 'cp': cp, 'dp': max(1, dp)}


def detect_torch_dist_path(base_model: str) -> tuple[str, str]:
    """
    Auto-detect torch_dist model path from HF path or model name.

    Args:
        base_model: Base model path, HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct"),
                   or torch_dist format path

    Returns:
        Tuple of (megatron_checkpoint_path, hf_model_path)
    """
    # Get model directory from environment (default: /data/models)
    model_dir = os.getenv("HF_HOME", "/data/models")

    # If base_model looks like a HuggingFace model name (contains "/"), resolve to local path
    # e.g., "Qwen/Qwen2.5-7B-Instruct" -> "/data/models/Qwen2.5-7B-Instruct"
    if "/" in base_model and not base_model.startswith("/"):
        model_name = base_model.split("/")[-1]  # Get "Qwen2.5-7B-Instruct" from "Qwen/Qwen2.5-7B-Instruct"
        local_path = os.path.join(model_dir, model_name)
        if os.path.exists(local_path):
            logger.info(f"Resolved HF model name to local path: {base_model} → {local_path}")
            base_model = local_path
        else:
            logger.warning(f"HF model {base_model} not found at {local_path}, will use as-is")

    if not base_model.endswith('_torch_dist'):
        torch_dist_path = f"{base_model}_torch_dist"
        if os.path.exists(torch_dist_path):
            logger.info(
                f"Auto-detected torch_dist model: {base_model} → {torch_dist_path}"
            )
            return torch_dist_path, base_model
        else:
            logger.warning(
                f"No torch_dist version found at {torch_dist_path}, "
                f"using {base_model} as-is"
            )
            return base_model, base_model
    else:
        # Already torch_dist format
        hf_path = base_model.replace('_torch_dist', '')
        logger.info(f"Using torch_dist model: {base_model}")
        return base_model, hf_path


def parse_checkpoint_uri(checkpoint_path: str, save_dir: str = "/data/checkpoints/tinker") -> str:
    """
    Parse tinker:// URI to filesystem checkpoint path.

    Args:
        checkpoint_path: Checkpoint path (tinker:// URI or filesystem path)
        save_dir: Base directory for checkpoints

    Returns:
        Filesystem checkpoint path

    Raises:
        ValueError: If tinker:// URI format is invalid
    """
    if checkpoint_path.startswith("tinker://"):
        import hashlib

        uri_parts = checkpoint_path.replace("tinker://", "").split("/")
        if len(uri_parts) >= 3 and uri_parts[1] == "weights":
            checkpoint_name = uri_parts[2]
            # Use same step_id calculation as save_weights
            step_id = int(
                hashlib.md5(checkpoint_name.encode()).hexdigest()[:8], 16
            ) % 100000
            filesystem_path = f"{save_dir}/iter_{step_id:07d}"
            logger.info(
                f"Checkpoint resume: {checkpoint_path} → {filesystem_path} "
                f"(step_id={step_id})"
            )
            return filesystem_path
        else:
            raise ValueError(f"Invalid tinker:// URI format: {checkpoint_path}")
    else:
        # Direct filesystem path
        logger.info(f"Checkpoint resume: loading from {checkpoint_path}")
        return checkpoint_path


def extract_model_name(args) -> str:
    """
    Extract HuggingFace model name from Slime args.

    Args:
        args: Slime argument Namespace

    Returns:
        HuggingFace model path/name
    """
    if hasattr(args, 'hf_checkpoint') and args.hf_checkpoint:
        return args.hf_checkpoint

    if hasattr(args, 'pretrained_checkpoint') and args.pretrained_checkpoint:
        # Remove _torch_dist suffix if present
        return args.pretrained_checkpoint.replace('_torch_dist', '')

    logger.warning("Could not extract model name from args")
    return "unknown"


def compute_sglang_mem_fraction(model_config: Dict[str, Any], model_name: str = "") -> float:
    """
    Compute SGLang memory fraction based on model size.

    Smaller models need less GPU memory for KV cache, so we can use a smaller fraction.
    This allows colocated training without offload for small models.

    Args:
        model_config: Model configuration dict from load_model_config()
        model_name: Optional model name/path for extracting size

    Returns:
        Memory fraction (0.0-1.0) for SGLang's mem_fraction_static
    """
    total_params = estimate_model_params(model_config, model_name)

    # Memory fraction based on model size:
    # - Larger models need more KV cache memory
    # - Smaller models can use less, leaving room for Megatron
    if total_params <= 1.0:       # <= 1B: 10% (~19GB on H200)
        mem_fraction = 0.10
    elif total_params <= 2.0:     # 1-2B: 15% (~28GB)
        mem_fraction = 0.15
    elif total_params <= 4.0:     # 2-4B: 20% (~38GB)
        mem_fraction = 0.20
    elif total_params <= 8.0:     # 4-8B: 30% (~57GB)
        mem_fraction = 0.30
    elif total_params <= 14.0:    # 8-14B: 40% (~76GB)
        mem_fraction = 0.40
    elif total_params <= 35.0:    # 14-35B: 50% (~95GB)
        mem_fraction = 0.50
    else:                          # > 35B: 70% (~132GB)
        mem_fraction = 0.70

    logger.info(
        f"SGLang mem_fraction={mem_fraction:.2f} for {total_params:.1f}B model"
    )
    return mem_fraction


def detect_architecture(model_name: str) -> str:
    """
    Detect model architecture from model name.

    Args:
        model_name: Model name or path

    Returns:
        Architecture name (qwen2.5, llama, mistral, etc.)
    """
    short_name = model_name.split("/")[-1].lower()

    # Match common architectures
    if "qwen2.5" in short_name or "qwen-2.5" in short_name:
        return "qwen2.5"
    elif "qwen2" in short_name or "qwen-2" in short_name:
        return "qwen2"
    elif "qwen" in short_name:
        return "qwen"
    elif "llama-3" in short_name or "llama3" in short_name:
        return "llama3"
    elif "llama-2" in short_name or "llama2" in short_name:
        return "llama2"
    elif "llama" in short_name:
        return "llama"
    elif "mistral" in short_name:
        return "mistral"
    elif "mixtral" in short_name:
        return "mixtral"
    elif "phi" in short_name:
        return "phi"
    elif "gemma" in short_name:
        return "gemma"
    else:
        logger.warning(f"Unknown architecture for model: {model_name}")
        return "unknown"