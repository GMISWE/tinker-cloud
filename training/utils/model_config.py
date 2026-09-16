"""
Backend-agnostic model helpers: GPU count, HF config loading, parameter
estimation, architecture detection. Miles-specific resolution (torch_dist
paths, SGLang memory, parallelism auto-detect) lives in
backends/miles/model_setup.py.
"""
import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


def detect_num_gpus() -> int:
    """
    Auto-detect the number of available GPUs.

    Resolution order:
    1. NUM_GPUS env var (canonical, backend-agnostic)
    2. SLIME_NUM_GPUS env var (legacy Miles compat)
    3. Ray cluster GPU resources (if Ray is initialized)
    4. nvidia-smi device count
    5. Fallback to 1

    Returns:
        Number of GPUs available.
    """
    # 1. Canonical env var
    num_gpus_env = os.getenv("NUM_GPUS")
    if num_gpus_env:
        n = int(num_gpus_env)
        logger.info("GPU count from NUM_GPUS env: %d", n)
        return n

    # 2. Legacy env var
    slime_env = os.getenv("SLIME_NUM_GPUS")
    if slime_env:
        n = int(slime_env)
        logger.info("GPU count from SLIME_NUM_GPUS env: %d", n)
        return n

    # 3. Ray cluster resources (a cluster with no GPU resource is not a source)
    import ray
    if ray.is_initialized():
        n = int(ray.cluster_resources().get("GPU", 0))
        if n > 0:
            logger.info("GPU count from Ray cluster: %d", n)
            return n

    # 4. nvidia-smi; absent or hung binary means "not a source", any other failure propagates
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        n = len(result.stdout.strip().splitlines())
        if n > 0:
            logger.info("GPU count from nvidia-smi: %d", n)
            return n

    raise RuntimeError(
        "cannot determine the GPU count: set NUM_GPUS (no Ray GPU resources, no nvidia-smi devices)"
    )


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
            # transformers 5.x moved rope_theta into the rope_parameters dict
            'rotary_base': (
                getattr(config, 'rope_theta', None)
                or (getattr(config, 'rope_parameters', None) or {}).get('rope_theta')
                or 10000
            ),
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