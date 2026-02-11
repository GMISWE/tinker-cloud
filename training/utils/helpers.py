"""
Helper utilities for the training API.

This module provides common helper functions used across the training API.
"""
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a unique request ID.

    Args:
        prefix: Prefix for the request ID

    Returns:
        Unique request ID in format: prefix_<hex16>
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def generate_model_id(prefix: str = "model") -> str:
    """
    Generate a unique model ID.

    Args:
        prefix: Prefix for the model ID

    Returns:
        Unique model ID in format: prefix_<hex16>
    """
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def generate_step_id(checkpoint_name: str, max_step: int = 100000) -> int:
    """
    Generate a consistent step ID from checkpoint name.

    Uses MD5 hash to ensure the same checkpoint name always
    maps to the same step ID, enabling deterministic checkpoint paths.

    Args:
        checkpoint_name: Name of the checkpoint
        max_step: Maximum step value (modulo)

    Returns:
        Step ID (integer)
    """
    hash_digest = hashlib.md5(checkpoint_name.encode()).hexdigest()
    return int(hash_digest[:8], 16) % max_step


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp as ISO string.

    Args:
        timestamp: Datetime to format (defaults to now)

    Returns:
        ISO formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    return timestamp.isoformat()


def extract_error_message(exception: Exception, include_type: bool = True) -> str:
    """
    Extract a clean error message from an exception.

    Args:
        exception: The exception to extract message from
        include_type: Whether to include exception type

    Returns:
        Formatted error message
    """
    if include_type:
        return f"{type(exception).__name__}: {str(exception)}"
    return str(exception)


def validate_batch_data(batch_data: Dict[str, Any]) -> None:
    """
    Validate batch data structure.

    Args:
        batch_data: Batch data to validate

    Raises:
        ValueError: If batch data is invalid
    """
    required_fields = ["prompts", "responses"]

    for field in required_fields:
        if field not in batch_data:
            raise ValueError(f"Missing required field: {field}")

    # Validate lengths match
    num_prompts = len(batch_data["prompts"])
    num_responses = len(batch_data["responses"])

    if num_prompts != num_responses:
        raise ValueError(
            f"Prompts and responses length mismatch: "
            f"{num_prompts} != {num_responses}"
        )

    if num_prompts == 0:
        raise ValueError("Empty batch data")


def calculate_batch_size(
    prompts: List[str],
    responses: List[str],
    group_size: Optional[int] = None
) -> int:
    """
    Calculate effective batch size.

    Args:
        prompts: List of prompts
        responses: List of responses
        group_size: Optional group size for grouped training

    Returns:
        Effective batch size
    """
    batch_size = len(prompts)

    if group_size and batch_size % group_size != 0:
        logger.warning(
            f"Batch size {batch_size} not divisible by group_size {group_size}"
        )

    return batch_size


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge configuration dicts with override.

    Args:
        base_config: Base configuration
        override_config: Override configuration (optional)

    Returns:
        Merged configuration dict
    """
    if override_config is None:
        return base_config.copy()

    merged = base_config.copy()
    merged.update(override_config)
    return merged


def parse_lora_config(lora_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse and validate LoRA configuration.

    Args:
        lora_config: LoRA configuration dict

    Returns:
        Validated LoRA configuration with defaults
    """
    if lora_config is None:
        lora_config = {}

    # Default values
    defaults = {
        "rank": 0,  # 0 = no LoRA (full fine-tuning)
        "alpha": 0,
        "dropout": 0.0,
        "seed": None,
        "train_unembed": True,
        "train_mlp": True,
        "train_attn": True,
    }

    # Merge with defaults
    config = defaults.copy()
    config.update(lora_config)

    # Validation
    if config["rank"] < 0:
        raise ValueError(f"LoRA rank must be >= 0, got {config['rank']}")

    if config["dropout"] < 0 or config["dropout"] > 1:
        raise ValueError(f"LoRA dropout must be in [0, 1], got {config['dropout']}")

    return config


def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize object to JSON-compatible format.

    Handles common non-serializable types like datetime, UUID, etc.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable version of object
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def format_size_bytes(size_bytes: int) -> str:
    """
    Format byte size as human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def extract_learning_rates(optimizer: Optional[Any]) -> Dict[str, float]:
    """
    Extract learning rates from optimizer param groups.

    Args:
        optimizer: PyTorch optimizer instance

    Returns:
        Dict mapping param group names to learning rates
    """
    learning_rates = {}

    if optimizer and hasattr(optimizer, 'param_groups'):
        for pg_id, param_group in enumerate(optimizer.param_groups):
            lr_key = f"lr_pg_{pg_id}"
            learning_rates[lr_key] = float(param_group.get('lr', 0.0))

    return learning_rates


def find_model_with_rollout_manager(training_clients: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Find first model capable of serving sampling requests.

    Checks for:
    1. Miles backend: has 'rollout_manager' (SGLang HTTP sampling)
    2. NeMo RL backend: has 'backend_handle' with backend_type='nemo_rl' (vLLM Ray sampling)

    Args:
        training_clients: Dict of training client info

    Returns:
        Model ID if found, None otherwise
    """
    for model_id, client_info in training_clients.items():
        if "rollout_manager" in client_info and client_info["rollout_manager"] is not None:
            logger.debug(f"Found model with RolloutManager (Miles): {model_id}")
            return model_id
        handle = client_info.get("backend_handle")
        if handle is not None and getattr(handle, "backend_type", None) == "nemo_rl":
            logger.debug(f"Found model with NemoRL backend: {model_id}")
            return model_id

    logger.warning("No model with RolloutManager or NemoRL backend found")
    return None