"""
Helper utilities for the training API.

This module provides common helper functions used across the training API.
"""
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_ENGINE_NAME: Optional[str] = None
_ENGINE_ID = re.compile(r"^(?:req|model)_(?P<engine>[a-z0-9_]+)_[0-9a-f]{16}$")


def valid_engine_name(name: str) -> str:
    """`name` if it can live inside an id (lowercase letters, digits, underscores; not empty), else ValueError."""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"engine name {name!r}: lowercase letters, digits and underscores only")
    return name


def set_engine_name(name: str) -> None:
    """Name every id this process mints after its engine; the gateway routes an id by that name."""
    global _ENGINE_NAME
    _ENGINE_NAME = valid_engine_name(name)


def _mint(kind: str) -> str:
    if _ENGINE_NAME is None:
        raise RuntimeError("set_engine_name was not called: ids are minted by a named engine")
    return f"{kind}_{_ENGINE_NAME}_{uuid.uuid4().hex[:16]}"


def generate_request_id() -> str:
    """`req_<engine>_<hex16>`."""
    return _mint("req")


def generate_model_id() -> str:
    """`model_<engine>_<hex16>`."""
    return _mint("model")


def engine_name_of(id_: str) -> Optional[str]:
    """The engine an id names; None for an id in another shape (the engine answers for it)."""
    m = _ENGINE_ID.match(id_)
    return m.group("engine") if m else None


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
