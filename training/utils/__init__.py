"""
Utility modules for the training API.

Authentication, backend-agnostic model helpers, and common helpers.
"""
from .auth import APIKeyAuth, verify_api_key
from .helpers import (
    set_engine_name,
    valid_engine_name,
    engine_name_of,
    generate_request_id,
    generate_model_id,
    format_timestamp,
    extract_error_message,
    validate_batch_data,
    parse_lora_config,
    merge_configs,
)
from .model_config import (
    detect_num_gpus,
    load_model_config,
    estimate_model_params,
)

__all__ = [
    # Auth
    "APIKeyAuth",
    "verify_api_key",
    # Helpers
    "set_engine_name",
    "valid_engine_name",
    "engine_name_of",
    "generate_request_id",
    "generate_model_id",
    "format_timestamp",
    "extract_error_message",
    "validate_batch_data",
    "parse_lora_config",
    "merge_configs",
    # Model config
    "detect_num_gpus",
    "load_model_config",
    "estimate_model_params",
]