"""
Utility modules for the training API.

This package provides utilities for authentication, model configuration,
and common helper functions.
"""
from .auth import APIKeyAuth, verify_api_key
from .helpers import (
    generate_request_id,
    generate_model_id,
    generate_step_id,
    format_timestamp,
    extract_error_message,
    validate_batch_data,
    parse_lora_config,
    merge_configs,
)
from .model_config import (
    load_model_config,
    estimate_model_params,
    get_parallelism_config,
    detect_torch_dist_path,
    parse_checkpoint_uri,
    compute_sglang_mem_fraction,
)

__all__ = [
    # Auth
    "APIKeyAuth",
    "verify_api_key",
    # Helpers
    "generate_request_id",
    "generate_model_id",
    "generate_step_id",
    "format_timestamp",
    "extract_error_message",
    "validate_batch_data",
    "parse_lora_config",
    "merge_configs",
    # Model config
    "load_model_config",
    "estimate_model_params",
    "get_parallelism_config",
    "detect_torch_dist_path",
    "parse_checkpoint_uri",
    "compute_sglang_mem_fraction",
]