"""
Configuration management for the training API.

This module provides Pydantic-based configuration models following kgateway's
patterns for environment-based configuration with validation and defaults.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ModelInfo(BaseModel):
    """Information about a supported model."""

    model_name: str = Field(..., description="Model identifier/path")
    max_context_length: int = Field(..., description="Maximum context length")
    supports_lora: bool = Field(True, description="Whether LoRA is supported")
    model_params: Optional[float] = Field(None, description="Model size in billions")


class StorageConfig(BaseModel):
    """Storage configuration."""

    metadata_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("METADATA_DIR", "/data/metadata")),
        description="Base directory for metadata storage"
    )
    futures_db_name: str = Field(
        default="futures.db",
        description="SQLite database filename"
    )
    max_future_age_hours: int = Field(
        default=24,
        description="Maximum age for futures before cleanup"
    )

    @property
    def futures_db_path(self) -> Path:
        """Get full path to futures database."""
        return self.metadata_dir / self.futures_db_name

    @property
    def training_runs_dir(self) -> Path:
        """Get training runs directory."""
        return self.metadata_dir / "training_runs"

    @property
    def checkpoints_dir(self) -> Path:
        """Get checkpoints directory."""
        return self.metadata_dir / "checkpoints"


class RayConfig(BaseModel):
    """Ray cluster configuration."""

    address: Optional[str] = Field(
        default_factory=lambda: os.getenv("RAY_ADDRESS"),
        description="Ray cluster address (e.g., ray://head:10001)"
    )
    dashboard_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("RAY_DASHBOARD_URL"),
        description="Ray dashboard URL for monitoring"
    )
    namespace: str = Field(
        default_factory=lambda: os.getenv("RAY_NAMESPACE", "default"),
        description="Ray namespace for isolation"
    )
    timeout: int = Field(
        default=30,
        description="Connection timeout in seconds"
    )
    ignore_reinit_error: bool = Field(
        default=True,
        description="Ignore Ray re-initialization errors"
    )

    @validator("address")
    def validate_address(cls, v):
        """Ensure Ray address is provided."""
        if not v:
            # Try common patterns
            if os.getenv("RAY_HEAD_SERVICE_HOST"):
                host = os.getenv("RAY_HEAD_SERVICE_HOST")
                port = os.getenv("RAY_HEAD_SERVICE_PORT_CLIENT", "10001")
                return f"ray://{host}:{port}"
            # Default to local Ray
            return "ray://localhost:10001"
        return v


class ParallelismConfig(BaseModel):
    """Model parallelism configuration."""

    tensor_parallel: int = Field(
        default=1,
        description="Tensor parallelism degree",
        ge=1,
        le=8
    )
    pipeline_parallel: int = Field(
        default=1,
        description="Pipeline parallelism degree",
        ge=1,
        le=8
    )
    data_parallel: Optional[int] = Field(
        default=None,
        description="Data parallelism degree (auto-calculated if None)"
    )
    world_size: Optional[int] = Field(
        default=None,
        description="Total GPUs (auto-calculated if None)"
    )

    @validator("world_size", always=True)
    def calculate_world_size(cls, v, values):
        """Calculate world size from parallelism settings."""
        if v is None:
            tp = values.get("tensor_parallel", 1)
            pp = values.get("pipeline_parallel", 1)
            dp = values.get("data_parallel", 1) or 1
            return tp * pp * dp
        return v


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(
        default_factory=lambda: os.getenv("TRAINING_HOST", "0.0.0.0"),
        description="Server host address"
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("TRAINING_PORT", "8000")),
        description="Server port"
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("KGATEWAY_LOG_LEVEL", "INFO").upper(),
        description="Logging level"
    )
    access_log: bool = Field(
        default_factory=lambda: os.getenv("KGATEWAY_ACCESS_LOG", "false").lower() == "true",
        description="Enable HTTP access logging"
    )


class AuthConfig(BaseModel):
    """Authentication configuration."""

    api_key: str = Field(
        default_factory=lambda: os.getenv("TINKER_API_KEY", "slime-dev-key"),
        description="API key for authentication"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )
    enabled: bool = Field(
        default=True,
        description="Enable API key authentication"
    )

    @validator("api_key")
    def validate_api_key(cls, v):
        """Ensure API key is not empty in production."""
        if not v or v == "slime-dev-key":
            env = os.getenv("ENV", "development")
            if env == "production":
                raise ValueError("Production requires a secure API key")
            logger.warning("Using default development API key")
        return v


class BackendConfig(BaseModel):
    """Backend selection and configuration."""

    backend_type: str = Field(
        default_factory=lambda: os.getenv("TINKERCLOUD_BACKEND", "miles"),
        description="Training backend: 'miles' or 'nemo_rl'"
    )
    backend_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific configuration overrides"
    )

    @validator("backend_type")
    def validate_backend_type(cls, v):
        """Ensure backend type is supported."""
        supported = ("miles", "nemo_rl")
        if v not in supported:
            raise ValueError(
                f"Unsupported backend: {v!r}. Supported: {', '.join(supported)}"
            )
        return v


class SlimeConfig(BaseModel):
    """Slime backend configuration."""

    cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("HF_HOME", "/data/models")),
        description="HuggingFace cache directory"
    )
    default_batch_size: int = Field(
        default=8,
        description="Default batch size for training"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Gradient accumulation steps"
    )
    mixed_precision: str = Field(
        default="bf16",
        description="Mixed precision training mode"
    )
    use_flash_attn: bool = Field(
        default=True,
        description="Use Flash Attention if available"
    )


class TrainingConfig(BaseModel):
    """Main training API configuration."""

    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Server configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configuration"
    )
    ray: RayConfig = Field(
        default_factory=RayConfig,
        description="Ray cluster configuration"
    )
    auth: AuthConfig = Field(
        default_factory=AuthConfig,
        description="Authentication configuration"
    )
    backend: BackendConfig = Field(
        default_factory=BackendConfig,
        description="Backend selection and configuration"
    )
    slime: SlimeConfig = Field(
        default_factory=SlimeConfig,
        description="Slime backend configuration"
    )
    supported_models: List[ModelInfo] = Field(
        default_factory=lambda: TrainingConfig._get_default_models(),
        description="List of supported models"
    )
    poll_tracking_enabled: bool = Field(
        default=True,
        description="Enable smart poll tracking for retrieve_future"
    )
    allow_partial_batches: bool = Field(
        default_factory=lambda: os.getenv("ALLOW_PARTIAL_BATCHES", "false").lower() == "true",
        description="Allow forward_backward batches that are not divisible by data-parallel size"
    )

    @staticmethod
    def _get_default_models() -> List[ModelInfo]:
        """Get default supported models from environment or hardcoded defaults."""
        env_models = os.getenv("SUPPORTED_MODELS")
        if env_models:
            try:
                models_data = json.loads(env_models)
                return [ModelInfo(**m) for m in models_data]
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse SUPPORTED_MODELS: {e}")

        # Default models
        return [
            ModelInfo(
                model_name="/data/models/Qwen2.5-0.5B-Instruct_torch_dist",
                max_context_length=512,
                supports_lora=True,
                model_params=0.5
            )
        ]

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """
        Create configuration from environment variables.

        Returns:
            TrainingConfig instance
        """
        config = cls()
        logger.info(f"Loaded configuration: log_level={config.server.log_level}")
        return config

    @classmethod
    def from_file(cls, file_path: Path) -> "TrainingConfig":
        """
        Load configuration from JSON or YAML file.

        Args:
            file_path: Path to configuration file

        Returns:
            TrainingConfig instance
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict(exclude_none=True)

    def save(self, file_path: Path) -> None:
        """
        Save configuration to file.

        Args:
            file_path: Path to save configuration
        """
        with open(file_path, "w") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                import yaml
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {file_path}")


# Global configuration instance
_config: Optional[TrainingConfig] = None


def get_config() -> TrainingConfig:
    """
    Get the global configuration instance.

    Returns:
        TrainingConfig instance
    """
    global _config
    if _config is None:
        _config = TrainingConfig.from_env()
    return _config


def set_config(config: TrainingConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: TrainingConfig instance
    """
    global _config
    _config = config
    logger.info("Updated global configuration")
