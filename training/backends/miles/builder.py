"""
Miles argument builder — wraps SlimeArgumentBuilder behind the
ArgumentBuilder ABC.

The actual logic remains in training/core/slime_builder.py. This module
re-exports it through the backend interface.
"""
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder
from ...core.slime_builder import SlimeArgumentBuilder


class MilesArgumentBuilder(ArgumentBuilder):
    """Adapter: SlimeArgumentBuilder → ArgumentBuilder ABC."""

    def __init__(self, default_save_dir: str = "/data/checkpoints/tinker"):
        self._inner = SlimeArgumentBuilder(default_save_dir=default_save_dir)

    def build_args(
        self,
        base_model: str,
        num_gpus: int = 4,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Delegate to SlimeArgumentBuilder.build_args() with compatible signature."""
        return self._inner.build_args(
            base_model=base_model,
            lora_config=lora_config or {},
            debug_train_only=kwargs.get("debug_train_only", False),
            checkpoint_path=kwargs.get("checkpoint_path"),
            parallelism_config=parallelism,
            max_batch_size=kwargs.get("max_batch_size", 4096),
            max_seq_len=kwargs.get("max_seq_len", 2048),
            rlve_config=kwargs.get("rlve_config"),
            wandb_config=kwargs.get("wandb_config"),
        )
