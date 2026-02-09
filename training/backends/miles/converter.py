"""
Miles data converter — wraps TinkerDataConverter behind the
DataConverter ABC.

The actual logic remains in training/core/data_converter.py. This module
re-exports it through the backend interface.
"""
from typing import Any, Dict, List

from ..base import DataConverter
from ...core.data_converter import TinkerDataConverter


class MilesDataConverter(DataConverter):
    """Adapter: TinkerDataConverter → DataConverter ABC."""

    def __init__(self):
        self._inner = TinkerDataConverter()

    def forward_to_backend(
        self,
        data: List[Dict],
        args: Any,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for forward pass."""
        return self._inner.forward_to_rollout(data)

    def forward_backward_to_backend(
        self,
        data: List[Dict],
        loss_fn: str,
        args: Any,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for training."""
        is_rl = not getattr(args, "debug_train_only", False)
        return self._inner.forward_backward_to_rollout(data, is_rl=is_rl)

    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert Miles forward result to Tinker format."""
        return self._inner.rollout_to_forward_result(
            result,
            loss_fn="cross_entropy",
            original_data=data,
        )

    def backend_to_forward_backward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert Miles forward_backward result to Tinker format."""
        return self._inner.rollout_to_forward_backward_result(
            result,
            loss_fn="cross_entropy",
            original_data=data,
        )
