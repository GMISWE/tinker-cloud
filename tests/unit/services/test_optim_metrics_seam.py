"""grad_norm must reach SDK clients via the metrics dict.

The SDK's OptimStepResponse carries only `metrics` (upstream shape), so a
grad_norm left at the top level of the optim_step payload is silently
dropped at the client seam — probe clients reading per-cycle grad norms
got null on every cycle. The service now mirrors it
into metrics; these tests keep that seam checked.
"""

import asyncio

import pytest

try:
    from training.services.training_service import TrainingService
except ImportError:  # pragma: no cover
    TrainingService = None


pytestmark = pytest.mark.skipif(
    TrainingService is None, reason="training package not importable"
)


class _StubBackend:
    def __init__(self, result):
        self._result = result

    async def apply_optimizer_step(self, handle, learning_rate=None, adam_params=None):
        return dict(self._result)


def _step(result):
    svc = TrainingService(backend=_StubBackend(result))
    return asyncio.run(svc.apply_optimizer_step("m1", {"backend_handle": object()}))


def test_grad_norm_mirrored_into_metrics():
    result = _step({"grad_norm": 2.5, "success": True})
    assert result["metrics"]["grad_norm"] == 2.5
    assert result["grad_norm"] == 2.5  # top level kept for raw-HTTP clients


def test_existing_metrics_not_clobbered():
    result = _step({"grad_norm": 2.5, "metrics": {"grad_norm": 7.0, "total_loss": 0.1}})
    assert result["metrics"]["grad_norm"] == 7.0  # backend's own value wins
    assert result["metrics"]["total_loss"] == 0.1


def test_absent_grad_norm_leaves_metrics_alone():
    result = _step({"success": True})
    assert "grad_norm" not in (result.get("metrics") or {})
