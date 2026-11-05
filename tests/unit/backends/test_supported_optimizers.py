"""Each backend declares the optimizer families it serves;
the model service refuses the rest before the backend sees them."""
import pytest

from tinkercloud.training.backends.factory import BackendFactory
from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.models.requests import OPTIMIZER_FAMILIES, OptimizerConfig
from tinkercloud.training.services.model_service import ModelService

ADAMW_ONLY = ("nemo_rl", "miles", "verl", "megatron_bridge")


@pytest.mark.parametrize("backend_type", ADAMW_ONLY)
def test_megatron_side_backends_serve_adamw_only(backend_type):
    try:
        cls = BackendFactory.backend_class(backend_type)
    except ImportError as e:
        pytest.skip(f"{backend_type} runtime not installed: {e}")
    assert cls.SUPPORTED_OPTIMIZERS == frozenset({"adamw"})


def test_automodel_and_fake_serve_both_families():
    try:
        automodel = BackendFactory.backend_class("automodel")
    except ImportError as e:
        pytest.skip(f"automodel runtime not installed: {e}")
    assert automodel.SUPPORTED_OPTIMIZERS == frozenset({"adamw", "dimuon"}) == frozenset(OPTIMIZER_FAMILIES)
    assert FakeBackend.SUPPORTED_OPTIMIZERS == frozenset(OPTIMIZER_FAMILIES)


def test_service_refuses_a_family_the_backend_lacks_naming_both(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(FakeBackend, "SUPPORTED_OPTIMIZERS", frozenset({"adamw"}))
    service = ModelService(backend=backend, store=None, sample_futures=None)
    service.require_optimizer(OptimizerConfig(type="adamw"))
    with pytest.raises(ValueError, match="dimuon.*FakeBackend|FakeBackend.*dimuon"):
        service.require_optimizer(OptimizerConfig(type="dimuon", version=1))


def test_wire_check_on_the_config():
    OptimizerConfig(type="adamw").check()
    OptimizerConfig(type="dimuon", version=1).check()
    with pytest.raises(ValueError, match="version"):
        OptimizerConfig(type="dimuon", version=2).check()
    with pytest.raises(ValueError, match="version"):
        OptimizerConfig(type="dimuon").check()
    with pytest.raises(ValueError, match="version"):
        OptimizerConfig(type="adamw", version=1).check()
    with pytest.raises(ValueError, match="muon"):
        OptimizerConfig(type="muon").check()
    assert OptimizerConfig(type="dimuon", version=1).record() == {"type": "dimuon", "version": 1}
    assert OptimizerConfig().record() == {"type": "adamw"}


def test_load_refuses_a_family_the_backend_lacks_naming_both(monkeypatch):
    from tinkercloud.training.checkpoints import CheckpointRef
    from tinkercloud.training.services.checkpoint_service import CheckpointService
    monkeypatch.setattr(FakeBackend, "SUPPORTED_OPTIMIZERS", frozenset({"adamw"}))
    service = CheckpointService(backend=FakeBackend(), store=None)
    ref = CheckpointRef.parse("tinker://m/weights/c1")
    client = {"optimizer_config": {"type": "adamw"}}
    for optimizer in (False, True):
        with pytest.raises(ValueError, match="dimuon.*FakeBackend|FakeBackend.*dimuon"):
            service.require_load_family(client, ref, optimizer, OptimizerConfig(type="dimuon", version=1))


def test_version_is_a_strict_integer():
    from pydantic import ValidationError
    for bad in (True, "1", 1.0):
        with pytest.raises(ValidationError):
            OptimizerConfig(type="dimuon", version=bad)
