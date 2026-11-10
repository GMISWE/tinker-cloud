"""gspo and inflight samplers are served where their meaning is implemented
(automodel, fake) and refused elsewhere by the per-backend declarations."""
import pytest

from tinkercloud.training.backends.factory import BackendFactory
from tinkercloud.training.backends.fake.backend import FakeBackend

OTHERS = ("nemo_rl", "miles", "verl", "megatron_bridge")


def _cls(backend_type):
    try:
        return BackendFactory.backend_class(backend_type)
    except ImportError as e:
        pytest.skip(f"{backend_type} runtime not installed: {e}")


@pytest.mark.parametrize("backend_type", OTHERS)
def test_other_backends_serve_neither(backend_type):
    cls = _cls(backend_type)
    assert "gspo" not in cls.SUPPORTED_LOSS_FNS
    assert cls.SUPPORTED_SAMPLER_KINDS == frozenset({"snapshot"})


def test_automodel_and_fake_serve_both():
    for cls in (_cls("automodel"), FakeBackend):
        assert "gspo" in cls.SUPPORTED_LOSS_FNS
        assert cls.SUPPORTED_SAMPLER_KINDS == frozenset({"snapshot", "inflight"})


@pytest.mark.parametrize("backend_type", (*OTHERS, "automodel", "fake"))
def test_verl_alone_reads_nk_targets(backend_type):
    """A backend that declared rank 2 without reading it would misread [N, K] as N * K positions."""
    assert _cls(backend_type).TARGET_TOKENS_MAX_RANK == (2 if backend_type == "verl" else 1)
