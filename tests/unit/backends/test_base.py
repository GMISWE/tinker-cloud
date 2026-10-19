"""The TrainingBackend contract every backend meets: error types, the factory, the ABC and each backend's signatures,
the checkpoint-root rule (exercised on the CPU backend) and the sample()/prepare_for_generation() error contracts."""
import asyncio
import importlib.util
import inspect

import pytest

from tinkercloud.training.backends.base import (
    BackendError,
    TrainingBackend,
    UnsupportedFeatureError,
)
from tinkercloud.training.backends.factory import BackendFactory, SUPPORTED_BACKENDS

HAS_NEMO_RL = importlib.util.find_spec("nemo_rl") is not None


class TestBackendError:
    """Test BackendError exception behavior."""

    def test_error_message_format(self):
        err = BackendError("test error", backend="miles", operation="forward")
        assert "[miles]" in str(err)
        assert "forward" in str(err)
        assert "test error" in str(err)

    def test_error_with_original_exception(self):
        original = ValueError("original error")
        err = BackendError(
            "wrapper", backend="nemo_rl", operation="create_model",
            original_error=original,
        )
        assert err.original_error is original
        assert "ValueError" in str(err)

    def test_error_attributes(self):
        err = BackendError("msg", backend="miles", operation="forward_backward")
        assert err.backend == "miles"
        assert err.operation == "forward_backward"


class TestUnsupportedFeatureError:
    """Test UnsupportedFeatureError subclass."""

    def test_inherits_backend_error(self):
        err = UnsupportedFeatureError("rlve_server", backend="nemo_rl")
        assert isinstance(err, BackendError)
        assert err.backend == "nemo_rl"
        assert err.operation == "feature_check"

    def test_with_suggestion(self):
        err = UnsupportedFeatureError(
            "rlve_server", backend="nemo_rl",
            suggestion="Use client-side RLVE via tinker-cookbook",
        )
        assert "tinker-cookbook" in str(err)


class TestBackendFactory:
    """Test BackendFactory dispatch."""

    def test_create_miles(self):
        backend = BackendFactory.create(backend_type="miles")
        assert backend is not None
        assert hasattr(backend, "create_model")

    def test_create_nemo_rl(self):
        backend = BackendFactory.create(backend_type="nemo_rl")
        assert backend is not None
        assert hasattr(backend, "create_model")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            BackendFactory.create(backend_type="invalid_backend")


class TestTrainingBackendABC:
    """Test that TrainingBackend ABC enforces method signatures."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            TrainingBackend()

    def test_required_methods(self):
        required = [
            "create_model", "forward", "forward_backward",
            "apply_optimizer_step", "update_inference_weights",
            "save_checkpoint", "load_checkpoint", "delete_model",
            "get_logprobs", "sample", "prepare_for_generation",
        ]
        for method_name in required:
            assert hasattr(TrainingBackend, method_name), (
                f"TrainingBackend missing required method: {method_name}"
            )


class TestBackendSignatureContract:
    """Every registered backend must accept what the services pass to the ABC methods."""

    _VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

    @pytest.mark.parametrize("backend_type", SUPPORTED_BACKENDS)
    def test_signatures_compatible_with_abc(self, backend_type):
        try:
            cls = BackendFactory.backend_class(backend_type)
        except ImportError as e:
            pytest.skip(f"{backend_type} runtime not installed: {e}")
        assert issubclass(cls, TrainingBackend)
        for name in sorted(TrainingBackend.__abstractmethods__):
            abc_sig = inspect.signature(getattr(TrainingBackend, name))
            impl_sig = inspect.signature(getattr(cls, name))
            impl_params = impl_sig.parameters
            impl_variadic = any(p.kind in self._VARIADIC for p in impl_params.values())
            for p in abc_sig.parameters.values():
                if p.kind in self._VARIADIC:
                    continue
                assert p.name in impl_params or impl_variadic, (
                    f"{cls.__name__}.{name} lacks parameter {p.name!r} declared by TrainingBackend"
                )
            for q in impl_params.values():
                if q.name in abc_sig.parameters or q.kind in self._VARIADIC:
                    continue
                assert q.default is not inspect.Parameter.empty, (
                    f"{cls.__name__}.{name} adds required parameter {q.name!r} absent from TrainingBackend"
                )

    def test_unknown_backend_class_raises(self):
        with pytest.raises(ValueError):
            BackendFactory.backend_class("nope")


class TestCheckpointRootContract:
    """save_checkpoint writes only under the root it is handed; persist=False
    writes nothing; load_checkpoint reads the same root back. The CPU adapter
    is the one that runs here; the GPU adapters share the signature (above)."""

    def test_fake_writes_under_root_and_round_trips(self, tmp_path):
        from tinkercloud.training.backends.fake.backend import FakeBackend
        b = FakeBackend()
        h = asyncio.run(b.create_model("m", "r", "fake/tiny", 0, native_root=tmp_path / "native"))
        h.w = 0.75
        root = tmp_path / "weights" / "c1"
        root.mkdir(parents=True)
        asyncio.run(b.save_checkpoint(h, root, step=3, persist=True))
        written = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
        assert written == {root.relative_to(tmp_path) / "fake_state.json"}
        h2 = asyncio.run(b.create_model("m2", "r", "fake/tiny", 0, native_root=tmp_path / "n2"))
        asyncio.run(b.load_checkpoint(h2, root))
        assert h2.w == 0.75

    def test_persist_false_writes_nothing(self, tmp_path):
        from tinkercloud.training.backends.fake.backend import FakeBackend
        b = FakeBackend()
        h = asyncio.run(b.create_model("m", "r", "fake/tiny", 0, native_root=tmp_path / "native"))
        root = tmp_path / "sampler_weights" / "e1"
        asyncio.run(b.save_checkpoint(h, root, step=None, persist=False))
        assert not root.exists()


class TestSamplingContract:
    """Test sample()/prepare_for_generation() error contracts (no GPU needed)."""

    def test_miles_sample_without_router_raises(self):
        from tinkercloud.training.backends.miles.backend import MilesBackend, MilesHandle
        backend = MilesBackend()
        handle = MilesHandle(model_id="test", backend_type="miles")
        with pytest.raises(BackendError, match="router not available"):
            asyncio.run(backend.sample(handle, "req-1", [1, 2, 3], num_samples=1))

    def test_miles_prepare_without_router_raises(self):
        from tinkercloud.training.backends.miles.backend import MilesBackend, MilesHandle
        backend = MilesBackend()
        handle = MilesHandle(model_id="test", backend_type="miles")
        with pytest.raises(BackendError, match="router not available"):
            asyncio.run(backend.prepare_for_generation(handle))

    def test_nemo_rl_sample_without_generation_raises(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend, NemoRLHandle
        backend = NemoRLBackend()
        handle = NemoRLHandle(model_id="test", backend_type="nemo_rl")
        with pytest.raises(BackendError, match="generation engine not initialized"):
            asyncio.run(backend.sample(handle, "req-1", [1, 2, 3], num_samples=1))

    def test_nemo_rl_prepare_without_generation_raises(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend, NemoRLHandle
        backend = NemoRLBackend()
        handle = NemoRLHandle(model_id="test", backend_type="nemo_rl")
        with pytest.raises(BackendError, match="generation engine not initialized"):
            asyncio.run(backend.prepare_for_generation(handle))

    @pytest.mark.skipif(not HAS_NEMO_RL, reason="nemo_rl not installed")
    def test_nemo_rl_configless_ppo_clips_at_contract_defaults(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle, _step_loss_fn
        from tinkercloud.training.backends.nemo_rl.loss_config import TINKER_PG_LOSS_DEFAULTS
        from tinkercloud.training.backends.nemo_rl.losses import TinkerSumPGLoss
        h = NemoRLHandle(model_id="t", backend_type="nemo_rl",
                         config={"loss_fn": dict(TINKER_PG_LOSS_DEFAULTS)},
                         loss_fn=TinkerSumPGLoss(TINKER_PG_LOSS_DEFAULTS), loss_fn_name="ppo")
        fn = _step_loss_fn(h, None)
        assert fn is not h.loss_fn
        assert (fn.ratio_clip_min, fn.ratio_clip_max) == pytest.approx((0.2, 0.2))
        fn = _step_loss_fn(h, {"clip_low_threshold": 0.9, "clip_high_threshold": 1.3})
        assert (fn.ratio_clip_min, fn.ratio_clip_max) == pytest.approx((0.1, 0.3))
        h.loss_fn_name = "importance_sampling"
        assert _step_loss_fn(h, None) is h.loss_fn
