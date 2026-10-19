"""core.loss_registry validation and the service-level backend support check."""
import asyncio

import pytest

from tinkercloud.training.core import loss_registry
from tinkercloud.training.backends.base import BackendHandle, TrainingBackend, UnsupportedFeatureError
from tinkercloud.training.services.training_service import TrainingService


@pytest.mark.parametrize("name", sorted(loss_registry.LOSS_FNS))
def test_every_name_accepts_empty_config(name):
    loss_registry.validate(name, None)
    loss_registry.validate(name, {})


def test_unknown_name_rejected():
    with pytest.raises(ValueError, match="Unknown loss_fn"):
        loss_registry.validate("ppo_loss", None)


@pytest.mark.parametrize("name,cfg", [
    ("cross_entropy", {"beta": 1.0}),
    ("importance_sampling", {"clip_low_threshold": 0.9}),
    ("ppo", {"beta": 0.1}),
    ("dro", {"clip_low_threshold": 0.9}),
])
def test_disallowed_key_rejected(name, cfg):
    with pytest.raises(ValueError, match="not valid for"):
        loss_registry.validate(name, cfg)


def test_clip_thresholds_bracket_one():
    loss_registry.validate("ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2})
    with pytest.raises(ValueError, match="clip thresholds"):
        loss_registry.validate("cispo", {"clip_low_threshold": 1.1})
    with pytest.raises(ValueError, match="must be a number"):
        loss_registry.validate("ppo", {"clip_low_threshold": "0.8"})


def test_clip_thresholds_defaults():
    assert loss_registry.clip_thresholds(None) == (0.8, 1.2)
    assert loss_registry.clip_thresholds({"clip_high_threshold": 1.3}) == (0.8, 1.3)


class _Stub(TrainingBackend):
    SUPPORTED_LOSS_FNS = frozenset({"cross_entropy"})

    def __init__(self):
        self.calls = []

    async def create_model(self, *a, **k): ...
    async def forward(self, handle, data, loss_fn, loss_fn_config=None):
        self.calls.append(("forward", loss_fn, loss_fn_config)); return {"loss_fn_outputs": [], "metrics": {}}
    async def forward_backward(self, handle, data, loss_fn, loss_fn_config=None):
        self.calls.append(("fb", loss_fn, loss_fn_config)); return {"loss_fn_outputs": [], "metrics": {}}
    async def apply_optimizer_step(self, handle, learning_rate=None, adam_params=None): ...
    async def update_inference_weights(self, handle): ...
    async def save_checkpoint(self, handle, checkpoint_path, step_id=None): ...
    async def load_checkpoint(self, handle, checkpoint_path): ...
    async def delete_model(self, handle): ...
    async def get_logprobs(self, handle, data): ...
    async def sample(self, handle, request_id, prompt_tokens, num_samples, sampling_params=None,
                     prompt_logprobs=False, pinned_version=None): ...
    async def prepare_for_generation(self, handle): ...


def test_service_rejects_unsupported_loss_before_backend_call():
    stub = _Stub()
    svc = TrainingService(backend=stub)
    ci = {"backend_handle": BackendHandle(model_id="m", backend_type="stub")}
    with pytest.raises(UnsupportedFeatureError, match="ppo"):
        asyncio.run(svc.forward_backward("m", [], "ppo", ci, loss_fn_config={"clip_low_threshold": 0.9}))
    assert stub.calls == []
    asyncio.run(svc.forward_backward("m", [], "cross_entropy", ci))
    asyncio.run(svc.forward("m", [], "cross_entropy", ci, loss_fn_config=None))
    assert stub.calls == [("fb", "cross_entropy", None), ("forward", "cross_entropy", None)]
