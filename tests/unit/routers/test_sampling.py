"""routers.sampling: a prompt the router cannot read is a client error, refused before anything is queued."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinkercloud.training.config import AuthConfig
from tinkercloud.training.core.runtime_state import TrainingRuntimeState
from tinkercloud.training.routers import sampling
from tinkercloud.training.storage.sample_futures import SampleFutureStore
from tinkercloud.training.utils.auth import APIKeyAuth
from tinkercloud.training.utils.helpers import set_engine_name


@pytest.fixture
def client():
    set_engine_name("fake")
    app = FastAPI()
    app.include_router(sampling.router)
    app.state.auth = APIKeyAuth(AuthConfig(api_key="k"))
    app.state.runtime = TrainingRuntimeState()
    app.state.sample_futures = SampleFutureStore()
    # the prompt is read before the sampler is resolved, so neither is consulted
    app.state.sampling_service = SimpleNamespace()
    app.state.checkpoint_store = SimpleNamespace()
    with TestClient(app, headers={"X-API-Key": "k"}) as c:
        yield c, app.state.sample_futures


@pytest.mark.parametrize("prompt", [{"chunks": []}, {"tokens": []}, {}])
def test_a_prompt_with_no_tokens_is_a_400_and_registers_no_sample(client, prompt):
    c, sample_futures = client
    response = c.post("/api/v1/asample", json={
        "prompt": prompt, "sampling_session_id": "s", "seq_id": 0, "sampling_params": {"max_tokens": 4},
    })
    assert response.status_code == 400
    assert "prompt" in response.json()["detail"]
    assert sample_futures.get("anything") is None
