"""routers.health: probes stay keyless, but live model ids go only to a caller holding the API key."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinkercloud.training.core.runtime_state import TrainingRuntimeState
from tinkercloud.training.routers import health
from tinkercloud.training.utils.auth import APIKeyAuth

MODEL = "model_e_0123456789abcdef"


def test_health_shows_model_ids_only_to_a_key_holder():
    app = FastAPI()
    app.include_router(health.router)
    app.state.runtime = TrainingRuntimeState(training_clients={MODEL: {}})
    app.state.auth = APIKeyAuth(api_key="k")
    with TestClient(app) as c:
        for path in ("/health", "/api/v1/health"):
            for headers in ({}, {"X-API-Key": "nope"}):
                body = c.get(path, headers=headers).json()
                assert body["status"] == "healthy" and body["model_ids"] is None, (path, headers)
            body = c.get(path, headers={"X-API-Key": "k"}).json()
            assert body["model_ids"] == [MODEL] and body["active_training_clients"] == 1
