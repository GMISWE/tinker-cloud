"""The cleanup route validates retention before touching the futures store."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinkercloud.training.config import AuthConfig
from tinkercloud.training.core.runtime_state import TrainingRuntimeState
from tinkercloud.training.models.requests import CleanupFuturesRequest
from tinkercloud.training.routers import futures
from tinkercloud.training.storage.futures import FuturesStorage
from tinkercloud.training.storage.sample_futures import SampleFutureStore
from tinkercloud.training.utils.auth import APIKeyAuth


@pytest.fixture
def client(tmp_path):
    store = FuturesStorage(tmp_path / "futures.db")
    app = FastAPI()
    app.include_router(futures.router)
    app.state.futures_storage = store
    app.state.sample_futures = SampleFutureStore()
    app.state.runtime = TrainingRuntimeState()
    app.state.auth = APIKeyAuth(AuthConfig(api_key="test-key"))
    try:
        with TestClient(app, headers={"X-API-Key": "test-key"}) as c:
            yield c, store
    finally:
        store.close()


@pytest.mark.parametrize("hours", [-1, 0, 1, 2])
def test_cleanup_below_three_hours_is_422_and_leaves_the_result(client, hours):
    c, store = client
    result = {"metrics": {"grad_norm": 1.0}}
    store.save_future("r1", "optim_step", {}, model_id="m", seq_id=7)
    assert store.update_status("r1", "completed", result)

    response = c.post("/api/v1/cleanup_futures", json={"max_age_hours": hours})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "max_age_hours"]
    assert error["ctx"]["ge"] == 3
    assert "3" in error["msg"]
    assert store.get_future("r1")["result"] == result
    assert store.save_future("r2", "optim_step", {}, model_id="m", seq_id=7) == "r1"


@pytest.mark.parametrize("body", [{}, {"max_age_hours": 3}, {"max_age_hours": 24}])
def test_cleanup_accepts_the_minimum_and_default_and_the_result_is_retrievable(client, body):
    c, store = client
    result = {"metrics": {"grad_norm": 1.0}}
    store.save_future("r1", "optim_step", {}, model_id="m", seq_id=7)
    assert store.update_status("r1", "completed", result)

    response = c.post("/api/v1/cleanup_futures", json=body)
    assert response.status_code == 200
    assert response.json()["futures_cleaned"] == 0
    response = c.post("/api/v1/retrieve_future", json={"request_id": "r1"})
    assert response.status_code == 200
    assert response.json() == result
    assert store.save_future("r2", "optim_step", {}, model_id="m", seq_id=7) == "r1"


def test_cleanup_schema_names_the_minimum_and_keeps_the_24_hour_default():
    field = CleanupFuturesRequest.model_json_schema()["properties"]["max_age_hours"]
    assert field["minimum"] == 3
    assert field["default"] == CleanupFuturesRequest().max_age_hours == 24
