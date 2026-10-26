"""Every test under tests/gpu carries the `gpu` marker (registered in pyproject),
allows 25 minutes (vLLM boots in ~2) and runs in a Ray instance of its own,
started per module."""
import os

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.gpu)
        item.add_marker(pytest.mark.timeout(1500))


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    import ray
    import torch
    ray.init(num_gpus=torch.cuda.device_count(), runtime_env={"env_vars": {
        "PYTHONPATH": os.environ["PYTHONPATH"], "HF_HUB_OFFLINE": "1"}})
    yield
    ray.shutdown()
