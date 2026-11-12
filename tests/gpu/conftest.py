"""Every test under tests/gpu carries the `gpu` marker (registered in pyproject),
allows 25 minutes (vLLM boots in ~2) and runs in a Ray instance of its own,
started per module. `loop` serves the verl suites; `student` serves `test_verl_nk.py` and
`test_verl_optim.py`."""
import asyncio
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


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def student(loop):
    """(backend, handle) of the student; deleted after the module, and no actor may outlive it."""
    from tinkercloud.training.backends.verl.backend import VerlBackend

    from .verl_smollm3 import STUDENT, create_full_weights, verl_actors

    backend = VerlBackend()
    handle = create_full_weights(loop, backend, "nk_student", STUDENT)
    try:
        yield backend, handle
    finally:
        loop.run_until_complete(backend.delete_model(handle))
    assert verl_actors() == []
