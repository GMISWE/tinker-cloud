"""Every test under tests/gpu carries the `gpu` marker (registered in pyproject),
allows 25 minutes (vLLM boots in ~2) and runs in a Ray instance of its own,
started per module. `loop` serves the verl suites; `student` serves `test_verl_nk.py` and
`test_verl_optim.py`; `live_sdk` gives a test the tinker SDK against this tree's server."""
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


@pytest.fixture(scope="module")
def live_sdk(tmp_path_factory, ray_local):
    """`live_sdk(backend, **env)` boots this tree's API server on `backend`, as a driver of the module's
    Ray instance, and returns a `tinker.ServiceClient` on it. A test on the backend object skips what a
    client goes through: the SDK's encoding, HTTP, the wire codec, the request model and the queue.
    The servers are stopped after the module."""
    import ray
    import tinker

    from tests.conformance.conftest import API_KEY, REPO_ROOT, Server

    servers = []

    def start(backend: str, **env: str):
        server = Server(tmp_path_factory.mktemp("live"), backend=backend,
                        RAY_ADDRESS=ray.get_runtime_context().gcs_address,
                        # the server imports `training` from the tree, its workers what the tests' do
                        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ["PYTHONPATH"]]), **env)
        servers.append(server)
        server.wait_healthy(120)
        return tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY)

    yield start
    for server in servers:
        server.stop()
