"""The chunk limit a server hands the SDK: the estimated request bytes at which the SDK closes a
forward or forward_backward chunk. Each chunk is a pass of its own on the server, so the value is the
backend's to declare."""
from tinkercloud.training.backends.base import TrainingBackend
from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.backends.verl.backend import VerlBackend


def test_the_config_route_serves_the_backends_chunk_limit(server, service_client):
    """A route that leaves the field out leaves the SDK on its own default, whatever the backend
    declares: the client's config must hold the server's value."""
    served = server.post("/api/v1/client/config", {"sdk_version": "0.38.0"}).json()["fwdbwd_max_chunk_bytes_count"]
    assert served == FakeBackend.FWDBWD_MAX_CHUNK_BYTES == TrainingBackend.FWDBWD_MAX_CHUNK_BYTES
    client = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    assert client.holder._client_config.fwdbwd_max_chunk_bytes_count == served


def test_verl_declares_a_larger_chunk_limit_than_the_base():
    """The CPU suites' guard for verl's override: without it verl serves the base class's value, the
    SDK's default, which cuts a long-sequence update into many passes, each padded to a row per GPU.
    Only a GPU test would notice."""
    assert VerlBackend.FWDBWD_MAX_CHUNK_BYTES > TrainingBackend.FWDBWD_MAX_CHUNK_BYTES
