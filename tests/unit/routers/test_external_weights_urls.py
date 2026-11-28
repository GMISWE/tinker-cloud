"""external_weights_urls: each signed URL opens its file as given, whatever the checkpoint and file are named.

The URLs are opened over a real socket with urllib, as `tinker checkpoint download` opens them: a test
client re-encodes and decodes paths on its own, so it can pass a URL no other client can open."""
import json
import socket
import threading
import time
import urllib.request
from urllib.parse import quote

import pytest
import uvicorn
from fastapi import FastAPI

from tinkercloud.training.checkpoints import CheckpointKind, CheckpointSaveOptions, CheckpointStore
from tinkercloud.training.config import AuthConfig, ExternalWeightsConfig, TrainingConfig
from tinkercloud.training.routers import checkpoints
from tinkercloud.training.storage.metadata import MetadataStorage
from tinkercloud.training.utils.auth import APIKeyAuth

MODEL = "model_e_0123456789abcdef"
FILES = {"adapter_config.json": b"{}", "sub/adapter model.safetensors": b"weights"}
SAVE = CheckpointSaveOptions(ttl_seconds=None, overwrite=True, user_metadata=None)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("external_weights")
    store = CheckpointStore(tmp / "ckpt", MetadataStorage(tmp / "meta"))
    app = FastAPI()
    app.include_router(checkpoints.router)
    app.state.checkpoint_store = store
    app.state.config = TrainingConfig(external_weights=ExternalWeightsConfig(url_signing_key="k"))
    app.state.auth = APIKeyAuth(AuthConfig(api_key="key"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    srv = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not srv.started:
        assert thread.is_alive(), "uvicorn exited before serving"
        time.sleep(0.01)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}", store
    srv.should_exit = True
    thread.join(timeout=10)


# urllib refuses a URL with a space or a non-ASCII character; a raw "%", "#" or "?" names another path
@pytest.mark.parametrize("name", ["hf1", "step 1", "é", "a%41", "v2#b", "x?y"])
def test_each_url_opens_its_file_with_urllib(server, name):
    base, store = server
    t = store.begin_save(MODEL, CheckpointKind.EXTERNAL_WEIGHTS, name, options=SAVE, optimizer_config={"type": "adamw"})
    for relpath, data in FILES.items():
        (t.root / relpath).parent.mkdir(parents=True, exist_ok=True)
        (t.root / relpath).write_bytes(data)
    store.complete(t.ref)

    listing = urllib.request.Request(
        f"{base}/api/v1/training_runs/{MODEL}/checkpoints/external_weights/{quote(name, safe='')}/external_weights_urls",
        headers={"X-API-Key": "key"})
    with urllib.request.urlopen(listing, timeout=10) as r:
        urls = json.load(r)["urls"]
    assert set(urls) == set(FILES)
    for relpath, url in urls.items():
        with urllib.request.urlopen(url, timeout=10) as r:
            assert r.read() == FILES[relpath]
