"""The decision returns a destination or a status, never bytes, and never awaits."""
import asyncio

import orjson

from fastapi import FastAPI

from training.config import GatewayConfig
from training.gateway import forward
from training.gateway.app import build_app
from training.gateway.decide import Decider, Reject, Route
from training.gateway.registry import Registry
from training.gateway.sessions import SamplerCache, SessionStore
from training.proto import tinker_public_pb2 as pb

from .support import API_KEY, MODEL, config

HEADERS = [(b"host", b"sdk:8000"), (b"x-api-key", API_KEY.encode()), (b"content-type", b"application/json")]


def _decider():
    registry, store = Registry(), SessionStore(None)
    registry.register("fake", "fake", [], "/tmp/fake.sock", 1)
    return Decider({API_KEY}, "X-API-Key", registry, SamplerCache(store), store), store


def test_decide_is_synchronous_and_covers_every_shape():
    assert not asyncio.iscoroutinefunction(Decider.decide)
    d, store = _decider()
    assert isinstance(d.decide("/api/v1/healthz", b"", [(b"host", b"x")], None), Route)                 # public path, no key needed
    r = d.decide("/api/v1/create_model", b"", [(b"host", b"x")], b"{}")
    assert isinstance(r, Reject) and r.status == 401
    r = d.decide("/api/v1/create_model", b"", HEADERS[1:], b"{}")
    assert isinstance(r, Reject) and r.status == 400 and "Host" in r.message
    r = d.decide("/api/v1/optim_step", b"", HEADERS, orjson.dumps({"model_id": "model_other_0123456789abcdef"}))
    assert isinstance(r, Reject) and r.status == 404 and "'other'" in r.message
    sid = store.create_session([], {}, "t", None)
    store.add_model(sid, MODEL, 0, "fake/tiny", "fake")
    store.add_sampler("s_1", sid, MODEL, "fake/tiny", None, 5)
    r = d.decide("/api/v1/asample", b"", HEADERS, orjson.dumps({"sampling_session_id": "s_1", "prompt": {"tokens": [1]}}))
    assert isinstance(r, Route) and r.engine.name == "fake"
    assert r.add_headers == [(b"x-model-id", MODEL.encode()), (b"x-pinned-version", b"5")]
    proto = pb.ForwardBackwardRequest(model_id=MODEL, loss_fn="cross_entropy").SerializeToString()
    r = d.decide("/api/v1/forward_backward", b"", [(b"host", b"x"), (b"x-api-key", API_KEY.encode()), (b"content-type", b"application/x-protobuf")], proto)
    assert isinstance(r, Route) and r.add_headers == [] and r.record is None
    r = d.decide("/api/v1/create_model", b"", HEADERS, orjson.dumps({"session_id": sid, "model_seq_id": 1, "base_model": "fake/tiny"}))
    assert isinstance(r, Route) and r.record is not None and r.record.session_id == sid
    assert r.add_headers == [(b"x-model-id", r.record.model_id.encode())]
    r = d.decide("/api/v1/retrieve_future/req_fake_0123456789abcdef", b"", HEADERS, None)
    assert isinstance(r, Route) and r.add_headers == []


def test_the_sdk_face_is_one_fastapi_app_and_the_config_names_no_server():
    assert isinstance(build_app(config(), Registry(), SessionStore(None)), FastAPI)
    assert "server" not in GatewayConfig.model_fields


def test_only_forward_opens_engine_connections():
    """The decision module holds no client, pool or socket; forward.py is where bytes move."""
    import inspect

    import training.gateway.decide as decide_module
    src = inspect.getsource(decide_module)
    for word in ("aiohttp", "httpx", "socket", "open_unix_connection", "async def"):
        assert word not in src, word
    assert inspect.isclass(forward.Pool) and inspect.isclass(forward.AiohttpClient)
