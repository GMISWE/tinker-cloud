"""The routing decision: pure CPU, returns a destination or a status, never bytes.

`Decider.decide` reads the API key, the id the request carries (path, query, JSON field or
field 1 of the proto skeleton), the sampler cache and the registry, and answers with a
`Route` (the engine, the headers to add, what to record) or a `Reject` (a status and a
message). It never awaits; the store it touches is the control path's (sessions on
`create_model`), and a sampler lookup reads the store only on a cache miss.
"""
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

from pydantic import ValidationError

from ..models.requests import CreateModelRequest, LoadWeightsRequest
from ..utils.helpers import BASE_MODEL_SAMPLING_UNSUPPORTED, MODEL_ID_HEADER, PINNED_VERSION_HEADER, mint_id
from . import routing
from .registry import EngineRow, Registry
from .sessions import SamplerCache, SessionStore

Headers = List[Tuple[bytes, bytes]]

PUBLIC_PATHS = frozenset({
    "/health", "/api/v1/health", "/api/v1/healthz",
    "/api/v1/client/config", "/api/v1/client/dynamic_config", "/api/v1/telemetry",
})
PUBLIC_PREFIXES = ("/api/v1/external_weights/",)   # signed URLs carry their own token
CREATE_MODEL = "/api/v1/create_model"
LOAD_WEIGHTS = "/api/v1/load_weights"
SAMPLE_OPS = frozenset({"/api/v1/asample", "/api/v1/sample", "/api/v1/create_sampling_client"})
RETRIEVE_FUTURES = "/api/v1/retrieve_futures"
DROP_MODEL_OPS = frozenset({"/api/v1/delete_model", "/api/v1/unload_model"})
_MODEL_ID_HEADER = MODEL_ID_HEADER.lower().encode()
_PINNED_HEADER = PINNED_VERSION_HEADER.lower().encode()


@dataclass(frozen=True)
class ModelRecord:
    """A model to record at submit: the engine that gets it, the session that owns it."""
    model_id: str
    session_id: str
    model_seq_id: int
    base_model: Optional[str]   # None: a create-via-load that names none (the checkpoint's)


@dataclass(frozen=True)
class Route:
    engine: EngineRow
    add_headers: Headers
    record: Optional[ModelRecord] = None
    drop_model: Optional[str] = None   # a delete or unload: the model's row goes once the engine has answered


@dataclass(frozen=True)
class Reject:
    status: int
    message: str


def _header(headers: Headers, name: bytes) -> bytes:
    for k, v in headers:
        if k.lower() == name:
            return v
    return b""


class Decider:
    def __init__(self, keys: Optional[Set[str]], key_header: str, registry: Registry,
                 samplers: SamplerCache, sessions: SessionStore) -> None:
        self._keys = keys                              # None: auth disabled
        self._key_header = key_header.lower().encode()
        self._registry = registry
        self._samplers = samplers
        self._sessions = sessions

    def authorized(self, path: str, headers: Headers) -> Optional[str]:
        """None when the request may pass; else the 401 message."""
        if self._keys is None or path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return None
        key = _header(headers, self._key_header).decode("latin-1")
        if not key:
            return f"Missing API key (provide via {self._key_header.decode()} header)"
        if key not in self._keys:
            return "Invalid API key"
        return None

    def decide(self, path: str, query: bytes, headers: Headers, body: Optional[bytes]) -> Union[Route, Reject]:
        """`body` is the whole body when `routing.reads_body(path)`, else None (it streams past the decision)."""
        denied = self.authorized(path, headers)
        if denied is not None:
            return Reject(401, denied)
        if not _header(headers, b"host"):
            return Reject(400, "a Host header is required: the engine builds the URLs it returns from it")
        engine = self._registry.only()
        if engine is None:
            return Reject(503, "no engine registered")
        content_type = _header(headers, b"content-type")
        content_encoding = _header(headers, b"content-encoding")
        doc = routing.json_doc(content_type, content_encoding, body) if body is not None and routing.reads_body(path) else None
        named = routing.named_engine(path, query, content_type, content_encoding, body, doc)
        if named is not None and named[1] != engine.name:
            return Reject(404, f"no engine named {named[1]!r} is registered for id {named[0]!r}")
        if doc is None:
            return Route(engine, [])
        # `doc` is the SDK's body as sent; what a missing key means is each branch's to say
        if path in SAMPLE_OPS:
            return self._sampling(engine, doc.get("sampling_session_id"))   # absent: the engine resolves model_path or base_model itself
        if path == RETRIEVE_FUTURES:
            target = doc.get("target")   # absent: the engine's 422
            sid = target.get("sampling_session_id") if isinstance(target, dict) else None   # absent: the engine's 422
            if isinstance(sid, str) and self._samplers.lookup(sid) is None:
                return Reject(404, f"Unknown sampling_session_id: {sid}")
            return Route(engine, [])
        if path == CREATE_MODEL:
            return self._create_model(engine, doc)
        if path == LOAD_WEIGHTS and "session_id" in doc:
            return self._create_via_load(engine, doc)
        if path in DROP_MODEL_OPS:
            model_id = doc.get("model_id")   # absent: the engine's 422, and no row is named
            return Route(engine, [], drop_model=model_id if isinstance(model_id, str) else None)
        return Route(engine, [])

    def _sampling(self, engine: EngineRow, sid: object) -> Union[Route, Reject]:
        """A sampler names its model and the version it pinned; both travel as headers.
        A request that names no sampler (model_path, base_model) is the engine's to resolve."""
        if not isinstance(sid, str) or not sid:
            return Route(engine, [])
        sampler = self._samplers.lookup(sid)
        if sampler is None:
            return Reject(404, f"Unknown sampling_session_id: {sid}")
        if not sampler.model_id:
            return Reject(400, BASE_MODEL_SAMPLING_UNSUPPORTED)
        add: Headers = [(_MODEL_ID_HEADER, sampler.model_id.encode())]
        if sampler.pinned_version is not None:
            add.append((_PINNED_HEADER, str(sampler.pinned_version).encode()))
        return Route(engine, add)

    def _create_model(self, engine: EngineRow, doc: dict) -> Union[Route, Reject]:
        """The gateway mints the model id and records the model before the engine sees the request;
        the engine creates the model under that id. The body is read with the engine's own request
        model, so what the engine would accept is recorded and what it would refuse is its 422, as today."""
        try:
            req = CreateModelRequest.model_validate(doc)
        except ValidationError:
            return Route(engine, [])
        if not self._sessions.exists(req.session_id):
            return Reject(404, f"Session not found: {req.session_id}")
        model_id = mint_id("model", engine.name)
        return Route(engine, [(_MODEL_ID_HEADER, model_id.encode())],
                     record=ModelRecord(model_id, req.session_id, req.model_seq_id, req.base_model))

    def _create_via_load(self, engine: EngineRow, doc: dict) -> Union[Route, Reject]:
        """A load addressed by session creates its model: the SDK derives the id
        `{session_id}:train:{model_seq_id}` and the gateway records it under that id.
        A body that fails LoadWeightsRequest is the engine's 400 or 422 and records nothing."""
        try:
            req = LoadWeightsRequest.model_validate(doc)
            creates = req.creates()
        except (ValidationError, ValueError):
            return Route(engine, [])
        if not creates:
            return Route(engine, [])
        assert req.session_id is not None and req.model_seq_id is not None  # creates() checked both
        if not self._sessions.exists(req.session_id):
            return Reject(404, f"Session not found: {req.session_id}")
        model_id = f"{req.session_id}:train:{req.model_seq_id}"
        return Route(engine, [], record=ModelRecord(model_id, req.session_id, req.model_seq_id, req.base_model))
