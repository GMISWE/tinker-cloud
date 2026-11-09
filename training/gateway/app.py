"""The gateway: the SDK's face on the TCP listener, the engines' face on the unix socket.

The SDK face is a FastAPI app. It answers the session, sampler and health routes itself and
forwards everything else: `Decider.decide` picks the engine and the headers (the decision),
`forward.forward` moves the bytes. The engines face is plain ASGI and takes register,
heartbeat, unregister and the samplers an engine mints. Both faces share one `Gateway`.
"""
import asyncio
import contextlib
import logging
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

import orjson
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.requests import ClientDisconnect

from ..config import TrainingConfig
from ..models.requests import (
    CreateSamplingSessionRequest,
    CreateSessionRequest,
    FinishSessionRequest,
    JoinSamplingSessionRequest,
    LOCAL_OWNER,
    DEFAULT_PROJECT,
    SessionHeartbeatRequest,
    SessionSearchRequest,
    UnknownProject,
)
from ..models.responses import (
    CreateSamplingSessionResponse,
    CreateSessionResponse,
    FinishSessionResponse,
    GetSamplerResponse,
    GetSessionResponse,
    HealthResponse,
    JoinSamplingSessionResponse,
    ListSessionsResponse,
    ResolveSamplerTargetResponse,
    SessionHeartbeatResponse,
    SessionSearchResponse,
    SessionSummary,
)
from ..utils.helpers import BASE_MODEL_SAMPLING_UNSUPPORTED, engine_name_of
from . import routing
from .decide import Decider, Headers, Reject
from .forward import EngineResponse, EngineUnreachable, Pool, forward
from .reaper import ModelReaper
from .search import search, status_of, str_metadata
from .registry import EngineRow, Registry, RegistryConflict, RegistryUnavailable
from .sessions import SamplerCache, SessionStore

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_S = 1.5   # the 503 for an engine that hangs lands inside the 2 s the face promises


class _Everything:
    def __contains__(self, _item: object) -> bool:
        return True


_EVERY_MODEL = _Everything()
_ENGINE_PATH = re.compile(r"^/engines/([a-z0-9_]+)(/heartbeat|/samplers|/models/([^/]+))?$")

ReadBody = Callable[[], Awaitable[bytes]]
StreamBody = Callable[[], Any]


class ClientDisconnected(Exception):
    """The SDK went away before its request body was complete; nothing is forwarded or answered."""


class Reply:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: Headers, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body


def _json_reply(status: int, doc: dict) -> Reply:
    body = orjson.dumps(doc)
    return Reply(status, [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())], body)


def _error(status: int, message: str, path: str) -> Reply:
    """The same shape as the engine's own error handler, so the SDK sees one error format."""
    return _json_reply(status, {"error": message, "status_code": status, "path": path})


class Gateway:
    """Everything the two faces share: the registry, the session store, the sampler cache, the
    engine pool, the decision, the reaper, and the gateway's own calls to an engine."""

    def __init__(self, config: TrainingConfig, registry: Registry, store: SessionStore) -> None:
        self.config = config
        self.registry = registry
        self.store = store
        self.samplers = SamplerCache(store)
        self.pool = Pool(config.gateway, registry)
        keys, self._own_key = config.auth.parsed_keys()   # None, None: auth off, the engine takes bare calls
        if keys is not None and not keys:
            raise ValueError("auth is enabled and the API key setting names no key")
        self.decider = Decider(keys, config.auth.api_key_header, registry, self.samplers, store)
        self.reaper = ModelReaper(self.free_model)
        self._tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._refresh_forever(), name="registry-refresh"))
        cfg = self.config
        if cfg.session_timeout_s >= 0:
            # every model of a claimed session is freed: the claim already took its rows, so "live" is everything
            self._tasks.append(asyncio.create_task(self.reaper.run_forever(
                cfg.session_reap_interval_s, lambda: self.store.claim_expired(cfg.session_timeout_s), lambda: _EVERY_MODEL),
                name="session-reaper"))
            logger.info("Session reaper enabled: timeout=%ss interval=%ss", cfg.session_timeout_s, cfg.session_reap_interval_s)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        await self.reaper.shutdown([])   # live models keep training through a gateway restart; pending reaps stay queued
        await self.pool.aclose()

    async def _refresh_forever(self) -> None:
        while True:
            await asyncio.sleep(self.config.gateway.registry_refresh_s)
            try:
                self.registry.refresh()
            except RegistryUnavailable:   # this period's rows stay; the next period reads again
                logger.exception("Registry refresh failed")

    # ------------------------------------------------------------ the forwarded path

    async def handle(self, method: str, path: str, query: bytes, headers: Headers,
                     read_body: ReadBody, stream_body: StreamBody) -> Reply:
        if routing.reads_body(path):
            body = await read_body()
        else:
            body = b"" if method in ("GET", "HEAD") else stream_body()
        decision = self.decider.decide(path, query, headers, body if isinstance(body, bytes) else None)
        if isinstance(decision, Reject) and decision.status == 503:
            # this worker knows no engine: another worker may have taken the registration since its last refresh
            self.registry.refresh()
            decision = self.decider.decide(path, query, headers, body if isinstance(body, bytes) else None)
        if isinstance(decision, Reject):
            return _error(decision.status, decision.message, path)
        if decision.record is not None:
            r = decision.record
            self.store.add_model(r.session_id, r.model_id, r.model_seq_id, r.base_model, decision.engine.name)
        try:
            resp = await forward(self.pool.client_for(decision.engine), method, path, query, headers, decision.add_headers, body)
        except EngineUnreachable as exc:
            if _caused_by(exc, ClientDisconnect):
                raise                 # the SDK left mid-body: the engine is fine and nobody is left to answer
            logger.warning("Engine %s unreachable for %s %s: %s", decision.engine.name, method, path, exc)
            self.registry.refresh()   # a replaced or departed engine is seen now, not at the next period
            return _error(503, f"engine {decision.engine.name} unreachable: {exc}", path)
        if decision.drop_model is not None and (200 <= resp.status < 300 or resp.status == 404):
            self.store.remove_model(decision.drop_model)
        if decision.record is not None and not 200 <= resp.status < 300:
            self.store.remove_model(decision.record.model_id)   # the engine refused the create: no model exists
        return Reply(resp.status, resp.headers, resp.body)

    # ------------------------------------------------------------ the gateway's own calls

    async def engine_call(self, engine: EngineRow, method: str, path: str, doc: Optional[dict]) -> EngineResponse:
        """A request of the gateway's own to an engine, with its API key; raises EngineUnreachable."""
        headers: Headers = [(b"host", b"gateway")]
        if self._own_key is not None:
            headers.append((self.config.auth.api_key_header.lower().encode(), self._own_key.encode()))
        body = b""
        if doc is not None:
            body = orjson.dumps(doc)
            headers.append((b"content-type", b"application/json"))
        return await forward(self.pool.client_for(engine), method, path, b"", headers, [], body)

    async def free_model(self, model_id: str, reason: str) -> None:
        """Delete a model on its engine and acknowledge the reap; never raises (FreeFn's contract)."""
        name = engine_name_of(model_id)
        # an id in another shape (create-via-load's `{session}:train:{seq}`) names no engine: its delete
        # goes where every request carrying it went, to the only engine (Decider.decide)
        engine = self.registry.get(name) if name else self.registry.only()
        logger.info("Freeing model %s (%s)", model_id, reason)
        delete_confirmed = False
        if engine is None:
            logger.warning("Model %s has no registered engine; its delete will be retried", model_id)
        else:
            try:
                resp = await self.engine_call(engine, "POST", "/api/v1/delete_model", {"model_id": model_id})
                if 200 <= resp.status < 300 or resp.status == 404:
                    delete_confirmed = True
                else:
                    logger.error("Engine %s answered %d to the delete of %s; it will be retried: %s",
                                 engine.name, resp.status, model_id, resp.body[:500])
            except asyncio.CancelledError:
                self.store.retry_reap(model_id)
                raise
            except EngineUnreachable as exc:
                logger.error("Engine %s unreachable; model %s delete will be retried: %s", engine.name, model_id, exc)
        if delete_confirmed:
            self.store.remove_model(model_id)
        else:
            self.store.retry_reap(model_id)

    async def health(self, path: str) -> Reply:
        """The engine's own health, with the engines the gateway knows; 503 while none answers."""
        engine = self.registry.only()
        if engine is None:
            self.registry.refresh()
            engine = self.registry.only()
        if engine is None:
            return _error(503, "no engine registered", path)
        try:
            resp = await asyncio.wait_for(self.engine_call(engine, "GET", "/health", None), HEALTH_TIMEOUT_S)
        except (EngineUnreachable, asyncio.TimeoutError) as exc:
            return _error(503, f"engine {engine.name} unreachable: {exc}", path)
        if resp.status != 200:
            return _error(503, f"engine {engine.name} answered {resp.status} to /health", path)
        health = HealthResponse.model_validate(orjson.loads(resp.body))   # the engine's body, checked once here
        health.engines = [{"name": engine.name, "state": "UP"}]
        return _json_reply(200, health.model_dump())


# ------------------------------------------------------------ the SDK face

def sdk_face(gateway: Gateway, engines) -> FastAPI:
    """The FastAPI app: the SDK's routes, with the engines face on the unix listener in front of them."""
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=_lifespan)
    app.state.gateway = gateway
    app.state.engines = engines
    app.add_middleware(ListenerSplit, engines=engines)
    store = gateway.store

    async def require_key(request: Request) -> None:
        denied = gateway.decider.authorized(request.url.path, request.headers.raw)
        if denied is not None:
            raise HTTPException(status_code=401, detail=denied)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status_code": exc.status_code, "path": request.url.path})

    @app.post("/api/v1/create_session", response_model=CreateSessionResponse)
    async def create_session(request: CreateSessionRequest, _: None = Depends(require_key)):
        # the SDK may send no metadata; the session reads back {} (its dict shape), as before the split
        session_id = store.create_session(request.tags, request.user_metadata or {}, request.sdk_version,  # no metadata reads back {}
                                          request.display_name)
        return CreateSessionResponse(session_id=session_id, info_message=None, warning_message=None, error_message=None)

    @app.post("/api/v1/session_heartbeat", response_model=SessionHeartbeatResponse)
    async def session_heartbeat(request: SessionHeartbeatRequest, _: None = Depends(require_key)):
        touched = store.touch(request.session_id)
        if touched is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
        if touched is False:   # the SDK stops its heartbeat loop on 410
            session = store.get_session(request.session_id)
            if session is None:   # finished, then reaped between the two reads
                raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
            raise HTTPException(status_code=410, detail=f"Session finished ({session.finish_reason}): {request.session_id}")
        return SessionHeartbeatResponse()

    @app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
    async def create_sampling_session(request: CreateSamplingSessionRequest, _: None = Depends(require_key)):
        """A sampler names its owning model: a tinker:// model_path does, through the engine that
        holds the checkpoint. A bare base_model is a base-model sampler when the engine serves that
        base model; the engine's verdict (400 when it does not) is the answer."""
        if not store.exists(request.session_id):
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
        if not request.model_path and not request.base_model:
            raise HTTPException(status_code=400, detail="Provide sampling_session_id or a tinker:// model_path")
        engine = gateway.registry.only()
        if engine is None:
            raise HTTPException(status_code=503, detail="no engine registered")
        if not request.model_path and not engine.base_model_sampling:
            raise HTTPException(status_code=400, detail=BASE_MODEL_SAMPLING_UNSUPPORTED)
        ask = {"model_path": request.model_path} if request.model_path else {"base_model": request.base_model}
        try:
            resp = await gateway.engine_call(engine, "POST", "/api/v1/resolve_sampler_target", ask)
        except EngineUnreachable as exc:
            raise HTTPException(status_code=503, detail=f"engine {engine.name} unreachable: {exc}") from exc
        if resp.status != 200:   # the engine's verdict on the path: 404 unknown, 425 still saving, 400 malformed
            return Response(content=resp.body, status_code=resp.status, headers=_str_headers(resp.headers))
        target = ResolveSamplerTargetResponse.model_validate(orjson.loads(resp.body))   # the engine's reply, checked once here
        if request.base_model and request.base_model != target.base_model:
            raise HTTPException(status_code=400, detail=f"base_model {request.base_model!r} is not the model's ({target.base_model!r})")
        sampling_session_id = f"{request.session_id}_{request.sampling_session_seq_id}_{uuid.uuid4().hex[:8]}"
        sampler = store.add_sampler(sampling_session_id, request.session_id, target.model_id,
                                    target.base_model, request.model_path, target.pinned_version)
        gateway.samplers.put(sampler)
        logger.info("Created sampling session %s (parent=%s, model=%s, pinned=%s)",
                    sampling_session_id, request.session_id, target.model_id, target.pinned_version)
        return CreateSamplingSessionResponse(sampling_session_id=sampling_session_id)

    @app.post("/api/v1/join_sampling_session", response_model=JoinSamplingSessionResponse)
    async def join_sampling_session(request: JoinSamplingSessionRequest, _: None = Depends(require_key)):
        """A client id for a new client (unpickled SamplingClient) of an existing sampling session:
        the creator is 0; joiners get 1, 2, ... and put their seq_ids in the block 1e9 * client id."""
        counter = store.join(request.sampling_session_id)
        if counter is None:
            raise HTTPException(status_code=404, detail=f"Sampling session not found: {request.sampling_session_id}")
        return JoinSamplingSessionResponse(client_counter=counter)

    @app.post("/api/v1/sessions/{session_id}/finish", response_model=FinishSessionResponse)
    async def finish_session(session_id: str, request: FinishSessionRequest, _: None = Depends(require_key)):
        """First-wins; models and samplers are untouched: the reaper frees them once heartbeats stop."""
        session = store.finish(session_id, request.reason.type, request.detail)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        assert session.finished_at is not None and session.finish_reason is not None
        return FinishSessionResponse(session_id=session_id, reason=session.finish_reason, detail=session.finish_detail,
                                     finished_at=session.finished_at)

    @app.get("/api/v1/sessions", response_model=ListSessionsResponse)
    async def list_sessions(limit: int = 20, offset: int = 0, _: None = Depends(require_key)):
        return ListSessionsResponse(sessions=store.list_sessions(limit, offset))

    @app.get("/api/v1/sessions/{session_id}", response_model=GetSessionResponse)
    async def get_session(session_id: str, _: None = Depends(require_key)):
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return GetSessionResponse(training_run_ids=session.model_ids, sampler_ids=session.sampler_ids,
                                  user_metadata=str_metadata(session.user_metadata), display_name=session.display_name)

    @app.post("/api/v1/sessions/search", response_model=SessionSearchResponse)
    async def search_sessions(request: SessionSearchRequest, _: None = Depends(require_key)):
        try:
            page, next_cursor = search(store.sessions_newest_first(), request)
        except UnknownProject as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        engine = gateway.registry.only()
        summaries = []
        for s in page:
            summaries.append(SessionSummary(
                session_id=s.session_id, created_at=s.created_at, owner=LOCAL_OWNER, project_id=DEFAULT_PROJECT,
                status=status_of(s), checkpoints_count=await _checkpoints_count(engine, s.model_ids),
                display_name=s.display_name, user_metadata=str_metadata(s.user_metadata)))
        return SessionSearchResponse(sessions=summaries, next_cursor=next_cursor)

    async def _checkpoints_count(engine: Optional[EngineRow], model_ids: List[str]) -> int:
        """Checkpoints of the session's training runs, as the engine lists them. A run the
        engine no longer holds (404) has none; with no engine registered the count is unknown."""
        if not model_ids:
            return 0
        if engine is None:
            raise HTTPException(status_code=503, detail="no engine registered")
        total = 0
        for model_id in model_ids:
            try:
                resp = await gateway.engine_call(engine, "GET", f"/api/v1/training_runs/{model_id}/checkpoints", None)
            except EngineUnreachable as exc:
                raise HTTPException(status_code=503, detail=f"engine {engine.name} unreachable: {exc}") from exc
            if resp.status == 404:
                continue
            if resp.status != 200:
                raise HTTPException(status_code=502, detail=f"engine listed {model_id}'s checkpoints with {resp.status}")
            total += len(orjson.loads(resp.body)["checkpoints"])
        return total

    @app.get("/api/v1/samplers/{sampler_id}", response_model=GetSamplerResponse)
    async def get_sampler(sampler_id: str, _: None = Depends(require_key)):
        sampler = gateway.samplers.lookup(sampler_id)
        if sampler is None:
            raise HTTPException(status_code=404, detail=f"Sampler not found: {sampler_id}")
        if sampler.base_model is None:
            raise HTTPException(status_code=500, detail=f"Sampler {sampler_id} was recorded without a base model")
        return GetSamplerResponse(sampler_id=sampler.sampler_id, base_model=sampler.base_model, model_path=sampler.model_path)

    @app.get("/health")
    @app.get("/api/v1/health")
    async def health(request: Request):
        try:
            return _response(await gateway.health(request.url.path))
        except RegistryUnavailable as exc:
            return _response(_error(503, f"engine registry busy: {exc}", request.url.path))

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def forward_everything_else(path: str, request: Request):
        # as the SDK encoded them: FastAPI's `path` is decoded, and `request.url` re-splits a decoded `?` into a query
        raw_path, query = request.scope["raw_path"].decode("latin-1"), request.scope["query_string"]
        try:
            reply = await gateway.handle(request.method, raw_path, query,
                                         request.headers.raw, request.body, request.stream)
        except ClientDisconnect:
            return Response(status_code=499)   # nobody is left to read it
        except EngineUnreachable as exc:
            # the engine saw a cut-off body because the SDK left mid-stream: nobody is left to read it either
            if _caused_by(exc, ClientDisconnect):
                return Response(status_code=499)
            raise
        except RegistryUnavailable as exc:
            return _response(_error(503, f"engine registry busy: {exc}", raw_path))
        return _response(reply)

    return app


def _caused_by(exc: BaseException, kind: type) -> bool:
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, kind):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def _str_headers(headers: Headers) -> Dict[str, str]:
    return {k.decode("latin-1"): v.decode("latin-1") for k, v in headers}


def _response(reply: Reply) -> Response:
    return Response(content=reply.body, status_code=reply.status, headers=_str_headers(reply.headers))


# ------------------------------------------------------------ the engines face

async def _read_all(receive) -> bytes:
    chunks: List[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ClientDisconnected()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_reply(send, reply: Reply) -> None:
    await send({"type": "http.response.start", "status": reply.status, "headers": reply.headers})
    await send({"type": "http.response.body", "body": reply.body})


def engines_face(registry: Registry):
    """register, heartbeat, unregister, list: what an engine needs from the registry alone."""
    async def app(scope, receive, send) -> None:
        method, path = scope["method"], scope["path"]
        try:
            body = await _read_all(receive)
        except ClientDisconnected:
            return
        if method == "GET" and path == "/engines":
            now = time.time()
            reply = _json_reply(200, {"engines": [r.as_dict(now) for r in registry.rows()]})
        elif method == "POST" and path == "/engines/register":
            reply = _register(registry, body, path)
        else:
            m = _ENGINE_PATH.match(path)
            if m is None:
                reply = _error(404, f"no such route: {method} {path}", path)
            elif method == "POST" and m.group(2) == "/heartbeat":
                reply = _json_reply(200, {}) if registry.heartbeat(m.group(1)) else _error(404, f"engine {m.group(1)!r} is not registered", path)
            elif method == "DELETE" and not m.group(2):
                reply = Reply(204, [], b"") if registry.unregister(m.group(1)) else _error(404, f"engine {m.group(1)!r} is not registered", path)
            else:
                reply = _error(404, f"no such route: {method} {path}", path)
        await _send_reply(send, reply)
    return app


def _store_route(path: str) -> bool:
    """`/engines/<n>/samplers` and `/engines/<n>/models/<id>` write the gateway's store; the rest is the registry's."""
    m = _ENGINE_PATH.match(path)
    return m is not None and (m.group(2) == "/samplers" or m.group(3) is not None)


def store_face(gateway: Gateway):
    """`samplers` (an engine's ephemeral sampler, recorded before its future completes) and `models/<id>`
    (a create that failed: the row goes)."""
    async def app(scope, receive, send) -> None:
        method, path = scope["method"], scope["path"]
        try:
            body = await _read_all(receive)
        except ClientDisconnected:
            return
        m = _ENGINE_PATH.match(path)
        if m is not None and method == "POST" and m.group(2) == "/samplers":
            reply = _sampler_created(gateway, m.group(1), body, path)
        elif m is not None and method == "DELETE" and m.group(3) is not None:
            reply = Reply(204, [], b"") if gateway.store.remove_model(m.group(3)) else _error(404, f"model {m.group(3)!r} is not recorded", path)
        else:
            reply = _error(404, f"no such route: {method} {path}", path)
        await _send_reply(send, reply)
    return app


def gateway_engines_face(registry: Registry, gateway: Gateway):
    """The engines listener: store routes to the store face, everything else to the registration face."""
    registration, store = engines_face(registry), store_face(gateway)

    async def app(scope, receive, send) -> None:
        # an engine holds the same API key as its SDK clients; without it nobody registers,
        # unregisters or records anything here, whatever the socket's permissions
        if not gateway.decider.key_valid(scope["headers"]):
            await _send_reply(send, _error(401, "an engine request needs the API key", scope["path"]))
            return
        await (store if _store_route(scope["path"]) else registration)(scope, receive, send)
    return app


def _register(registry: Registry, body: bytes, path: str) -> Reply:
    try:
        row = orjson.loads(body)
        gpus = row["gpus"]
        if gpus is not None and not (isinstance(gpus, list) and all(isinstance(g, int) for g in gpus)):
            raise TypeError("gpus must be a list of ints or null")
        base_model_sampling = row["base_model_sampling"]
        if not isinstance(base_model_sampling, bool):
            raise TypeError("base_model_sampling must be a bool")
        registry.register(name=row["name"], backend=row["backend"], gpus=gpus,
                          socket=row["socket"], pid=int(row["pid"]), base_model_sampling=base_model_sampling)
    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _error(400, f"bad registration: {type(exc).__name__}: {exc}", path)
    except RegistryConflict as exc:
        return _error(409, str(exc), path)
    return _json_reply(200, {"state": "UP"})


def _sampler_created(gateway: Gateway, engine: str, body: bytes, path: str) -> Reply:
    """An engine minted a sampler for one of its models (save_weights_for_sampler); the gateway
    records it under the model's session so sampling requests can name it."""
    if gateway.registry.get(engine) is None:
        return _error(404, f"engine {engine!r} is not registered", path)
    try:
        doc = orjson.loads(body)
        sampler_id, model_id, pinned = doc["sampler_id"], doc["model_id"], doc["pinned_version"]
        model_path, base_model = doc["model_path"], doc["base_model"]
        if not isinstance(sampler_id, str) or not isinstance(model_id, str) or not isinstance(base_model, str):
            raise TypeError("sampler_id, model_id and base_model must be strings")   # a sampler without a base model is no sampler
        if pinned is not None and not isinstance(pinned, int):
            raise TypeError("pinned_version must be an int or null")
        if model_path is not None and not isinstance(model_path, str):
            raise TypeError("model_path must be a str or null")
    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _error(400, f"bad sampler: {type(exc).__name__}: {exc}", path)
    session_id = gateway.store.session_of_model(model_id)
    if session_id is None:
        return _error(404, f"model {model_id!r} is not recorded under any session", path)
    sampler = gateway.store.add_sampler(sampler_id, session_id, model_id, base_model, model_path, pinned)
    gateway.samplers.put(sampler)
    return _json_reply(200, {"session_id": session_id})


# ------------------------------------------------------------ one app, two listeners

@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    gateway: Gateway = app.state.gateway
    await gateway.start()
    try:
        yield
    finally:
        await gateway.stop()
        gateway.store.close()
        gateway.registry.close()


def is_unix_listener(scope) -> bool:
    """The ASGI scope's `server` for a unix-socket listener: None (ASGI spec, uvicorn ≤ 0.38) or
    `(path, None)` with the socket's absolute path (uvicorn ≥ 0.53, `protocols/utils.py get_local_addr`;
    the gateway binds absolute paths). A TCP listener carries a host name, with or without a port."""
    server = scope["server"]
    if server is None:
        return True
    host, port = server
    return port is None and isinstance(host, str) and host.startswith("/")


class ListenerSplit:
    """The outermost ASGI layer: a request on the unix listener is the engines'; everything else is the SDK's."""

    def __init__(self, app, engines) -> None:
        self._app = app
        self._engines = engines

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and is_unix_listener(scope):
            await self._engines(scope, receive, send)
        else:
            await self._app(scope, receive, send)


def build_app(config: TrainingConfig, registry: Optional[Registry] = None, store: Optional[SessionStore] = None) -> FastAPI:
    """The gateway over the given registry and store, or over the shared files under METADATA_DIR.
    A store not given lives where the registry lives: in memory beside an in-memory registry, else
    `sessions.db` beside the registry's file (the two files are siblings by design).
    `app.state.gateway` is the `Gateway`; `app.state.engines` the engines face."""
    if registry is None:
        config.storage.metadata_dir.mkdir(parents=True, exist_ok=True)
        registry = Registry(config.storage.metadata_dir / "engines.db")
    if store is None:   # the store lives where the registry lives (docstring)
        store = SessionStore(None if registry.path is None else registry.path.parent / "sessions.db")
    gateway = Gateway(config, registry, store)
    return sdk_face(gateway, gateway_engines_face(registry, gateway))
