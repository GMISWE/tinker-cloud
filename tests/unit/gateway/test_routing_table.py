"""The gateway's routing table is pinned to the engine's routes and request models.

`routing.BODY_ID_OPS` and `routing.ID_FIELDS` are hand-written so the gateway stays light; this
test derives what they must contain from the FastAPI route table and the pydantic models behind
it, so a new request model with an id or a tinker:// field cannot slip past the gateway.
"""
import re

from fastapi.routing import APIRoute
from pydantic import BaseModel

from training.api import create_app
from training.gateway import routing

# these read the raw body themselves (JSON or proto) and carry model_id in it
RAW_BODY_OPS = {"/api/v1/forward", "/api/v1/forward_backward"}
ID_NAMES = {"model_id", "request_id"}


def _walk(routes):
    """FastAPI >= 0.141 keeps an included router as one `_IncludedRouter` entry whose `original_router` holds the routes."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        else:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                yield from _walk(inner.routes)


def _routes():
    found = [r for r in _walk(create_app().routes) if r.path.startswith("/api/v1/")]
    assert len(found) >= 30, len(found)   # the route table was read; an empty walk would pass every check below
    return found


def _body_models(route):
    """Every pydantic model among the route's body parameters; a body parameter of another shape fails here."""
    models = []
    for param in route.dependant.body_params:
        annotation = param.field_info.annotation   # FastAPI's pydantic-v2 ModelField: the declared body type
        assert isinstance(annotation, type) and issubclass(annotation, BaseModel), (route.path, param.name, annotation)
        models.append(annotation)
    return models


def _routing_fields(model) -> set:
    """Fields that carry an engine-routable id: the id names, and anything described as a tinker path."""
    found = set()
    for name, info in model.model_fields.items():
        desc = (info.description or "").lower()
        if name in ID_NAMES or name == "path" or name.endswith("_path") or "tinker" in desc:
            found.add(name)
    return found


def test_every_post_route_with_an_id_field_is_routed_by_body():
    seen_body_ops = set()
    for route in _routes():
        if "POST" not in route.methods:
            continue
        if route.path in RAW_BODY_OPS:
            assert route.path in routing.BODY_ID_OPS, route.path
            seen_body_ops.add(route.path)
            continue
        models = _body_models(route)
        if not models:
            continue   # a POST that reads no body model (raw Request handlers are listed above)
        fields = set().union(*(_routing_fields(m) for m in models))
        names = [m.__name__ for m in models]
        if fields:
            assert route.path in routing.BODY_ID_OPS, (route.path, names, fields)
            missing = fields - set(routing.ID_FIELDS)
            assert not missing, (route.path, names, missing)
            seen_body_ops.add(route.path)
        else:
            assert route.path not in routing.BODY_ID_OPS, (route.path, names)
    stale = set(routing.BODY_ID_OPS) - seen_body_ops
    assert not stale, stale   # a body-routed op the engine no longer serves


def test_every_path_with_an_id_segment_is_routed_by_path():
    for route in _routes():
        for marker in ("{model_id}", "{request_id}"):
            if marker in route.path:
                sample = route.path.replace(marker, "model_fake_0123456789abcdef").replace("{kind}", "weights")
                sample = re.sub(r"\{[a-z_]+(:path)?\}", "x", sample)
                assert routing.PATH_ID.match(sample), (route.path, sample)
                assert routing.named_engine(sample, b"", b"", b"", None) == ("model_fake_0123456789abcdef", "fake"), route.path
