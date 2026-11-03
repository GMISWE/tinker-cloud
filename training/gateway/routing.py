"""Which engine a request names, read from the id in its path, query or body."""
import re
from typing import Any, Optional, Tuple
from urllib.parse import parse_qs, unquote

import orjson
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError

from ..utils.helpers import engine_name_of

# the ops whose JSON body carries model_id, request_id, or a tinker:// path that starts with a model id
# (proto bodies carry model_id in field 1)
BODY_ID_OPS = frozenset({
    "/api/v1/forward", "/api/v1/forward_backward", "/api/v1/optim_step",
    "/api/v1/save_weights", "/api/v1/save_weights_for_sampler", "/api/v1/save_weights_external",
    "/api/v1/load_weights", "/api/v1/delete_model", "/api/v1/unload_model", "/api/v1/get_info",
    "/api/v1/weights_info", "/api/v1/retrieve_future", "/api/v1/cancel_future",
    "/api/v1/create_sampling_client", "/api/v1/create_sampling_session", "/api/v1/asample", "/api/v1/sample",
})
_TINKER_URI = re.compile(r"^tinker://([^/]+)/")
# the request fields that hold an id or a tinker:// URI; tests/unit/gateway/test_routing_table.py pins
# this and BODY_ID_OPS to the request models behind the engine's routes
ID_FIELDS = ("model_id", "request_id", "model_path", "path", "tinker_path")
PATH_ID = re.compile(r"^/api/v1/(?:retrieve_future|training_runs|external_weights)/([^/?]+)")
# spelled as training/proto/wire.py spells it; that module needs numpy and the gateway must not
PROTO_CONTENT_TYPE = b"application/x-protobuf"


def _skeleton_class() -> Any:
    """A message type declaring only field 1 (model_id) of the forward/forward_backward request: parsing it skips the token arrays."""
    fdp = descriptor_pb2.FileDescriptorProto(name="route.proto", package="route", syntax="proto3")
    msg = fdp.message_type.add(name="ModelRoute")
    msg.field.add(name="model_id", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
                  label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("route.ModelRoute"))


_ModelRoute = _skeleton_class()


def reads_body(path: str) -> bool:
    """True when routing needs the whole body (the id is inside it)."""
    return path in BODY_ID_OPS


def _named(id_: Any) -> Optional[Tuple[str, str]]:
    """(id, engine) when `id_` is an id that names an engine; a tinker:// path names the model it starts with."""
    if not isinstance(id_, str):
        return None
    m = _TINKER_URI.match(id_)
    if m:
        id_ = m.group(1)
    name = engine_name_of(id_)
    return (id_, name) if name else None


def named_engine(path: str, query: bytes, content_type: bytes, content_encoding: bytes,
                 body: Optional[bytes]) -> Optional[Tuple[str, str]]:
    """(id, engine name) for the id the request carries, or None when it names no engine
    (sessions, create_model, health, ids in another shape)."""
    m = PATH_ID.match(path)
    if m:
        return _named(unquote(m.group(1)))
    if path == "/api/v1/get_tokenizer":
        ids = parse_qs(query.decode()).get("model_id")
        return _named(ids[0]) if ids else None
    if body is None or path not in BODY_ID_OPS or content_encoding:
        return None   # a compressed body is not opened here; the engine decompresses it
    if content_type.startswith(PROTO_CONTENT_TYPE):
        route = _ModelRoute()
        try:
            route.ParseFromString(body)
        except DecodeError:
            return None   # the engine rejects it with 422
        return _named(route.model_id)
    try:
        doc = orjson.loads(body)
    except orjson.JSONDecodeError:
        return None   # the engine rejects it with 422
    if not isinstance(doc, dict):
        return None
    for field in ID_FIELDS:
        named = _named(doc.get(field))
        if named:
            return named
    return None
