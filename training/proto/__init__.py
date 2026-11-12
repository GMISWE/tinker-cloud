"""Tinker public wire schema.

`tinker_public_pb2` is the generated module shipped in the tinker SDK
(0.27.0, `src/tinker/proto/tinker_public_pb2.py`), vendored so the server
does not depend on which SDK version is installed beside it. Regenerate by
copying the SDK's file; never hand-edit it. The conversions between that
schema and the server's request and result dicts live in `request_conv.py` and
`response_conv.py`, each the inverse of the SDK's file of the same name.
"""
PROTO_CONTENT_TYPE = "application/x-protobuf"
