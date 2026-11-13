"""`Content-Encoding: zstd` on a request body. ASGI servers do not decode request bodies."""
from ..proto.request_conv import MAX_DECOMPRESSED_BODY_BYTES, WireError

try:
    import zstandard
except ImportError:  # optional: the server then advertises proto_compress_fwdbwd=False  # noqa: TREE002  (#192)
    zstandard = None

# The body is read in pieces, so the bound reserves nothing; it only stops a body that keeps growing.
_DECOMPRESS_PIECE_BYTES = 1 << 20


def zstd_available() -> bool:
    return zstandard is not None


def decompress_zstd(body: bytes) -> bytes:
    """`Content-Encoding: zstd` body -> raw bytes. A frame that states no content size never
    reserves the whole bound."""
    if zstandard is None:
        raise WireError("zstd request bodies are not supported on this server (zstandard not installed)")
    out = bytearray()
    try:
        with zstandard.ZstdDecompressor().stream_reader(body) as reader:
            while piece := reader.read(_DECOMPRESS_PIECE_BYTES):
                out += piece
                if len(out) > MAX_DECOMPRESSED_BODY_BYTES:
                    raise WireError(f"zstd body decompresses past {MAX_DECOMPRESSED_BODY_BYTES} bytes")
    except zstandard.ZstdError as e:
        raise WireError(f"zstd decompression failed: {e}") from None
    return bytes(out)
