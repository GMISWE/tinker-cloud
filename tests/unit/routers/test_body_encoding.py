"""`Content-Encoding: zstd` request bodies (training/routers/body_encoding.py)."""
import pytest

from tinkercloud.training.proto.request_conv import WireError
from tinkercloud.training.routers import body_encoding
from tinkercloud.training.routers.body_encoding import decompress_zstd


def test_a_zstd_body_past_the_bound_is_refused_and_an_unsized_frame_still_reads(monkeypatch):
    zstandard = pytest.importorskip("zstandard")
    monkeypatch.setattr(body_encoding, "MAX_DECOMPRESSED_BODY_BYTES", 1 << 20)
    small = b"x" * 1000
    assert decompress_zstd(zstandard.ZstdCompressor(write_content_size=False).compress(small)) == small
    for compressor in (zstandard.ZstdCompressor(), zstandard.ZstdCompressor(write_content_size=False)):
        with pytest.raises(WireError, match="decompresses past"):
            decompress_zstd(compressor.compress(b"\0" * ((1 << 20) + 1)))


def test_zstd_round_trip_and_garbage():
    zstandard = pytest.importorskip("zstandard")
    raw = b"a request body " * 64
    assert decompress_zstd(zstandard.ZstdCompressor().compress(raw)) == raw
    with pytest.raises(WireError, match="zstd"):
        decompress_zstd(b"not zstd")
