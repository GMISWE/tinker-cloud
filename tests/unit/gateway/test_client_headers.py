"""The aiohttp client adds nothing of its own: the engine sees the SDK's headers and bytes, and the response bytes come back as written."""
from .support import API_KEY, Harness, run, serve_uds


def test_the_aiohttp_client_adds_no_headers_of_its_own(tmp_path):
    """The engine sees what the SDK sent plus the hop's own framing: nothing of the client's
    (its User-Agent, Accept, Accept-Encoding), and a response body as the engine wrote it."""
    async def main():
        h = Harness(str(tmp_path), "aiohttp")
        try:
            async with serve_uds(h.engine, h.engine_socket):
                await h.register()
                sent = {"host": "sdk:8000", "x-api-key": API_KEY, "content-type": "application/json",
                        "accept": "*/*", "accept-encoding": "identity", "user-agent": "sdk-test/1"}
                r = await h.sdk.post("/api/v1/echo", content=b'{"a": 1}', headers=sent)
                assert r.status_code == 200 and r.content == b'{"a": 1}'
                seen = {k.lower(): v for k, v in h.engine.requests[-1]["headers"].items()}
                extra = set(seen) - set(sent) - {"content-length", "transfer-encoding", "connection"}   # the hop's own framing
                assert extra == set(), {k: seen[k] for k in extra}
                assert all(seen[k] == v for k, v in sent.items()), seen
                r = await h.sdk.post("/api/v1/echo", content=b"\x1f\x8bnot-really-gzip",
                                     headers={"X-API-Key": API_KEY, "content-type": "application/octet-stream", "content-encoding": "gzip"})
                assert r.status_code == 200 and r.content == b"\x1f\x8bnot-really-gzip"   # bytes through, not inflated
        finally:
            await h.close()
    run(main())
