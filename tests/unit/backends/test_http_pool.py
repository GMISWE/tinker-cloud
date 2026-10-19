"""backends.http_pool: one persistent client per URL, reused across requests;
backends.miles.sglang_client: the /generate codec on top of it."""
import asyncio
import json

import httpx

from tinkercloud.training.backends.http_pool import HttpClientPool
from tinkercloud.training.backends.miles.sglang_client import SGLangClient


def _sglang_transport(hits):
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.host)
        assert request.url.path == "/generate"
        return httpx.Response(200, json={
            "text": "x",
            "meta_info": {
                "output_token_logprobs": [[-0.5, 7], [-0.25, 8]],
                "finish_reason": {"type": "stop"},
            },
        })
    return httpx.MockTransport(handler)


def test_pool_returns_one_client_per_url():
    pool = HttpClientPool(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    a1 = pool.for_url("http://a:1")
    a2 = pool.for_url("http://a:1/")
    b = pool.for_url("http://b:2")
    assert a1 is a2 and a1 is not b
    assert len(pool) == 2
    asyncio.run(pool.aclose())
    assert len(pool) == 0 and a1.is_closed and b.is_closed


def test_pool_applies_limits_and_timeout_to_its_clients():
    pool = HttpClientPool(max_connections=3, keepalive_expiry=2.0,
                          timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None),
                          transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    c = pool.for_url("http://a:1")
    assert pool._limits == httpx.Limits(max_connections=3, max_keepalive_connections=64, keepalive_expiry=2.0)
    assert c.timeout.connect == 5.0 and c.timeout.read is None
    asyncio.run(pool.aclose())


def test_generate_reuses_the_pooled_connection():
    hits = []
    pool = HttpClientPool(transport=_sglang_transport(hits))
    client = SGLangClient("http://router:30000", pool.for_url("http://router:30000"))
    params = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2}

    async def main():
        r1 = await client.generate([1, 2, 3], params)
        r2 = await client.generate([1, 2, 3], params)
        return r1, r2

    r1, r2 = asyncio.run(main())
    assert r1["tokens"] == [7, 8] and r1["logprobs"] == [-0.5, -0.25] and r1["stop_reason"] == "stop"
    assert r2 == r1
    assert hits == ["router", "router"]
    assert client._http.is_closed is False  # persistent between calls, not per-request
    asyncio.run(pool.aclose())
    assert client._http.is_closed


def test_client_forwards_every_sampling_param():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json={"meta_info": {"output_token_logprobs": [[-1.0, 3]], "finish_reason": {"type": "length"}}})

    pool = HttpClientPool(transport=httpx.MockTransport(handler))
    c = SGLangClient("http://r:1", pool.for_url("http://r:1"))
    params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 5, "top_k": 40,
              "stop": ["\n"], "stop_token_ids": [2], "seed": 11}
    out = asyncio.run(c.generate([5], params, lora_path="slot_3"))
    assert seen["sampling_params"] == {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 5,
                                       "top_k": 40, "stop": ["\n"], "stop_token_ids": [2], "sampling_seed": 11}
    assert seen["lora_path"] == "slot_3" and seen["return_logprob"] is True
    assert out["stop_reason"] == "length"
    asyncio.run(pool.aclose())
