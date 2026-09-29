"""NeMo RL sampling over the workers' HTTP servers (specs/019): one
/inference/v1/generate request per sample, rows in choice order, the cache
salt keyed on the served weight version, round-robin over DP leaders, and
engine faults surfacing as BackendError naming the worker."""
import asyncio
import json
import math

import httpx
import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.http_pool import HttpClientPool
from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle, _worker_server_roots
from tinkercloud.training.backends.nemo_rl.generation import sample_over_http
from tinkercloud.training.core import routing


class FakeTokenizer:
    def decode(self, toks, skip_special_tokens=False):
        return "".join(chr(97 + t % 26) for t in toks)


def vllm_transport(seen, *, finish="length", choices=None, prompt_logprobs=None, status=200):
    """A vLLM worker: records each request; answers n choices in reverse index order."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append((request.url.host, request.url.path, body))
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "boom"}})
        sp = body["sampling_params"]
        out = choices if choices is not None else [
            {"index": i, "finish_reason": finish, "token_ids": [10 + i, 11],
             "logprobs": {"content": [{"token": "token_id:10", "logprob": -0.5 - i},
                                      {"token": "token_id:11", "logprob": -9999.0}]}}
            for i in reversed(range(sp["n"]))
        ]
        pl = prompt_logprobs
        if pl is None and sp.get("prompt_logprobs"):
            pl = [None] + [{str(t): {"logprob": -1.0 - i, "rank": 1, "decoded_token": "x"}}
                           for i, t in enumerate(body["token_ids"][1:])]
        return httpx.Response(200, json={"request_id": "r", "choices": out, "prompt_logprobs": pl})
    return httpx.MockTransport(handler)


def handle(version=3):
    return NemoRLHandle(model_id="m", backend_type="nemo_rl", tokenizer=FakeTokenizer(),
                        generation_synced_version=version)


@pytest.fixture
def leaders():
    routing.table.publish("m", routing.InferenceEndpoint(("http://n1:8001", "http://n2:8001")))
    yield
    routing.table.withdraw("m")


def run(pool, h, **kw):
    args = dict(request_id="r", prompt_tokens=[5, 6, 7], num_samples=1,
                sampling_params={"temperature": 1.0, "top_p": 0.9, "max_tokens": 8}, prompt_logprobs=False)
    args.update(kw)
    return asyncio.run(sample_over_http(h, pool, **args))


def test_one_request_per_sample_with_every_parameter_and_the_version_salt(leaders):
    seen = []
    res = run(HttpClientPool(transport=vllm_transport(seen)), handle(version=3), num_samples=3,
              sampling_params={"temperature": 0.7, "top_p": 0.9, "max_tokens": 8, "top_k": 40,
                               "seed": 11, "stop": ["\n"], "stop_token_ids": [2]})
    (host, path, body), = seen
    assert path == "/inference/v1/generate" and body["token_ids"] == [5, 6, 7]
    assert body["cache_salt"] == "m@3"
    assert body["sampling_params"] == {
        "n": 3, "max_tokens": 8, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 11,
        "stop": ["\n"], "stop_token_ids": [2], "logprobs": 1,
        "include_stop_str_in_output": True, "detokenize": True, "output_kind": 2,
    }
    assert [s["tokens"] for s in res["sequences"]] == [[10, 11], [11, 11], [12, 11]]   # choice order restored
    assert res["sequences"][1]["logprobs"] == [-1.5, -math.inf]                         # -9999 -> -inf
    assert res["sequences"][0]["text"] == "kl" and res["sequences"][0]["stop_reason"] == "length"
    assert res["prompt_logprobs"] is None


def test_stop_reason_follows_finish_reason(leaders):
    res = run(HttpClientPool(transport=vllm_transport([], finish="stop")), handle())
    assert res["sequences"][0]["stop_reason"] == "stop"


def test_scoring_request_skips_detokenize_and_stop_strings_and_returns_prompt_logprobs(leaders):
    seen = []
    res = run(HttpClientPool(transport=vllm_transport(seen)), handle(), prompt_logprobs=True,
              sampling_params={"temperature": 1.0, "top_p": 1.0, "max_tokens": 1, "stop": ["x"]})
    sp = seen[0][2]["sampling_params"]
    assert sp["detokenize"] is False and sp["prompt_logprobs"] == 1 and "stop" not in sp
    assert res["prompt_logprobs"] == [None, -1.0, -2.0]


def test_round_robin_over_dp_leaders(leaders):
    seen = []
    pool = HttpClientPool(transport=vllm_transport(seen))
    h = handle()
    for _ in range(3):
        run(pool, h)
    assert [host for host, _, _ in seen] == ["n1", "n2", "n1"]


def test_abort_and_http_errors_name_the_worker(leaders):
    with pytest.raises(BackendError, match="n1:8001 aborted"):
        run(HttpClientPool(transport=vllm_transport([], finish="abort")), handle())
    with pytest.raises(BackendError, match="n1:8001 returned 500"):
        run(HttpClientPool(transport=vllm_transport([], status=500)), handle())

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)
    with pytest.raises(BackendError, match="n1:8001 unreachable"):
        run(HttpClientPool(transport=httpx.MockTransport(refuse)), handle())


def test_token_logprob_mismatch_and_missing_prompt_logprob_fail(leaders):
    short = [{"index": 0, "finish_reason": "length", "token_ids": [1, 2],
              "logprobs": {"content": [{"token": "t", "logprob": -0.1}]}}]
    with pytest.raises(BackendError, match="2 tokens but 1 logprobs"):
        run(HttpClientPool(transport=vllm_transport([], choices=short)), handle())
    with pytest.raises(BackendError, match="returned 1 of 2 sequences"):
        run(HttpClientPool(transport=vllm_transport([], choices=short)), handle(), num_samples=2)
    with pytest.raises(BackendError, match="no logprob for prompt position 1"):
        run(HttpClientPool(transport=vllm_transport([], prompt_logprobs=[None, {"999": {"logprob": -1}}, {"7": {"logprob": -1}}])),
            handle(), prompt_logprobs=True)


def test_unpublished_model_is_a_backend_error():
    with pytest.raises(BackendError, match="not available"):
        run(HttpClientPool(transport=vllm_transport([])), handle())


def test_worker_server_roots_strip_the_v1_suffix():
    class Gen:
        dp_openai_server_base_urls = ["http://10.0.0.1:41000/v1", "http://10.0.0.2:41002/v1"]
    assert _worker_server_roots(Gen()) == ("http://10.0.0.1:41000", "http://10.0.0.2:41002")
    assert _worker_server_roots(None) == ()

    class NoServer:
        dp_openai_server_base_urls = [None]
    with pytest.raises(BackendError, match="no HTTP server"):
        _worker_server_roots(NoServer())
