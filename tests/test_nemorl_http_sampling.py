"""NeMo RL sampling over the workers' HTTP servers (specs/019): one
/inference/v1/generate request per sample, rows in choice order, the cache
salt keyed on the served weight version, round-robin over DP leaders, and
engine faults surfacing as BackendError naming the worker. Top-k: the request
carries topk_sample_logprobs / topk_prompt_logprobs beside vLLM's logprob
budget, and the route's top_logprobs / prompt_top_logprobs come back as
(token_id, logprob) rows best first."""
import asyncio
import json
import math

import httpx
import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.http_pool import HttpClientPool
from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle, _worker_server_roots
from tinkercloud.training.backends.nemo_rl.generation import sample_over_http
from tinkercloud.training.backends.nemo_rl.vllm_client import VllmGenerateClient
from tinkercloud.training.core import routing


class FakeTokenizer:
    def decode(self, toks, skip_special_tokens=False):
        return "".join(chr(97 + t % 26) for t in toks)


def vllm_transport(seen, *, finish="length", choices=None, prompt_logprobs=None, status=200):
    """A worker's /tinkercloud/v1/generate: records each request; answers n choices in reverse index order."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append((request.url.host, request.url.path, body))
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "boom"}})
        sp = body["sampling_params"]
        out = choices if choices is not None else [
            {"index": i, "finish_reason": finish, "token_ids": [10 + i, 11],
             "logprobs": {"content": [{"logprob": -0.5 - i}, {"logprob": -9999.0}]}}   # -9999 = floored -inf
            for i in reversed(range(sp["n"]))
        ]
        pl = prompt_logprobs
        if pl is None and sp.get("prompt_logprobs"):
            pl = [None] + [-1.0 - i for i in range(len(body["token_ids"]) - 1)]
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
    assert path == "/tinkercloud/v1/generate" and body["token_ids"] == [5, 6, 7]
    assert body["cache_salt"] == "m@3"
    assert body["sampling_params"] == {
        "n": 3, "max_tokens": 8, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 11,
        "stop": ["\n"], "stop_token_ids": [2], "logprobs": 0,
        "include_stop_str_in_output": True, "detokenize": True,
    }
    assert [s["tokens"] for s in res["sequences"]] == [[10, 11], [11, 11], [12, 11]]   # choice order restored
    assert res["sequences"][1]["logprobs"] == [-1.5, -math.inf]                         # -9999 -> -inf
    assert res["sequences"][0]["text"] == "kl" and res["sequences"][0]["stop_reason"] == "length"
    assert res["prompt_logprobs"] is None


def test_sampling_without_stop_strings_does_not_detokenize(leaders):
    """vLLM only needs text to match stop strings; TinkerCloud decodes text itself, and a
    tokenizer on the request makes vLLM decode a string per logprob entry per step."""
    seen = []
    res = run(HttpClientPool(transport=vllm_transport(seen)), handle(),
              sampling_params={"temperature": 1.0, "top_p": 0.9, "max_tokens": 8, "stop": [2, 3],
                               "stop_token_ids": [2, 3]})
    sp = seen[0][2]["sampling_params"]
    assert sp["detokenize"] is False and "stop" not in sp and "include_stop_str_in_output" not in sp
    assert sp["stop_token_ids"] == [2, 3]
    assert res["sequences"][0]["text"] == "kl"          # text still decoded, by TinkerCloud


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
              "logprobs": {"content": [{"logprob": -0.1}]}}]
    with pytest.raises(BackendError, match="2 tokens but 1 logprobs"):
        run(HttpClientPool(transport=vllm_transport([], choices=short)), handle())
    with pytest.raises(BackendError, match="returned 1 of 2 sequences"):
        run(HttpClientPool(transport=vllm_transport([], choices=short)), handle(), num_samples=2)
    with pytest.raises(BackendError, match="prompt logprobs for 2 of 3 prompt tokens"):
        run(HttpClientPool(transport=vllm_transport([], prompt_logprobs=[None, -1.0])), handle(), prompt_logprobs=True)


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


def test_stamp_is_the_submission_version_and_the_span_is_not_certified_on_the_split(monkeypatch):
    """D16: a refit landing mid-sequence leaves weight_version = the version
    at submission (a lower bound) and latest_weight_version = the version at
    completion; colocated, the ver(S) certificate still raises on the gap."""
    from tinkercloud.training.backends.nemo_rl import backend as nb, generation as gen

    async def refit_mid_serve(h, pool, request_id, prompt_tokens, num_samples, sp, pl, **topk):
        h.generation_synced_version = h.weight_version = h.weight_version + 1
        return {"samples": []}

    monkeypatch.setattr(gen, "sample_over_http", refit_mid_serve)
    backend = nb.NemoRLBackend()
    args = dict(request_id="r", prompt_tokens=[1], num_samples=1)

    h = NemoRLHandle(model_id="m", backend_type="nemo_rl", policy_generation=object(),
                     colocated_inference=False, weight_version=3, generation_synced_version=3)
    res = asyncio.run(backend.sample(h, **args))
    assert (res["weight_version"], res["latest_weight_version"]) == (3, 4)

    h = NemoRLHandle(model_id="m", backend_type="nemo_rl", policy_generation=object(),
                     colocated_inference=True, weight_version=3, generation_synced_version=3)
    with pytest.raises(BackendError, match="certificate violation"):
        asyncio.run(backend.sample(h, **args))


def control_transport(seen, *, ok=True):
    """A worker's refit routes: records (host, route), answers {"ok": ok}."""
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.url.path))
        return httpx.Response(200, json={"ok": ok})
    return httpx.MockTransport(handler)


def test_split_refit_broadcasts_while_every_leader_receives_over_http(leaders, monkeypatch):
    """D16: on the split the engine side of the NCCL refit is driven through the
    workers' app, one POST per leader concurrent with the trainer's broadcast,
    then a prefix-cache reset per leader; a leader that fails to load is an error."""
    import ray
    from tinkercloud.training.backends.nemo_rl import backend as nb

    got = {}
    monkeypatch.setattr(ray, "get", lambda refs: got.setdefault("waited", refs))

    class Policy:
        def broadcast_weights_for_collective(self):
            return ["ref-0", "ref-1"]

    seen = []
    h = NemoRLHandle(model_id="m", backend_type="nemo_rl", policy=Policy(), policy_generation=object(),
                     colocated_inference=False, http_pool=HttpClientPool(transport=control_transport(seen)))
    asyncio.run(nb._refit(h))
    assert got["waited"] == ["ref-0", "ref-1"]
    assert sorted(seen) == [
        ("n1", "/tinkercloud/v1/reset_prefix_cache"), ("n1", "/tinkercloud/v1/update_weights_from_collective"),
        ("n2", "/tinkercloud/v1/reset_prefix_cache"), ("n2", "/tinkercloud/v1/update_weights_from_collective"),
    ]
    # update before reset on every leader
    assert [r for _, r in seen if "update" in r] == ["/tinkercloud/v1/update_weights_from_collective"] * 2
    assert seen.index(("n1", "/tinkercloud/v1/reset_prefix_cache")) > seen.index(("n1", "/tinkercloud/v1/update_weights_from_collective"))

    h.http_pool = HttpClientPool(transport=control_transport([], ok=False))
    with pytest.raises(RuntimeError, match="failed to load"):
        asyncio.run(nb._refit(h))


# ---------------------------------------------------------------- top-k logprobs

def _topk_body(k_sample, k_prompt, prompt_len, n=1):
    """A /tinkercloud/v1/generate answer at top-k: top_logprobs per sampled entry
    and prompt_top_logprobs per prompt position, best first."""
    def top(k, base):
        return [{"token": 100 + base + j, "logprob": -0.1 * (j + 1) - base} for j in range(k)]
    choices = [{"index": i, "finish_reason": "length", "token_ids": [10 + i, 11],
                "logprobs": {"content": [{"logprob": -0.5, "top_logprobs": top(k_sample, 0)},
                                         {"logprob": -9999.0, "top_logprobs": top(k_sample, 1)}]}}
               for i in range(n)]
    return {"request_id": "r", "choices": choices,
            "prompt_logprobs": [None] + [-1.0 - i for i in range(prompt_len - 1)],
            "prompt_top_logprobs": [None] + [top(k_prompt, i + 1) for i in range(prompt_len - 1)]}


def test_topk_request_carries_the_counts_and_the_vllm_logprob_budget():
    client = VllmGenerateClient("http://n1:8001", None)
    body = client.build_request([5, 6, 7], {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2, "stop": ["x"]},
                                2, False, "m@3", topk_sample_logprobs=2, topk_prompt_logprobs=3)
    assert body["topk_sample_logprobs"] == 2 and body["topk_prompt_logprobs"] == 3
    sp = body["sampling_params"]
    assert sp["logprobs"] == 2 and sp["prompt_logprobs"] == 3
    assert "stop" not in sp and sp["detokenize"] is False       # prompt scoring: no stop strings
    flat = client.build_request([5, 6, 7], {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2}, 1, True, "m@3")
    assert flat["sampling_params"]["logprobs"] == 0 and flat["sampling_params"]["prompt_logprobs"] == 1
    assert flat["topk_sample_logprobs"] == 0 and flat["topk_prompt_logprobs"] == 0


def test_topk_response_parses_rows_best_first_with_the_floor_unclamped():
    client = VllmGenerateClient("http://n1:8001", None)
    body = _topk_body(2, 3, prompt_len=3)
    body["choices"][0]["logprobs"]["content"][1]["top_logprobs"][1]["logprob"] = -9999.0
    out = client.parse_response(body, [5, 6, 7], 1, True, topk_sample_logprobs=2, topk_prompt_logprobs=3)
    row = out["rows"][0]
    assert row["logprobs"] == [-0.5, -math.inf]
    assert row["topk_logprobs"] == [[(100, -0.1), (101, -0.2)], [(101, -1.1), (102, -math.inf)]]
    assert out["prompt_logprobs"] == [None, -1.0, -2.0]
    assert out["topk_prompt_logprobs"] == [None, [(101, -1.1), (102, -1.2), (103, -1.3)],
                                           [(102, -2.1), (103, -2.2), (104, -2.3)]]
    plain = client.parse_response(_topk_body(0, 0, prompt_len=3), [5, 6, 7], 1, True)
    assert "topk_logprobs" not in plain["rows"][0] and plain["topk_prompt_logprobs"] is None


def test_topk_from_a_route_that_predates_it_names_the_worker():
    client = VllmGenerateClient("http://n1:8001", None)
    old = {"request_id": "r", "prompt_logprobs": [None, -1.0, -2.0],
           "choices": [{"index": 0, "finish_reason": "length", "token_ids": [10, 11],
                        "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.6}]}}]}
    with pytest.raises(BackendError, match="n1:8001 returned no top_logprobs for topk_sample_logprobs=2"):
        client.parse_response(old, [5, 6, 7], 1, False, topk_sample_logprobs=2)
    with pytest.raises(BackendError, match="n1:8001 returned prompt top-k for None of 3 prompt tokens"):
        client.parse_response(old, [5, 6, 7], 1, True, topk_prompt_logprobs=2)
    short = {**old, "prompt_top_logprobs": [None, [{"token": 1, "logprob": -1.0}]]}
    with pytest.raises(BackendError, match="prompt top-k for 2 of 3 prompt tokens"):
        client.parse_response(short, [5, 6, 7], 1, True, topk_prompt_logprobs=1)


def test_topk_reaches_the_result_dict_over_http(leaders):
    seen = []
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        seen.append(body)
        return httpx.Response(200, json=_topk_body(body["topk_sample_logprobs"], body["topk_prompt_logprobs"],
                                                   len(body["token_ids"]), n=body["sampling_params"]["n"]))
    res = run(HttpClientPool(transport=httpx.MockTransport(handler)), handle(), num_samples=2, prompt_logprobs=True,
              topk_sample_logprobs=2, topk_prompt_logprobs=1)
    assert seen[0]["sampling_params"]["logprobs"] == 2 and seen[0]["sampling_params"]["prompt_logprobs"] == 1
    assert [s["topk_sample_logprobs"] for s in res["sequences"]] == [[[(100, -0.1), (101, -0.2)], [(101, -1.1), (102, -1.2)]]] * 2
    assert res["topk_prompt_logprobs"] == [None, [(101, -1.1)], [(102, -2.1)]]
    assert res["prompt_logprobs"] == [None, -1.0, -2.0]
    plain = run(HttpClientPool(transport=httpx.MockTransport(handler)), handle())
    assert "topk_sample_logprobs" not in plain["sequences"][0] and "topk_prompt_logprobs" not in plain
