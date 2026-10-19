"""SDK 0.33 sampling fields at the HTTP boundary and in the service: the request
model carries them, the router refuses the unsupported ones by name (400), the
service caps top-k at the engine's limit and masks prompt scores before a
prompt_logprobs_last_n suffix, and the fake backend's top-k is deterministic."""
import asyncio

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tinkercloud.training.backends.base import SampleRequestError
from tinkercloud.training.backends.fake.backend import FAKE_MAX_TOPK, FakeBackend, logprobs_for, topk_for
from tinkercloud.training.models.requests import ASampleRequest
from tinkercloud.training.models.responses import validate_result
from tinkercloud.training.routers.sampling import check_sample_extensions
from tinkercloud.training.services.sampling_service import _check_sample_request, _mask_prompt_prefix

PROMPT = [1, 2, 3, 4, 5]


def request(**fields):
    return ASampleRequest.model_validate(
        {"prompt": {"tokens": PROMPT}, "sampling_params": {"max_tokens": 4}, **fields})


def test_new_fields_default_off_and_are_kept():
    r = request()
    assert (r.topk_sample_logprobs, r.topk_prompt_logprobs, r.prompt_alt_tokens_k) == (0, 0, 0)
    assert r.prompt_logprobs_last_n is None and r.target_prompt_logprobs is None
    r = request(prompt_logprobs=True, topk_prompt_logprobs=2, topk_sample_logprobs=3, prompt_logprobs_last_n=2,
                target_prompt_logprobs={"data": [7, -1], "dtype": "int64", "shape": [1, 2]}, prompt_alt_tokens_k=4)
    assert (r.topk_sample_logprobs, r.topk_prompt_logprobs, r.prompt_logprobs_last_n) == (3, 2, 2)
    assert r.target_prompt_logprobs.data == [7, -1] and r.prompt_alt_tokens_k == 4


@pytest.mark.parametrize("field", ["topk_sample_logprobs", "topk_prompt_logprobs", "prompt_alt_tokens_k"])
def test_negative_counts_are_rejected_by_the_model(field):
    with pytest.raises(ValidationError):
        request(**{field: -1})


@pytest.mark.parametrize("fields, named", [
    ({"target_prompt_logprobs": {"data": [7], "dtype": "int64", "shape": [1, 1]}}, "target_prompt_logprobs is unsupported"),
    ({"prompt_alt_tokens_k": 1}, "prompt_alt_tokens_k is unsupported"),
    ({"prompt_logprobs_last_n": 2}, "prompt_logprobs_last_n requires prompt_logprobs"),
    ({"prompt_logprobs": True, "prompt_logprobs_last_n": 0}, "prompt_logprobs_last_n must be between 1 and len(prompt) - 1 = 4, got 0"),
    ({"prompt_logprobs": True, "prompt_logprobs_last_n": 5}, "prompt_logprobs_last_n must be between 1 and len(prompt) - 1 = 4, got 5"),
])
def test_router_refuses_by_field_name_with_400(fields, named):
    with pytest.raises(HTTPException) as ei:
        check_sample_extensions(request(**fields), len(PROMPT))
    assert ei.value.status_code == 400 and named in ei.value.detail


def test_router_accepts_the_supported_combination():
    check_sample_extensions(request(prompt_logprobs=True, prompt_logprobs_last_n=4, topk_prompt_logprobs=2,
                                    topk_sample_logprobs=2), len(PROMPT))
    check_sample_extensions(request(prompt_logprobs=True, prompt_logprobs_last_n=1), len(PROMPT))


def test_mask_prompt_prefix_leaves_only_the_last_n_scored():
    result = {"prompt_logprobs": [None, -1.0, -2.0, -3.0, -4.0],
              "topk_prompt_logprobs": [None, [(1, -1.0)], [(2, -2.0)], [(3, -3.0)], [(4, -4.0)]]}
    _mask_prompt_prefix(result, 5, 2)
    assert result["prompt_logprobs"] == [None, None, None, -3.0, -4.0]
    assert result["topk_prompt_logprobs"] == [None, None, None, [(3, -3.0)], [(4, -4.0)]]
    untouched = {"prompt_logprobs": [None, -1.0]}
    _mask_prompt_prefix(untouched, 2, None)
    assert untouched == {"prompt_logprobs": [None, -1.0]}
    absent = {"prompt_logprobs": None, "sequences": []}
    _mask_prompt_prefix(absent, 5, 2)
    assert absent["prompt_logprobs"] is None and "topk_prompt_logprobs" not in absent


def _fake_model():
    b = FakeBackend()
    return b, asyncio.run(b.create_model("m", "r", "fake/tiny", 0))


def test_service_caps_topk_at_the_engine_limit():
    _, h = _fake_model()
    assert h.max_topk_logprobs == FAKE_MAX_TOPK
    _check_sample_request(h, [1, 2], {"max_tokens": 2}, FAKE_MAX_TOPK, FAKE_MAX_TOPK)
    with pytest.raises(SampleRequestError, match=f"topk_sample_logprobs {FAKE_MAX_TOPK + 1} exceeds the engine cap {FAKE_MAX_TOPK}"):
        _check_sample_request(h, [1, 2], {"max_tokens": 2}, FAKE_MAX_TOPK + 1, 0)
    with pytest.raises(SampleRequestError, match="topk_prompt_logprobs 21 exceeds"):
        _check_sample_request(h, [1, 2], {"max_tokens": 2}, 0, 21)
    h.max_topk_logprobs = None   # unknown cap: not checked
    _check_sample_request(h, [1, 2], {"max_tokens": 2}, 1000, 1000)


def test_fake_topk_is_deterministic_and_validates():
    b, h = _fake_model()
    kw = dict(sampling_params={"max_tokens": 3, "seed": 1}, prompt_logprobs=True,
              topk_sample_logprobs=2, topk_prompt_logprobs=2)
    a = asyncio.run(b.sample(h, "r", PROMPT, 2, **kw))
    assert a == asyncio.run(b.sample(h, "r", PROMPT, 2, **kw))
    for seq in a["sequences"]:
        rows = seq["topk_sample_logprobs"]
        assert len(rows) == len(seq["tokens"])
        # rank 0 is the sampled token at its own logprob; ranks are best first
        assert [row[0] for row in rows] == list(zip(seq["tokens"], seq["logprobs"]))
        assert all(row[0][1] > row[1][1] for row in rows)
    assert a["topk_prompt_logprobs"][0] is None
    assert a["topk_prompt_logprobs"][1:] == topk_for(PROMPT, 2)[1:]
    assert [row[0][1] for row in a["topk_prompt_logprobs"][1:]] == logprobs_for(PROMPT)[1:]
    plain = asyncio.run(b.sample(h, "r", PROMPT, 1, sampling_params={"max_tokens": 3, "seed": 1}))
    assert "topk_sample_logprobs" not in plain["sequences"][0] and "topk_prompt_logprobs" not in plain
    out = validate_result("asample", a)
    assert out["sequences"][0]["topk_sample_logprobs"] == a["sequences"][0]["topk_sample_logprobs"]
    assert out["topk_prompt_logprobs"] == a["topk_prompt_logprobs"]
