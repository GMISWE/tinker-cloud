"""SDK 0.33 sampling fields at the HTTP boundary and in the service: the request
model carries them, the router refuses the unsupported ones by name (400), the
service caps top-k at the engine's limit and masks prompt scores before a
prompt_logprobs_last_n suffix."""
import asyncio
import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tinkercloud.training.backends.base import SampleRequestError
from tinkercloud.training.backends.fake.backend import FAKE_MAX_TOPK, FakeBackend
from tinkercloud.training.checkpoints.store import CheckpointStore
from tinkercloud.training.models.requests import ASampleRequest, SampleRequest, SamplingParams
from tinkercloud.training.routers.sampling import check_sample_extensions, sample
from tinkercloud.training.services.sampling_service import SamplingService, _check_sample_request, _mask_prompt_prefix
from tinkercloud.training.storage.metadata import MetadataStorage
from tinkercloud.training.storage.sample_futures import SampleFutureStore
from tinkercloud.training.utils import helpers

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
    return b, asyncio.run(b.create_model("m", "r", "fake/tiny", 0, optimizer="adamw"))


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


def test_sync_sample_forwards_pinned_version_to_backend(tmp_path, monkeypatch):
    """`/api/v1/sample` resolved the sampler's pin and dropped it, so the backend saw `pinned_version=None`.
    The second request carries no pin: the control."""
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("FAKE_BACKEND_TRACE", str(trace_path))
    monkeypatch.setattr(helpers, "_ENGINE_NAME", "t")

    async def _exercise():
        backend = FakeBackend()
        model_id = "model_t_0123456789abcdef"
        handle = await backend.create_model(model_id, "r1", "fake/tiny", 0, optimizer="adamw")
        clients = {model_id: {"backend_handle": handle}}
        service = SamplingService(backend)
        futures = SampleFutureStore()
        store = CheckpointStore(tmp_path / "ckpt", MetadataStorage(tmp_path / "meta"))

        params = SamplingParams(max_tokens=2)
        reqs = [
            (SampleRequest(prompts=[[1, 2]], num_samples=1, sampling_params=params, sampling_session_id="s1"),
             {"x_model_id": model_id, "x_pinned_version": "1", "x_base_model": None}),
            (SampleRequest(prompts=[[5, 6]], num_samples=1, sampling_params=params, sampling_session_id="s2"),
             {"x_model_id": model_id, "x_pinned_version": None, "x_base_model": None}),
        ]
        for req, hdrs in reqs:
            op = await sample(req, None, service, futures, clients, store, **hdrs)
            sf = futures.get(op.request_id)
            assert sf is not None
            await sf.done.wait()
            assert sf.status == "completed"

    asyncio.run(_exercise())
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [t["pinned_version"] for t in trace if t["op"] == "sample"] == [1, None]
