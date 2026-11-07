"""The SDK path of an on-policy distillation teacher, create_sampling_client(base_model=...), on an
engine that serves that base model (the fake configured with base_models): compute_logprobs and
sample reach the backend's base-model handle, not a model; another base model is refused by name."""
import json
import math

import pytest
import tinker
from tinker import types

from .conftest import API_KEY, Server

BASE = "fake/tiny"


@pytest.fixture(scope="module", params=["single", "split"])
def base_server(request, tmp_path_factory):
    srv = Server(tmp_path_factory.mktemp("base_sampling"), request.param,
                 TINKERCLOUD_BACKEND_OVERRIDES=json.dumps({"base_models": [BASE]}))
    try:
        srv.wait_healthy()
        yield srv
    finally:
        srv.stop()


def test_a_base_model_sampler_scores_and_samples(base_server):
    client = tinker.ServiceClient(base_url=base_server.base_url, api_key=API_KEY)
    sampler = client.create_sampling_client(base_model=BASE).result()
    prompt = types.ModelInput.from_ints([5, 6, 7, 8])
    logprobs = sampler.compute_logprobs(prompt).result()
    assert len(logprobs) == 4 and logprobs[0] is None
    assert all(isinstance(lp, float) and math.isfinite(lp) for lp in logprobs[1:])
    out = sampler.sample(prompt=prompt, num_samples=2, sampling_params=types.SamplingParams(max_tokens=3)).result()
    assert len(out.sequences) == 2
    assert all(0 < len(s.tokens) <= 3 and s.logprobs is not None and len(s.logprobs) == len(s.tokens) for s in out.sequences)
    ops = [t for t in base_server.trace() if t["op"] == "sample"]
    assert [t["model_id"] for t in ops] == [f"base:{BASE}"] * 2
    assert [t["prompt_logprobs"] for t in ops] == [True, False]


def test_a_base_model_the_engine_does_not_serve_is_refused_by_name(base_server):
    client = tinker.ServiceClient(base_url=base_server.base_url, api_key=API_KEY)
    with pytest.raises(Exception) as ei:
        client.create_sampling_client(base_model="other/model").result().compute_logprobs(
            types.ModelInput.from_ints([1, 2])).result()
    assert "other/model" in str(ei.value) and "not served" in str(ei.value)
