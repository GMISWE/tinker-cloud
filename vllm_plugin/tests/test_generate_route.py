"""The generate route over a fake engine, with vLLM's own output types.

Runs in the vLLM env, which has no pytest:
    python -m unittest discover -s vllm_plugin/tests
"""

import math
import unittest
from importlib.metadata import entry_points

from fastapi import FastAPI
from starlette.testclient import TestClient
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import Logprob
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from tinkercloud_vllm_plugin import CLAMPED_LOGPROB, TinkerCloudEndpoints

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = LoRARequest(lora_name="m1@v1", lora_int_id=1, lora_path="/data/m1/v1")
PROMPT = [11, 12, 13]


class FakeModels:
    def __init__(self):
        self.lora_requests = {ADAPTER.lora_name: ADAPTER}

    def is_base_model(self, name: str) -> bool:
        return name == BASE


class FakeEngine:
    """Answers every request with two sequences of two tokens each."""

    def __init__(self, raises: Exception | None = None):
        self.calls = []
        self.raises = raises

    async def generate(self, prompt, params, request_id, *, lora_request=None):
        self.calls.append((prompt, params, lora_request))
        if self.raises is not None:
            raise self.raises
        # top-3 per position; the sampled token is always among the keys
        def pos(sampled, lp):
            return {sampled: Logprob(lp), 901: Logprob(-0.1), 902: Logprob(-5.0), 903: Logprob(-7.0)}
        outputs = [
            CompletionOutput(index=1, text="", token_ids=[7, 8], cumulative_logprob=None,
                             logprobs=[pos(7, -1.0), pos(8, -math.inf)], finish_reason="length"),
            CompletionOutput(index=0, text="", token_ids=[5, 6], cumulative_logprob=None,
                             logprobs=[pos(5, -0.5), pos(6, -0.25)], finish_reason="stop"),
        ]
        prompt_lps = None
        if params.prompt_logprobs is not None:
            prompt_lps = [None] + [pos(t, -2.0) for t in PROMPT[1:]]
        yield RequestOutput(request_id=request_id, prompt=None, prompt_token_ids=PROMPT,
                            prompt_logprobs=prompt_lps, outputs=outputs, finished=True)


def client(engine: FakeEngine) -> TestClient:
    app = FastAPI()
    TinkerCloudEndpoints().attach_router(app)
    app.state.engine_client = engine
    app.state.openai_serving_models = FakeModels()
    return TestClient(app)


def body(**overrides):
    req = {"model": BASE, "token_ids": PROMPT, "sampling_params": {"n": 2, "max_tokens": 2, "logprobs": 0},
           "cache_salt": "m1@0"}
    req.update(overrides)
    return req


class GenerateRouteTest(unittest.TestCase):
    def test_base_model_flat_response(self):
        engine = FakeEngine()
        r = client(engine).post("/tinkercloud/v1/generate", json=body())
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        prompt, params, lora = engine.calls[0]
        self.assertIsNone(lora)
        self.assertEqual(prompt["prompt_token_ids"], PROMPT)
        self.assertEqual(prompt["cache_salt"], "m1@0")
        self.assertEqual(params.output_kind, RequestOutputKind.FINAL_ONLY)
        by_index = {c["index"]: c for c in out["choices"]}
        self.assertEqual(by_index[0]["token_ids"], [5, 6])
        self.assertEqual(by_index[0]["finish_reason"], "stop")
        self.assertEqual(by_index[0]["logprobs"], {"content": [{"logprob": -0.5}, {"logprob": -0.25}]})
        # -inf has no JSON form; it is floored
        self.assertEqual(by_index[1]["logprobs"]["content"][1], {"logprob": CLAMPED_LOGPROB})
        self.assertIsNone(out["prompt_logprobs"])
        self.assertNotIn("prompt_top_logprobs", out)

    def test_adapter_name_selects_the_adapter(self):
        engine = FakeEngine()
        r = client(engine).post("/tinkercloud/v1/generate", json=body(model="m1@v1"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIs(engine.calls[0][2], ADAPTER)

    def test_unknown_model_is_404_naming_it(self):
        engine = FakeEngine()
        r = client(engine).post("/tinkercloud/v1/generate", json=body(model="m1@v0"))
        self.assertEqual(r.status_code, 404)
        self.assertIn("'m1@v0'", r.json()["error"])
        self.assertEqual(engine.calls, [])

    def test_missing_model_is_422(self):
        req = body()
        del req["model"]
        r = client(FakeEngine()).post("/tinkercloud/v1/generate", json=req)
        self.assertEqual(r.status_code, 422)

    def test_invalid_sampling_params_is_422(self):
        r = client(FakeEngine()).post("/tinkercloud/v1/generate",
                                      json=body(sampling_params={"temperature": -1.0}))
        self.assertEqual(r.status_code, 422, r.text)

    def test_topk_above_logprobs_is_422(self):
        r = client(FakeEngine()).post("/tinkercloud/v1/generate", json=body(topk_sample_logprobs=2))
        self.assertEqual(r.status_code, 422)

    def test_topk_sample_and_prompt(self):
        sp = {"n": 2, "max_tokens": 2, "logprobs": 2, "prompt_logprobs": 2}
        r = client(FakeEngine()).post("/tinkercloud/v1/generate", json=body(
            sampling_params=sp, topk_sample_logprobs=2, topk_prompt_logprobs=2))
        self.assertEqual(r.status_code, 200, r.text)
        out = r.json()
        first = {c["index"]: c for c in out["choices"]}[0]["logprobs"]["content"][0]
        self.assertEqual(first["top_logprobs"], [{"token": 901, "logprob": -0.1}, {"token": 5, "logprob": -0.5}])
        self.assertEqual(out["prompt_logprobs"], [None, -2.0, -2.0])
        self.assertIsNone(out["prompt_top_logprobs"][0])
        self.assertEqual(out["prompt_top_logprobs"][1][0], {"token": 901, "logprob": -0.1})

    def test_engine_validation_error_is_400(self):
        engine = FakeEngine(raises=VLLMValidationError("The decoder prompt (length 9) is longer than ..."))
        r = client(engine).post("/tinkercloud/v1/generate", json=body())
        self.assertEqual(r.status_code, 400)
        self.assertIn("longer than", r.json()["error"])


class EntryPointTest(unittest.TestCase):
    def test_registered_under_vllm_endpoint_plugins(self):
        eps = {e.name: e for e in entry_points(group="vllm.endpoint_plugins")}
        self.assertIs(eps["tinkercloud"].load(), TinkerCloudEndpoints)


if __name__ == "__main__":
    unittest.main()
