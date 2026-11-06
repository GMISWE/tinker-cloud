"""TinkerCloud's /tinkercloud/v1/generate route as a vLLM endpoint plugin.

The route takes a token-id prompt, a SamplingParams dict and a model name, and
answers with flat logprobs: a choice carries token_ids, finish_reason and
logprobs as {"content": [{"logprob": x}, ...]} (the sampled token's logprob,
floored to CLAMPED_LOGPROB where vLLM has no finite value); prompt_logprobs is
a flat list with None at position 0. topk_sample_logprobs=k > 0 (with
sampling_params.logprobs >= k) adds top_logprobs to each entry, the k most
likely tokens of that position best first as [{"token", "logprob"}];
topk_prompt_logprobs=k > 0 (with sampling_params.prompt_logprobs >= k) adds
prompt_top_logprobs, one such list per prompt position and None at position 0.
vLLM's own generate route builds two pydantic objects per token; this one
encodes with msgspec.

`model` selects the weights: the served base model name samples the base, the
name of an adapter loaded through /v1/load_lora_adapter samples that adapter,
and any other name is a 404. A client disconnect aborts the engine request.
The route answers from the last RequestOutput, so it forces
output_kind=FINAL_ONLY, the only kind that carries every child of an n > 1
request.

Loaded by vLLM >= 0.26 when VLLM_PLUGINS names `tinkercloud`.
"""

import math
from argparse import Namespace
from typing import Any, Optional

import msgspec
from fastapi import APIRouter, FastAPI, Request, Response
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.utils.api_utils import with_cancellation
from vllm.exceptions import VLLMValidationError
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils import random_uuid

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0


class GenerateRequest(msgspec.Struct):
    model: str
    token_ids: list[int]
    sampling_params: dict[str, Any] = {}
    cache_salt: Optional[str] = None
    # top-k entries to emit per sampled / prompt position; 0 = none
    topk_sample_logprobs: int = 0
    topk_prompt_logprobs: int = 0


class TopLogprob(msgspec.Struct):
    token: int
    logprob: float


class LogprobEntry(msgspec.Struct, omit_defaults=True):
    logprob: float
    top_logprobs: Optional[list[TopLogprob]] = None


class Logprobs(msgspec.Struct):
    content: list[LogprobEntry]


class Choice(msgspec.Struct):
    index: int
    finish_reason: Optional[str]
    token_ids: list[int]
    logprobs: Optional[Logprobs]


class GenerateResponse(msgspec.Struct, omit_defaults=True):
    request_id: str
    choices: list[Choice]
    prompt_logprobs: Optional[list[Optional[float]]]
    prompt_top_logprobs: Optional[list[Optional[list[TopLogprob]]]] = None


class ErrorBody(msgspec.Struct):
    error: str


_decoder = msgspec.json.Decoder(GenerateRequest)


def _floored(logprob: float) -> float:
    return logprob if math.isfinite(logprob) else CLAMPED_LOGPROB


def _top_k(position: dict, k: int) -> list[TopLogprob]:
    """The k most likely tokens of one position, best first. vLLM's dict holds
    the top-k plus, when outside them, the sampled (or prompt) token itself."""
    ranked = sorted(position.items(), key=lambda kv: kv[1].logprob, reverse=True)
    return [TopLogprob(token=t, logprob=_floored(lp.logprob)) for t, lp in ranked[:k]]


def _error(status: int, message: str) -> Response:
    return Response(
        msgspec.json.encode(ErrorBody(error=message)),
        status_code=status,
        media_type="application/json",
    )


def _response(request_id: str, req: GenerateRequest, final, k_sample: int, k_prompt: int) -> Response:
    choices = []
    for out in final.outputs:
        ids = list(out.token_ids)
        logprobs = None
        if out.logprobs is not None:
            # the sampled token is always present in its position's dict
            logprobs = Logprobs(content=[
                LogprobEntry(
                    logprob=_floored(out.logprobs[i][t].logprob),
                    top_logprobs=_top_k(out.logprobs[i], k_sample) if k_sample else None,
                )
                for i, t in enumerate(ids)
            ])
        choices.append(Choice(index=out.index, finish_reason=out.finish_reason, token_ids=ids, logprobs=logprobs))
    prompt_logprobs = None
    prompt_top_logprobs = None
    if final.prompt_logprobs is not None:
        prompt_logprobs = [
            None if d is None else _floored(d[t].logprob)
            for d, t in zip(final.prompt_logprobs, req.token_ids)
        ]
        if k_prompt:
            prompt_top_logprobs = [None if d is None else _top_k(d, k_prompt) for d in final.prompt_logprobs]
    return Response(
        msgspec.json.encode(GenerateResponse(
            request_id=request_id,
            choices=choices,
            prompt_logprobs=prompt_logprobs,
            prompt_top_logprobs=prompt_top_logprobs,
        )),
        media_type="application/json",
    )


@with_cancellation
async def _generate(body: bytes, raw_request: Request) -> Response:
    try:
        req = _decoder.decode(body)
        params = SamplingParams(**req.sampling_params)
    except (msgspec.DecodeError, msgspec.ValidationError, VLLMValidationError, TypeError, ValueError) as e:
        return _error(422, f"invalid generate request: {e}")
    k_sample, k_prompt = req.topk_sample_logprobs, req.topk_prompt_logprobs
    if k_sample < 0 or k_prompt < 0:
        return _error(422, "topk_sample_logprobs and topk_prompt_logprobs must be >= 0")
    if k_sample > (params.logprobs or 0):
        return _error(422, "topk_sample_logprobs exceeds sampling_params.logprobs")
    if k_prompt > (params.prompt_logprobs or 0):
        return _error(422, "topk_prompt_logprobs exceeds sampling_params.prompt_logprobs")
    state = raw_request.app.state
    models = state.openai_serving_models
    if models.is_base_model(req.model):
        lora_request = None
    elif req.model in models.lora_requests:
        lora_request = models.lora_requests[req.model]
    else:
        return _error(404, f"model {req.model!r} is neither the base model nor a loaded adapter")
    params.output_kind = RequestOutputKind.FINAL_ONLY
    prompt = TokensPrompt(prompt_token_ids=req.token_ids)
    if req.cache_salt is not None:
        prompt["cache_salt"] = req.cache_salt
    request_id = random_uuid()
    final = None
    try:
        async for res in state.engine_client.generate(
            prompt, params, request_id=request_id, lora_request=lora_request
        ):
            final = res
    except (VLLMValidationError, ValueError) as e:  # the engine rejected the request (e.g. prompt too long)
        return _error(400, str(e))
    if final is None:
        return _error(500, "engine returned no output")
    return _response(request_id, req, final, k_sample, k_prompt)


router = APIRouter()


@router.post("/tinkercloud/v1/generate")
async def generate(raw_request: Request) -> Response:
    # with_cancellation consumes the request's receive channel to watch for a
    # disconnect, so the body is read before it starts.
    return await _generate(await raw_request.body(), raw_request)


class TinkerCloudEndpoints:
    """The `vllm.endpoint_plugins` entry. The route reads the engine client and
    the adapter registry from app.state per request, so init_state has nothing
    to set up."""

    name = "tinkercloud"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        app.include_router(router)

    async def init_state(self, engine_client: Optional[EngineClient], state: State, args: Namespace) -> None:
        pass
