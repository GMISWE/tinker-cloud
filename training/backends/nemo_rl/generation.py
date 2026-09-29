"""
NeMo RL per-request sampling over the workers' HTTP servers (specs/019, D15).

Every sample() is one /inference/v1/generate request to the next
data-parallel leader of the model, taken from the routing table with a
pooled client, the same shape as Miles over its SGLang router. The request
carries the full SamplingParams (n = num_samples, vLLM seeds child i with
seed + i) and a cache salt keyed on the weight version the engine holds, so
a block computed by earlier weights is never reused. Cancelling the sample
task closes the request; the worker aborts the generation.
"""
import logging
from typing import Any, Dict, List

from ...core import routing
from ..base import BackendError
from ..http_pool import HttpClientPool
from .vllm_client import VllmGenerateClient

logger = logging.getLogger(__name__)


def cache_salt_for(handle: Any) -> str:
    """Prefix-cache namespace of the weights the engine holds right now."""
    return f"{handle.model_id}@{handle.generation_synced_version}"


def next_leader_url(handle: Any) -> str:
    """Round-robin over the model's published DP-leader servers."""
    try:
        endpoint = routing.table.endpoint_for(handle.model_id)
    except routing.RoutingError as e:
        raise BackendError(
            "vLLM workers not available", backend="nemo_rl", operation="sample",
        ) from e
    url = endpoint.base_urls[handle.next_leader % len(endpoint.base_urls)]
    handle.next_leader += 1
    return url


async def sample_over_http(
    handle: Any,
    pool: HttpClientPool,
    request_id: str,
    prompt_tokens: List[int],
    num_samples: int,
    sampling_params: Dict[str, Any],
    prompt_logprobs: bool,
) -> Dict[str, Any]:
    url = next_leader_url(handle)
    client = VllmGenerateClient(url, pool.for_url(url))
    out = await client.generate(
        token_ids=prompt_tokens,
        sampling_params=sampling_params,
        num_samples=num_samples,
        prompt_logprobs=prompt_logprobs,
        cache_salt=cache_salt_for(handle),
    )
    tokenizer = handle.tokenizer
    sequences = [
        {
            "tokens": row["tokens"],
            "logprobs": row["logprobs"],
            "text": tokenizer.decode(row["tokens"]),
            # vLLM stops at the token that completes a stop string or stop id;
            # tokens are returned as generated (S4), text = decode(tokens).
            "stop_reason": "stop" if row["finish_reason"] == "stop" else "length",
        }
        for row in out["rows"]
    ]
    logger.info(
        "[%s] vLLM generate on %s: %d rows, avg %.0f tokens/row",
        request_id, url, len(sequences),
        sum(len(s["tokens"]) for s in sequences) / max(len(sequences), 1),
    )
    return {"sequences": sequences, "prompt_logprobs": out["prompt_logprobs"]}
