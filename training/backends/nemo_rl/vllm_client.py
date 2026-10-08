"""
vLLM worker /tinkercloud/v1/generate codec over a pooled httpx client (specs/019, D15).

One request per sample(): the token-id prompt plus the full SamplingParams
(n = num_samples, per-request seed, stop ids and strings, logprobs, prompt
logprobs, detokenize). A cancelled request closes the connection, which is
how the worker aborts the generation. NeMo RL's async vLLM worker serves the
route when expose_http_server is set.
"""
import math
from typing import Any, Dict, List, Optional

import httpx
import orjson

from ..base import BackendError

# The worker floors a non-finite logprob to this (JSON has no infinity, SkyRL's
# convention); at or below it the value goes back to -inf on our wire.
CLAMPED_LOGPROB = -9999.0


def _unclamp(value: float) -> float:
    return -math.inf if value <= CLAMPED_LOGPROB else float(value)

class VllmGenerateClient:
    """The generate codec for one worker URL."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/tinkercloud/v1/generate"
        self._http = http

    @staticmethod
    def build_request(
        token_ids: List[int],
        sampling_params: Dict[str, Any],
        num_samples: int,
        prompt_logprobs: bool,
        cache_salt: str,
    ) -> Dict[str, Any]:
        """The validated API SamplingParams (temperature / top_p / max_tokens
        always present) as one vLLM SamplingParams. vLLM detokenizes only when
        stop STRINGS must be matched: TinkerCloud decodes text from the tokens
        itself, and a tokenizer on the request makes vLLM's front end decode a
        string per logprob entry per step that the route then discards
        (specs/019, 1.6 s per 2048x256 tokens). Logprob-only requests never
        detokenize (vLLM decodes the -1 sentinel in prompt-logprob tensors,
        BUG-013) and so carry no stop strings; nothing of theirs is generated."""
        temperature = float(sampling_params["temperature"])
        sp: Dict[str, Any] = {
            "n": num_samples,
            "max_tokens": int(sampling_params["max_tokens"]),
            "temperature": 0.0 if temperature <= 0.01 else temperature,
            "top_p": float(sampling_params["top_p"]),
            "logprobs": 0,   # the sampled token's logprob only; the flat route emits it as {"content": [{"logprob": x}]}
        }
        stop = sampling_params.get("stop")
        stop_strings: List[str] = []
        if stop and not prompt_logprobs:
            stop_strings = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]
        sp["detokenize"] = bool(stop_strings)
        if stop_strings:
            sp["stop"] = stop_strings
            sp["include_stop_str_in_output"] = True
        if sampling_params.get("top_k") is not None and int(sampling_params["top_k"]) > 0:
            sp["top_k"] = int(sampling_params["top_k"])
        if sampling_params.get("seed") is not None:
            sp["seed"] = int(sampling_params["seed"])   # child i is seeded seed + i by vLLM
        if sampling_params.get("stop_token_ids"):
            sp["stop_token_ids"] = [int(t) for t in sampling_params["stop_token_ids"]]
        if prompt_logprobs:
            sp["prompt_logprobs"] = 1
        return {"token_ids": list(token_ids), "sampling_params": sp, "cache_salt": cache_salt}

    def parse_response(
        self, body: Dict[str, Any], token_ids: List[int], num_samples: int, prompt_logprobs: bool,
    ) -> Dict[str, Any]:
        """Rows in choice order: tokens, sampled logprobs, finish_reason; plus the
        prompt logprobs (position 0 None, BUG-013 convention) when requested.
        The route floors a non-finite logprob to -9999; that is -inf here."""
        choices = body["choices"]
        if len(choices) != num_samples:
            raise BackendError(
                f"vLLM worker {self.base_url} returned {len(choices)} of {num_samples} sequences",
                backend="nemo_rl", operation="sample",
            )
        rows = []
        for choice in sorted(choices, key=lambda c: c["index"]):
            tokens = [int(t) for t in choice["token_ids"]]
            finish = choice["finish_reason"]
            if finish == "abort":
                raise BackendError(
                    f"vLLM worker {self.base_url} aborted the request (engine pause or shutdown)",
                    backend="nemo_rl", operation="sample",
                )
            logprobs = [_unclamp(e["logprob"]) for e in choice["logprobs"]["content"]]
            if len(logprobs) != len(tokens):
                raise BackendError(
                    f"vLLM worker {self.base_url} returned {len(tokens)} tokens but "
                    f"{len(logprobs)} logprobs",
                    backend="nemo_rl", operation="sample",
                )
            rows.append({"tokens": tokens, "logprobs": logprobs, "finish_reason": finish})
        prompt_lp: Optional[List[Optional[float]]] = None
        if prompt_logprobs:
            entries = body["prompt_logprobs"]
            if entries is None or len(entries) != len(token_ids):
                raise BackendError(
                    f"vLLM worker {self.base_url} returned prompt logprobs for "
                    f"{None if entries is None else len(entries)} of {len(token_ids)} prompt tokens",
                    backend="nemo_rl", operation="compute_logprobs",
                )
            prompt_lp = [None] + [_unclamp(v) for v in entries[1:]]
        return {"rows": rows, "prompt_logprobs": prompt_lp}

    async def generate(
        self,
        token_ids: List[int],
        sampling_params: Dict[str, Any],
        num_samples: int,
        prompt_logprobs: bool,
        cache_salt: str,
    ) -> Dict[str, Any]:
        body = self.build_request(token_ids, sampling_params, num_samples, prompt_logprobs, cache_salt)
        try:
            response = await self._http.post(self.generate_url, json=body)
        except httpx.TransportError as e:
            # A worker that cannot be reached: the model is one unit, a dead
            # leader breaks its refit too, so no retry and no other leader.
            raise BackendError(
                f"vLLM worker {self.base_url} unreachable: {e}",
                backend="nemo_rl", operation="sample", original_error=e,
            ) from e
        if response.status_code != 200:
            raise BackendError(
                f"vLLM worker {self.base_url} returned {response.status_code}: {response.text[:300]}",
                backend="nemo_rl", operation="sample",
            )
        # orjson: the body is ~160 bytes per sampled token (vLLM's per-token logprob
        # objects); stdlib json cost 0.8 s per 2048x256-token step on the event loop.
        return self.parse_response(orjson.loads(response.content), token_ids, num_samples, prompt_logprobs)


class VllmControlClient:
    """The refit routes of one worker URL (specs/021, D16): the worker serves
    them on the same loop as generate, so the engine core sees one sender."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self._http = http

    async def _post(self, route: str) -> bool:
        url = f"{self.base_url}/tinkercloud/v1/{route}"
        try:
            response = await self._http.post(url)
        except httpx.TransportError as e:
            raise BackendError(
                f"vLLM worker {self.base_url} unreachable during {route}: {e}",
                backend="nemo_rl", operation="refit", original_error=e,
            ) from e
        if response.status_code != 200:
            raise BackendError(
                f"vLLM worker {self.base_url} returned {response.status_code} on {route}: {response.text[:300]}",
                backend="nemo_rl", operation="refit",
            )
        return bool(orjson.loads(response.content)["ok"])

    async def update_weights_from_collective(self) -> bool:
        """True when the engine loaded the broadcast weights."""
        return await self._post("update_weights_from_collective")

    async def reset_prefix_cache(self) -> None:
        await self._post("reset_prefix_cache")
