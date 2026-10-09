"""
vLLM worker /tinkercloud/v1/generate codec over a pooled httpx client (specs/019, D15).

One request per sample(): the token-id prompt plus the full SamplingParams
(n = num_samples, per-request seed, stop ids and strings, logprobs, prompt
logprobs, detokenize) and the top-k counts the route should emit
(topk_sample_logprobs / topk_prompt_logprobs). A cancelled request closes the
connection, which is how the worker aborts the generation. NeMo RL's async
vLLM worker serves the route when expose_http_server is set.

Wire (the route's msgspec structs): a choice is {index, finish_reason,
token_ids, logprobs: {content: [{logprob, top_logprobs?}]}} where top_logprobs
= [{token, logprob}, ...] best first, present only at topk_sample_logprobs > 0;
prompt_logprobs is a flat list (None at 0) and prompt_top_logprobs, present
only at topk_prompt_logprobs > 0, holds one such list per prompt position.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import httpx
import orjson

from ..base import BackendError

# The worker floors a non-finite logprob to this (JSON has no infinity, SkyRL's
# convention); at or below it the value goes back to -inf on our wire.
CLAMPED_LOGPROB = -9999.0


def _unclamp(value: float) -> float:
    return -math.inf if value <= CLAMPED_LOGPROB else float(value)


def _topk_row(entries: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    return [(int(e["token"]), _unclamp(e["logprob"])) for e in entries]

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
        topk_sample_logprobs: int = 0,
        topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        """The validated API SamplingParams (temperature / top_p / max_tokens
        always present) as one vLLM SamplingParams. vLLM detokenizes only when
        stop STRINGS must be matched: TinkerCloud decodes text from the tokens
        itself, and a tokenizer on the request makes vLLM's front end decode a
        string per logprob entry per step that the route then discards
        (specs/019, 1.6 s per 2048x256 tokens). Prompt-scoring requests never
        detokenize (vLLM decodes the -1 sentinel in prompt-logprob tensors,
        BUG-013) and so carry no stop strings; nothing of theirs is generated.
        vLLM's `logprobs` / `prompt_logprobs` say how many it computes per
        position (0 = the sampled token only; prompt scoring needs >= 1); the
        route emits top-k entries only for the topk_* counts beside them."""
        temperature = float(sampling_params["temperature"])
        score_prompt = prompt_logprobs or topk_prompt_logprobs > 0
        sp: Dict[str, Any] = {
            "n": num_samples,
            "max_tokens": int(sampling_params["max_tokens"]),
            "temperature": 0.0 if temperature <= 0.01 else temperature,
            "top_p": float(sampling_params["top_p"]),
            "logprobs": topk_sample_logprobs,   # 0: the sampled token's logprob only, as {"content": [{"logprob": x}]}
        }
        stop = sampling_params.get("stop")
        stop_strings: List[str] = []
        if stop and not score_prompt:
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
        if score_prompt:
            sp["prompt_logprobs"] = max(1, topk_prompt_logprobs)
        return {
            "token_ids": list(token_ids), "sampling_params": sp, "cache_salt": cache_salt,
            "topk_sample_logprobs": topk_sample_logprobs, "topk_prompt_logprobs": topk_prompt_logprobs,
        }

    def parse_response(
        self, body: Dict[str, Any], token_ids: List[int], num_samples: int, prompt_logprobs: bool,
        topk_sample_logprobs: int = 0, topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        """Rows in choice order: tokens, sampled logprobs, finish_reason, and
        topk_logprobs (one (token_id, logprob) row per token, best first) when
        topk_sample_logprobs > 0; plus the prompt logprobs (position 0 None,
        BUG-013 convention) when requested and topk_prompt_logprobs rows in the
        same layout when asked. The route floors a non-finite logprob to -9999;
        that is -inf here. A route that predates top-k answers without the
        top_logprobs keys: that is a worker error, not a k=0 response."""
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
            content = choice["logprobs"]["content"]
            logprobs = [_unclamp(e["logprob"]) for e in content]
            if len(logprobs) != len(tokens):
                raise BackendError(
                    f"vLLM worker {self.base_url} returned {len(tokens)} tokens but "
                    f"{len(logprobs)} logprobs",
                    backend="nemo_rl", operation="sample",
                )
            row: Dict[str, Any] = {"tokens": tokens, "logprobs": logprobs, "finish_reason": finish}
            if topk_sample_logprobs:
                if any("top_logprobs" not in e for e in content):
                    raise BackendError(
                        f"vLLM worker {self.base_url} returned no top_logprobs for "
                        f"topk_sample_logprobs={topk_sample_logprobs} (route predates top-k)",
                        backend="nemo_rl", operation="sample",
                    )
                row["topk_logprobs"] = [_topk_row(e["top_logprobs"]) for e in content]
            rows.append(row)
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
        prompt_topk: Optional[List[Optional[List[Tuple[int, float]]]]] = None
        if topk_prompt_logprobs:
            top_entries = body["prompt_top_logprobs"] if "prompt_top_logprobs" in body else None
            if top_entries is None or len(top_entries) != len(token_ids):
                raise BackendError(
                    f"vLLM worker {self.base_url} returned prompt top-k for "
                    f"{None if top_entries is None else len(top_entries)} of {len(token_ids)} prompt tokens "
                    f"(topk_prompt_logprobs={topk_prompt_logprobs}; a missing key means the route predates top-k)",
                    backend="nemo_rl", operation="compute_logprobs",
                )
            prompt_topk = [None] + [_topk_row(e) for e in top_entries[1:]]
        return {"rows": rows, "prompt_logprobs": prompt_lp, "topk_prompt_logprobs": prompt_topk}

    async def generate(
        self,
        token_ids: List[int],
        sampling_params: Dict[str, Any],
        num_samples: int,
        prompt_logprobs: bool,
        cache_salt: str,
        topk_sample_logprobs: int = 0,
        topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        body = self.build_request(
            token_ids, sampling_params, num_samples, prompt_logprobs, cache_salt,
            topk_sample_logprobs=topk_sample_logprobs, topk_prompt_logprobs=topk_prompt_logprobs,
        )
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
        return self.parse_response(
            orjson.loads(response.content), token_ids, num_samples, prompt_logprobs,
            topk_sample_logprobs=topk_sample_logprobs, topk_prompt_logprobs=topk_prompt_logprobs,
        )


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
