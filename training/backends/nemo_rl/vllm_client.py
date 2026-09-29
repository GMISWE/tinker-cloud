"""
vLLM /inference/v1/generate codec over a pooled httpx client (specs/019, D15).

One request per sample(): the token-id prompt plus the full SamplingParams
(n = num_samples, per-request seed, stop ids and strings, logprobs, prompt
logprobs, detokenize). A cancelled request closes the connection, which is
how the worker aborts the generation. NeMo RL's async vLLM worker serves the
route when expose_http_server is set.
"""
import math
from typing import Any, Dict, List, Optional

import httpx

from ..base import BackendError

# vLLM's HTTP layer writes -inf as this (JSON has no infinity); a value at or
# below it is a genuine "zero probability" and goes back to -inf on our wire.
NEG_INF_CLAMP = -9999.0
# vllm.sampling_params.RequestOutputKind.FINAL_ONLY: the route answers from the
# last RequestOutput, and with n > 1 only this kind carries every child in it
# (a child that finished on an earlier step is otherwise left out).
OUTPUT_KIND_FINAL_ONLY = 2


def _unclamp(value: float) -> float:
    return -math.inf if value <= NEG_INF_CLAMP else float(value)


class VllmGenerateClient:
    """The generate codec for one worker URL."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/inference/v1/generate"
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
        always present) as one vLLM SamplingParams. Logprob-only requests skip
        detokenization (vLLM decodes the -1 sentinel in prompt-logprob tensors,
        BUG-013) and so carry no stop strings; nothing of theirs is generated."""
        temperature = float(sampling_params["temperature"])
        sp: Dict[str, Any] = {
            "n": num_samples,
            "max_tokens": int(sampling_params["max_tokens"]),
            "temperature": 0.0 if temperature <= 0.01 else temperature,
            "top_p": float(sampling_params["top_p"]),
            "logprobs": 1,   # 0 means off on this route; each entry carries the sampled token
            "include_stop_str_in_output": True,
            "detokenize": not prompt_logprobs,
            "output_kind": OUTPUT_KIND_FINAL_ONLY,
        }
        if sampling_params.get("top_k") is not None and int(sampling_params["top_k"]) > 0:
            sp["top_k"] = int(sampling_params["top_k"])
        if sampling_params.get("seed") is not None:
            sp["seed"] = int(sampling_params["seed"])   # child i is seeded seed + i by vLLM
        if sampling_params.get("stop_token_ids"):
            sp["stop_token_ids"] = [int(t) for t in sampling_params["stop_token_ids"]]
        stop = sampling_params.get("stop")
        if stop and not prompt_logprobs:
            sp["stop"] = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]
        if prompt_logprobs:
            sp["prompt_logprobs"] = 1
        return {"token_ids": list(token_ids), "sampling_params": sp, "cache_salt": cache_salt}

    def parse_response(
        self, body: Dict[str, Any], token_ids: List[int], num_samples: int, prompt_logprobs: bool,
    ) -> Dict[str, Any]:
        """Rows in choice order: tokens, sampled logprobs, finish_reason; plus the
        prompt logprobs (position 0 None, BUG-013 convention) when requested."""
        if len(body["choices"]) != num_samples:
            raise BackendError(
                f"vLLM worker {self.base_url} returned {len(body['choices'])} of "
                f"{num_samples} sequences",
                backend="nemo_rl", operation="sample",
            )
        rows = []
        for choice in sorted(body["choices"], key=lambda c: c["index"]):
            finish = choice["finish_reason"]
            if finish == "abort":
                raise BackendError(
                    f"vLLM worker {self.base_url} aborted the generation",
                    backend="nemo_rl", operation="sample",
                )
            tokens = list(choice["token_ids"])
            content = (choice["logprobs"] or {}).get("content") or []
            logprobs = [_unclamp(entry["logprob"]) for entry in content]
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
                    f"{0 if entries is None else len(entries)} of {len(token_ids)} prompt tokens",
                    backend="nemo_rl", operation="sample",
                )
            prompt_lp = [None]
            for pos in range(1, len(token_ids)):
                # JSON keys are strings; vLLM always includes the prompt token itself
                entry = (entries[pos] or {}).get(str(token_ids[pos]))
                if entry is None:
                    raise BackendError(
                        f"vLLM worker {self.base_url} returned no logprob for prompt "
                        f"position {pos}",
                        backend="nemo_rl", operation="sample",
                    )
                prompt_lp.append(_unclamp(entry["logprob"]))
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
        return self.parse_response(response.json(), token_ids, num_samples, prompt_logprobs)
