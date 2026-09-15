"""
SGLang Inference Client

Async HTTP client for SGLang inference server.
Handles generation with defensive prompt logprobs extraction.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SGLangClient:
    """
    Async HTTP client for SGLang inference server.

    Handles text generation with support for:
    - Input token IDs
    - Sampling parameters (temperature, top_p, max_tokens)
    - Prompt logprobs extraction with defensive None handling
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize SGLang client.

        Args:
            base_url: SGLang server URL (e.g., "http://router:8000")
            timeout: HTTP request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generate_url = f"{self.base_url}/generate"

    async def generate(
        self,
        input_ids: List[int],
        sampling_params: Dict[str, Any],
        prompt_logprobs: bool = False,
        lora_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate text completion from SGLang.

        Args:
            input_ids: List of input token IDs
            sampling_params: the validated SamplingParams dict (the API schema
                fills temperature / top_p; max_tokens is required there), with:
                - temperature, top_p, max_tokens (always present)
                - top_k: int, stop: list[str], stop_token_ids: list[int], seed: int
            prompt_logprobs: Whether to return prompt log probabilities

        Returns:
            Dict with:
                - tokens: List[int] - Generated token IDs
                - logprobs: List[float] - Log probabilities for generated tokens
                - text: Optional[str] - Generated text
                - stop_reason: str - "stop" or "length"
                - prompt_logprobs: Optional[List[Optional[float]]] - Prompt logprobs if requested

        Raises:
            httpx.HTTPError: If SGLang request fails
            ValueError: If response format is invalid
        """
        # Build request payload
        sgl_params: Dict[str, Any] = {
            "temperature": sampling_params["temperature"],
            "top_p": sampling_params["top_p"],
            "max_new_tokens": sampling_params["max_tokens"],
        }
        # Every documented Tinker sampling parameter is forwarded, none dropped.
        if sampling_params.get("top_k") is not None and int(sampling_params["top_k"]) > 0:
            sgl_params["top_k"] = int(sampling_params["top_k"])
        if sampling_params.get("stop"):
            sgl_params["stop"] = list(sampling_params["stop"])
        if sampling_params.get("stop_token_ids"):
            sgl_params["stop_token_ids"] = [int(t) for t in sampling_params["stop_token_ids"]]
        if sampling_params.get("seed") is not None:
            sgl_params["sampling_seed"] = int(sampling_params["seed"])
        payload = {
            "input_ids": input_ids,
            "sampling_params": sgl_params,
            "return_logprob": True,
        }

        # Multi-LoRA pool: route to the model's adapter (engine-side slot name).
        if lora_path is not None:
            payload["lora_path"] = lora_path

        # Request prompt logprobs if needed
        if prompt_logprobs:
            payload["logprob_start_len"] = 0
            logger.debug("Requesting prompt logprobs from SGLang")

        # Make async HTTP request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                sglang_output = response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"SGLang HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except httpx.RequestError as e:
                logger.error(f"SGLang request error: {e}")
                raise
            except Exception as e:
                logger.error(f"SGLang unexpected error: {e}")
                raise

        # Extract tokens and logprobs from response
        try:
            meta_info = sglang_output["meta_info"]

            # Output tokens and logprobs
            # Format: [(logprob, token_id), ...]
            token_logprobs = meta_info["output_token_logprobs"]
            output_tokens = [item[1] for item in token_logprobs]
            output_logprobs = [item[0] for item in token_logprobs]

            # Build result
            # finish_reason lives in meta_info as {"type": "stop"|"length", ...}
            # (older builds: a top-level string).
            finish = meta_info.get("finish_reason", sglang_output.get("finish_reason"))
            finish_type = finish.get("type") if isinstance(finish, dict) else finish
            result = {
                "tokens": output_tokens,
                "logprobs": output_logprobs,
                "text": sglang_output.get("text"),
                "stop_reason": "stop" if finish_type == "stop" else "length",
            }

            # Extract prompt logprobs if requested
            if prompt_logprobs:
                input_logprobs = meta_info.get("input_token_logprobs", [])

                # One entry per prompt token, position 0 = None (no logprob for
                # the first token), matching the NeMo RL backend. SGLang has
                # emitted the prompt entries with or without a leading None
                # sentinel; normalise both to the same shape.
                normalized = self._normalize_logprob_entries(input_logprobs)
                tail = normalized[1:] if normalized and normalized[0] is None else normalized
                result["prompt_logprobs"] = [None] + [0.0 if lp is None else lp for lp in tail]

            logger.debug(f"Generated {len(output_tokens)} tokens from SGLang")
            return result

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse SGLang response: {e}")
            logger.error(f"SGLang response: {sglang_output}")
            raise ValueError(f"Invalid SGLang response format: {e}")

    @staticmethod
    def _normalize_logprob_entries(entries: List[Any]) -> List[Optional[float]]:
        """Convert SGLang token logprob records into a flat list of floats/None.

        Historically SGLang has emitted formats such as:
        - [None, (-0.1, 123, \"a\"), ...]
        - [(-0.2, 456), ...]
        We only care about the first element (the logprob) and preserve None sentinels.
        """
        normalized: List[Optional[float]] = []
        for entry in entries or []:
            if entry is None:
                normalized.append(None)
            elif isinstance(entry, (list, tuple)) and entry:
                try:
                    normalized.append(float(entry[0]))
                except (TypeError, ValueError):
                    normalized.append(None)
            else:
                try:
                    normalized.append(float(entry))
                except (TypeError, ValueError):
                    normalized.append(None)
        return normalized

    async def batch_generate(
        self,
        input_ids_list: List[List[int]],
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple completions (one per input).

        Args:
            input_ids_list: List of input token ID lists
            sampling_params: Sampling parameters
            prompt_logprobs: Whether to return prompt logprobs

        Returns:
            List of generation results (one per input)
        """
        results = []
        for input_ids in input_ids_list:
            result = await self.generate(input_ids, sampling_params, prompt_logprobs)
            results.append(result)
        return results
