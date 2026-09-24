"""
NeMo RL per-request generation (specs/019).

Every sample() is one generate_rows_async call on the async vLLM engine, its
rows independent vLLM requests: samples complete independently, cancelling a
sample cancels its worker task (which aborts its rows' requests), and an
engine error fails only its own sample. Calls in flight per model are bounded
because Ray queues async-actor tasks past ~1000 per worker.
"""
import asyncio
import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class NemoRLRequestSubmitter:
    """One generate_rows_async call per sample, at most `max_in_flight` at a time."""

    def __init__(self, max_in_flight: int):
        self._sem = asyncio.Semaphore(max_in_flight)

    async def submit(
        self,
        handle: Any,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Dict[str, Any],
        prompt_logprobs: bool,
    ) -> Dict[str, Any]:
        import torch
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        params = sampling_params
        stop_strings = _stop_strings(params)
        stop_ids = [int(t) for t in (params.get("stop_token_ids") or [])]
        temperature = float(params.get("temperature", 0.7))
        seed = params.get("seed")
        n = num_samples
        # Sampling parameters travel as per-row `_tinker_*` columns; the vLLM
        # worker builds each row's SamplingParams from them. Rows differ only
        # by seed (a distinct stream per row).
        data = BatchedDataDict({
            "input_ids": torch.tensor([prompt_tokens] * n, dtype=torch.long),
            "input_lengths": torch.tensor([len(prompt_tokens)] * n, dtype=torch.long),
            "_tinker_max_new_tokens": [int(params.get("max_tokens") or 256)] * n,
            "_tinker_temperature": [0.0 if temperature <= 0.01 else temperature] * n,
            "_tinker_top_p": [float(params.get("top_p", 0.9))] * n,
            "_tinker_top_k": [params.get("top_k")] * n,
            "_tinker_seed": [None if seed is None else int(seed) + i for i in range(n)],
            "_tinker_stop_token_ids": [stop_ids] * n,
            "stop_strings": [stop_strings] * n,
            "_tinker_prompt_logprobs": [bool(prompt_logprobs)] * n,
        })
        async with self._sem:
            rows = await handle.policy_generation.generate_rows_async(data, False)
        results = [
            _row_result(handle.tokenizer, out, len(prompt_tokens), stop_strings, set(stop_ids), prompt_logprobs)
            for _idx, out in sorted(rows, key=lambda r: r[0])
        ]
        logger.info(
            "[%s] per-request generate: %d rows, avg %.0f tokens/row",
            request_id, num_samples,
            sum(len(r["sequence"]["tokens"]) for r in results) / max(num_samples, 1),
        )
        return {
            "sequences": [r["sequence"] for r in results],
            "prompt_logprobs": results[0]["prompt_logprobs"] if results else None,
        }


def _stop_strings(params: Dict[str, Any]) -> List[str]:
    raw = params.get("stop")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    return []


def _row_result(
    tokenizer: Any,
    out: Any,
    prompt_len: int,
    stop_strings: List[str],
    stop_ids: Set[int],
    want_prompt_logprobs: bool,
) -> Dict[str, Any]:
    """One row's sequence (stop-string truncation, stop_reason) and prompt logprobs."""
    gen_len = int(out["generation_lengths"][0])
    logprobs_tensor = out["logprobs"]
    out_tokens = out["output_ids"][0, prompt_len:prompt_len + gen_len].tolist()
    out_logprobs = logprobs_tensor[0, prompt_len:prompt_len + gen_len].tolist()
    text = tokenizer.decode(out_tokens)
    eos_id = tokenizer.eos_token_id

    stop_reason = "length"
    if out_tokens and ((eos_id is not None and out_tokens[-1] == eos_id) or out_tokens[-1] in stop_ids):
        stop_reason = "stop"
    elif text and stop_strings:
        for ss in stop_strings:
            if ss in text:
                stop_reason = "stop"
                break

    # Truncate at the earliest stop string (kept in the output, as vLLM does)
    if text and stop_strings:
        earliest_pos = len(text)
        matched_stop = None
        for ss in stop_strings:
            pos = text.find(ss)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                matched_stop = ss
        if matched_stop is not None:
            truncated_text = text[:earliest_pos + len(matched_stop)]
            if len(truncated_text) < len(text):
                trunc_count = len(out_tokens)
                for t in range(1, len(out_tokens) + 1):
                    decoded = tokenizer.decode(out_tokens[:t], skip_special_tokens=False)
                    if len(decoded) >= len(truncated_text):
                        trunc_count = t
                        break
                out_tokens = out_tokens[:trunc_count]
                out_logprobs = out_logprobs[:trunc_count]
                text = tokenizer.decode(out_tokens)
                stop_reason = "stop"

    prompt_logprobs_result = None
    if want_prompt_logprobs and logprobs_tensor.shape[1] >= prompt_len:
        raw = logprobs_tensor[0, :prompt_len].tolist()
        # Only position 0 has no logprob; interior values are real (BUG-013)
        prompt_logprobs_result = [None] + raw[1:] if raw else []

    return {
        "sequence": {
            "tokens": out_tokens,
            "logprobs": out_logprobs,
            "text": text,
            "stop_reason": stop_reason,
        },
        "prompt_logprobs": prompt_logprobs_result,
    }
