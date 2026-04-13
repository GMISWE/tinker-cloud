"""
Sampling Service - Business Logic for Model Sampling

Handles:
- Async sampling via SGLang (Miles backend)
- Async sampling via Policy.generate() (NeMo RL backend) with batch accumulation
- Sync sampling via SGLang
- SGLang sampling client creation
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..utils.sglang_client import SGLangClient
from ..utils.helpers import find_model_with_rollout_manager

logger = logging.getLogger(__name__)


@dataclass
class _PendingGenRequest:
    """A single sampling request waiting to be batched."""
    prompt_tokens: List[int]
    num_samples: int
    sampling_params: Dict[str, Any]
    prompt_logprobs: bool
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class NemoRLBatchAccumulator:
    """Accumulates per-sample NeMo RL generation requests and flushes them
    as a single batched generate() call for dramatically better throughput.

    PERF-002: Without batching, 2048 RL rollouts → 2048 sequential generate(1)
    calls (~34 min). With batching, one generate(2048) call (~1-2 min).
    """

    def __init__(self, flush_interval_ms: int = 50, max_batch_size: int = 4096):
        self._queue: List[_PendingGenRequest] = []
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._flush_interval = flush_interval_ms / 1000.0
        self._max_batch_size = max_batch_size
        self._flush_task: Optional[asyncio.Task] = None

    async def submit(
        self,
        handle: Any,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Dict[str, Any],
        prompt_logprobs: bool,
    ) -> Dict[str, Any]:
        """Submit a request for batching. Returns when the batch is flushed."""
        req = _PendingGenRequest(
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params,
            prompt_logprobs=prompt_logprobs,
        )
        async with self._lock:
            self._queue.append(req)
            queue_len = len(self._queue)
            if queue_len >= self._max_batch_size:
                self._flush_event.set()
            elif self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(
                    self._delayed_flush(handle, request_id)
                )
        return await req.future

    async def _delayed_flush(self, handle: Any, request_id: str):
        """Wait for flush_interval or until max_batch_size reached, then flush.

        Loops to drain any requests that arrived during the generate() call.
        """
        try:
            await asyncio.wait_for(
                self._flush_event.wait(), timeout=self._flush_interval,
            )
        except asyncio.TimeoutError:
            pass

        # Keep flushing until the queue is empty — new requests may arrive
        # while generate() is running (which takes seconds to minutes).
        while True:
            had_work = await self._flush(handle, request_id)
            if not had_work:
                break

    async def _flush(self, handle: Any, request_id: str) -> bool:
        """Flush all queued requests as a single batched generate() call.

        Returns True if work was done, False if queue was empty.
        """
        async with self._lock:
            if not self._queue:
                return False
            batch = self._queue[:]
            self._queue.clear()
            self._flush_event.clear()

        try:
            results = await _batched_nemo_rl_generate(handle, request_id, batch)
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        return True


async def _batched_nemo_rl_generate(
    handle: Any,
    request_id: str,
    batch: List[_PendingGenRequest],
) -> List[Dict[str, Any]]:
    """Generate completions for a batch of requests in a single vLLM call.

    Combines all prompts into one BatchedDataDict, calls generate() once,
    then splits results back to per-request responses.
    """
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from ..backends.nemo_rl.backend import _ensure_generation_ready

    # Expand requests: each request with num_samples>1 becomes multiple rows
    expanded_prompts = []
    expanded_params = []
    request_indices = []  # maps expanded index → (request_idx, sample_idx)
    for req_idx, req in enumerate(batch):
        for sample_idx in range(req.num_samples):
            expanded_prompts.append(req.prompt_tokens)
            expanded_params.append(req.sampling_params)
            request_indices.append((req_idx, sample_idx))

    total_samples = len(expanded_prompts)
    if total_samples == 0:
        return [{} for _ in batch]

    # Find max prompt length and pad with the model's pad token
    # (verify_right_padding checks that positions after input_length == pad_token_id)
    tokenizer = handle.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
    max_prompt_len = max(len(p) for p in expanded_prompts)
    input_ids = torch.full((total_samples, max_prompt_len), pad_id, dtype=torch.long)
    input_lengths = torch.zeros(total_samples, dtype=torch.long)
    for i, prompt in enumerate(expanded_prompts):
        plen = len(prompt)
        input_ids[i, :plen] = torch.tensor(prompt, dtype=torch.long)
        input_lengths[i] = plen

    # Pad to dp_size
    dp_size = handle.config.get("dp_size", 1)
    actual_size = total_samples
    padded_size = actual_size
    if actual_size % dp_size != 0:
        pad_count = dp_size - (actual_size % dp_size)
        padded_size = actual_size + pad_count
        input_ids = torch.cat([
            input_ids,
            input_ids[-1:].repeat(pad_count, 1),
        ])
        input_lengths = torch.cat([
            input_lengths,
            input_lengths[-1:].repeat(pad_count),
        ])

    data = BatchedDataDict({
        "input_ids": input_ids,
        "input_lengths": input_lengths,
    })

    # Extract sampling params from first request (batch assumes uniform params)
    # For heterogeneous params, use per-sample _tinker_ fields
    first_params = batch[0].sampling_params or {}
    max_new_tokens = first_params.get("max_tokens", 256)
    temperature = first_params.get("temperature", 0.7)
    top_p = first_params.get("top_p", 0.9)
    greedy = temperature <= 0.01

    # Per-sample params via _tinker_ fields
    data["_tinker_max_new_tokens"] = [
        (expanded_params[i] or {}).get("max_tokens", max_new_tokens)
        for i in range(actual_size)
    ] + [max_new_tokens] * (padded_size - actual_size)
    data["_tinker_temperature"] = [
        (expanded_params[i] or {}).get("temperature", temperature)
        for i in range(actual_size)
    ] + [temperature] * (padded_size - actual_size)
    data["_tinker_top_p"] = [
        (expanded_params[i] or {}).get("top_p", top_p)
        for i in range(actual_size)
    ] + [top_p] * (padded_size - actual_size)

    # Stop strings
    raw_stop = first_params.get("stop")
    stop_strings: List[str] = []
    if raw_stop is not None:
        if isinstance(raw_stop, str):
            stop_strings = [raw_stop]
        elif isinstance(raw_stop, list) and raw_stop and isinstance(raw_stop[0], str):
            stop_strings = list(raw_stop)
    if stop_strings:
        data["stop_strings"] = [stop_strings] * padded_size

    # Prompt logprobs
    any_prompt_logprobs = any(req.prompt_logprobs for req in batch)
    if any_prompt_logprobs:
        data["_tinker_prompt_logprobs"] = [True] * padded_size

    logger.info(
        "[%s] Batched NeMo RL generate: %d requests → %d samples (padded to %d)",
        request_id, len(batch), actual_size, padded_size,
    )

    # Generate — single call for entire batch
    await _ensure_generation_ready(handle)
    result = await asyncio.to_thread(handle.policy_generation.generate, data, greedy)

    # Split results back to per-request responses
    output_ids = result["output_ids"]
    gen_lengths = result["generation_lengths"]
    logprobs_tensor = result["logprobs"]
    tokenizer = handle.tokenizer
    eos_id = tokenizer.eos_token_id if tokenizer else None

    # Build per-expanded-sample results
    sample_results = []
    for i in range(actual_size):
        prompt_len = int(input_lengths[i].item())
        gen_len = int(gen_lengths[i].item())
        out_tokens = output_ids[i, prompt_len:prompt_len + gen_len].tolist()
        out_logprobs = logprobs_tensor[i, prompt_len:prompt_len + gen_len].tolist()
        text = tokenizer.decode(out_tokens) if tokenizer else None

        stop_reason = "length"
        if out_tokens and eos_id is not None and out_tokens[-1] == eos_id:
            stop_reason = "stop"
        elif text and stop_strings:
            for ss in stop_strings:
                if ss in text:
                    stop_reason = "stop"
                    break

        # Truncate at stop string
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
                    orig_count = len(out_tokens)
                    trunc_count = orig_count
                    for t in range(1, orig_count + 1):
                        decoded = tokenizer.decode(out_tokens[:t], skip_special_tokens=False)
                        if len(decoded) >= len(truncated_text):
                            trunc_count = t
                            break
                    out_tokens = out_tokens[:trunc_count]
                    out_logprobs = out_logprobs[:trunc_count]
                    text = tokenizer.decode(out_tokens)
                    stop_reason = "stop"

        prompt_logprobs_result = None
        req_idx, _ = request_indices[i]
        if batch[req_idx].prompt_logprobs and logprobs_tensor.shape[1] >= prompt_len:
            raw = logprobs_tensor[i, :prompt_len].tolist()
            prompt_logprobs_result = [None if v == 0.0 else v for v in raw]

        sample_results.append({
            "tokens": out_tokens,
            "logprobs": out_logprobs,
            "text": text,
            "stop_reason": stop_reason,
            "prompt_logprobs_result": prompt_logprobs_result,
        })

    # Reassemble per-request responses
    per_request_results = []
    for req_idx, req in enumerate(batch):
        sequences = []
        prompt_lp = None
        for exp_idx, (ri, si) in enumerate(request_indices):
            if ri == req_idx:
                sr = sample_results[exp_idx]
                sequences.append({
                    "tokens": sr["tokens"],
                    "logprobs": sr["logprobs"],
                    "text": sr["text"],
                    "stop_reason": sr["stop_reason"],
                })
                if prompt_lp is None:
                    prompt_lp = sr["prompt_logprobs_result"]
        per_request_results.append({
            "sequences": sequences,
            "prompt_logprobs": prompt_lp,
        })

    logger.info(
        "[%s] Batched generate complete: %d requests, avg %.0f tokens/sample",
        request_id, len(batch),
        sum(len(s["tokens"]) for s in sample_results) / max(len(sample_results), 1),
    )

    return per_request_results


class SamplingService:
    """Service for managing model sampling via SGLang."""

    def __init__(self):
        """Initialize SamplingService."""
        # PERF-002: Batch accumulator replaces the old per-request lock.
        # Collects concurrent sampling requests and flushes them as a single
        # batched generate() call to vLLM. ~20-30x speedup for RL sampling.
        self._nemo_rl_batch_accumulator = NemoRLBatchAccumulator()
        # Keep the lock for _ensure_generation_ready (state machine safety)
        self._nemo_rl_generate_lock = asyncio.Lock()

    async def async_sample(
        self,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        prompt_logprobs: bool,
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Async sampling via SGLang.

        Args:
            request_id: Request identifier for logging
            prompt_tokens: Input prompt tokens
            num_samples: Number of samples to generate
            sampling_params: Optional sampling parameters dict
            prompt_logprobs: Whether to return prompt logprobs
            training_clients: Global training clients dict

        Returns:
            Dict with sequences and optional prompt_logprobs

        Raises:
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Find model with RolloutManager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Async sampling for {model_id}")

        # Check if this is a NeMo RL backend (vLLM via Ray) or Miles (SGLang via HTTP)
        handle = client_info.get("backend_handle")
        if handle is not None and getattr(handle, "backend_type", None) == "nemo_rl":
            return await self._nemo_rl_sample(
                request_id=request_id,
                handle=handle,
                prompt_tokens=prompt_tokens,
                num_samples=num_samples,
                sampling_params=sampling_params,
                prompt_logprobs=prompt_logprobs,
            )

        # Miles path: SGLang HTTP
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")
        if not router_ip or not router_port:
            raise RuntimeError(f"Router address not available for model {model_id}")

        sglang_url = f"http://{router_ip}:{router_port}"

        # Use SGLang client
        sglang_client = SGLangClient(base_url=sglang_url)
        sequences = []
        prompt_logprobs_result = None

        for i in range(num_samples):
            result = await sglang_client.generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params or {},
                prompt_logprobs=prompt_logprobs
            )

            sequences.append({
                "tokens": result["tokens"],
                "logprobs": result["logprobs"],
                "text": result.get("text"),
                "stop_reason": result.get("stop_reason", "length")
            })

            if prompt_logprobs and prompt_logprobs_result is None:
                prompt_logprobs_result = result.get("prompt_logprobs")

        return {
            "sequences": sequences,
            "prompt_logprobs": prompt_logprobs_result
        }

    async def sync_sample(
        self,
        request_id: str,
        prompts: List[List[int]],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synchronous sampling via SGLang.

        Args:
            request_id: Request identifier for logging
            prompts: List of prompt token sequences
            num_samples: Number of samples per prompt
            sampling_params: Optional sampling parameters dict
            training_clients: Global training clients dict

        Returns:
            Dict with sequences list

        Raises:
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Find model with rollout manager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Sampling {num_samples} sequences")

        # Check if this is a NeMo RL backend
        handle = client_info.get("backend_handle")
        if handle is not None and getattr(handle, "backend_type", None) == "nemo_rl":
            all_sequences = []
            for prompt_tokens in prompts:
                result = await self._nemo_rl_sample(
                    request_id=request_id,
                    handle=handle,
                    prompt_tokens=prompt_tokens,
                    num_samples=num_samples,
                    sampling_params=sampling_params,
                    prompt_logprobs=False,
                )
                all_sequences.extend(result["sequences"])
            logger.info(f"[{request_id}] NeMo RL sampling completed")
            return {"sequences": all_sequences}

        # Miles path: SGLang HTTP
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")
        if not router_ip or not router_port:
            raise RuntimeError("SGLang router not available")

        sglang_url = f"http://{router_ip}:{router_port}"
        sglang_client = SGLangClient(base_url=sglang_url)

        # Sample from all prompts
        all_sequences = []
        for prompt_tokens in prompts:
            for _ in range(num_samples):
                result = await sglang_client.generate(
                    input_ids=prompt_tokens,
                    sampling_params=sampling_params or {},
                    prompt_logprobs=False
                )

                all_sequences.append({
                    "tokens": result["tokens"],
                    "logprobs": result["logprobs"],
                    "text": result.get("text"),
                    "stop_reason": result.get("stop_reason", "length")
                })

        logger.info(f"[{request_id}] Sampling completed")

        return {"sequences": all_sequences}

    async def create_sampling_client(
        self,
        request_id: str,
        model_path: Optional[str],
        base_model: Optional[str],
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create sampling client (SGLang).

        Args:
            request_id: Request identifier for logging
            model_path: Optional model path
            base_model: Optional base model
            training_clients: Global training clients dict

        Returns:
            Dict with sampling_client_id, model_path, status

        Raises:
            ValueError: If neither model_path nor base_model provided
            RuntimeError: If no model with RolloutManager found or router not available
        """
        # Determine model path
        resolved_model_path = model_path or base_model
        if not resolved_model_path:
            raise ValueError("Either model_path or base_model must be provided")

        # Find model with rollout manager
        model_id = find_model_with_rollout_manager(training_clients)
        if not model_id:
            raise RuntimeError("No model with RolloutManager found")

        client_info = training_clients[model_id]

        logger.info(f"[{request_id}] Creating sampling client for {resolved_model_path}")

        # NeMo RL backend: no external router needed, sampling goes through Policy.generate()
        handle = client_info.get("backend_handle")
        if handle is not None and getattr(handle, "backend_type", None) == "nemo_rl":
            sampling_client_id = f"sampler_{uuid.uuid4().hex[:8]}"
            logger.info(f"[{request_id}] NeMo RL sampling client created: {sampling_client_id}")
            return {
                "sampling_client_id": sampling_client_id,
                "model_path": resolved_model_path,
                "status": "ready",
            }

        # Miles path: Get SGLang router info
        router_ip = client_info.get("router_ip")
        router_port = client_info.get("router_port")

        if not router_ip or not router_port:
            raise RuntimeError("SGLang router not available")

        # Generate client ID
        sampling_client_id = f"sampler_{uuid.uuid4().hex[:8]}"

        logger.info(f"[{request_id}] Sampling client created: {sampling_client_id}")

        return {
            "sampling_client_id": sampling_client_id,
            "model_path": resolved_model_path,
            "status": "ready"
        }

    async def _nemo_rl_sample(
        self,
        request_id: str,
        handle: Any,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]],
        prompt_logprobs: bool,
    ) -> Dict[str, Any]:
        """
        Sample via NeMo RL Policy.generate() using vLLM Ray workers.

        PERF-002: Requests are accumulated by the batch accumulator and flushed
        as a single generate() call. This gives ~20-30x speedup for RL sampling
        (2048 individual requests → 1 batched call).
        """
        if handle.policy_generation is None:
            raise RuntimeError(
                "NeMo RL generation engine not initialized (debug_train_only mode?)"
            )

        return await self._nemo_rl_batch_accumulator.submit(
            handle=handle,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params or {},
            prompt_logprobs=prompt_logprobs,
        )
