"""
Sampling Service - Business Logic for Model Sampling

Handles:
- Async sampling via SGLang (Miles backend)
- Async sampling via Policy.generate() (NeMo RL backend)
- Sync sampling via SGLang
- SGLang sampling client creation
"""
import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional

from ..utils.sglang_client import SGLangClient
from ..utils.helpers import find_model_with_rollout_manager

logger = logging.getLogger(__name__)


class SamplingService:
    """Service for managing model sampling via SGLang."""

    def __init__(self):
        """Initialize SamplingService."""
        # Lock to serialize NeMo RL generation requests.
        # The offload→generate→refit cycle is stateful and cannot safely
        # interleave across concurrent requests (BUG-002 root cause).
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

        Bridges the TinkerCloud sampling API to NeMo RL's batch generation.
        Each call generates num_samples completions for a single prompt.

        The offload→generate→refit cycle is serialized via asyncio.Lock to
        prevent concurrent requests from corrupting shared Policy state
        (BUG-002 root cause).
        """
        import torch

        sampling_params = sampling_params or {}
        max_new_tokens = sampling_params.get("max_tokens", 256)
        temperature = sampling_params.get("temperature", 0.7)
        top_p = sampling_params.get("top_p", 0.9)
        greedy = temperature <= 0.01

        # Extract stop sequences from sampling_params
        # Tinker API sends stop as str | list[str] | list[int] | None
        raw_stop = sampling_params.get("stop")
        stop_strings: List[str] = []
        if raw_stop is not None:
            if isinstance(raw_stop, str):
                stop_strings = [raw_stop]
            elif isinstance(raw_stop, list) and raw_stop and isinstance(raw_stop[0], str):
                stop_strings = list(raw_stop)
            # list[int] stop token IDs are handled by the generation config's stop_token_ids

        policy = handle.policy
        policy_generation = handle.policy_generation

        if policy_generation is None:
            raise RuntimeError(
                "NeMo RL generation engine not initialized (debug_train_only mode?)"
            )

        # Build BatchedDataDict for VllmGeneration.generate()
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        input_ids_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
        input_lengths_tensor = torch.tensor([len(prompt_tokens)], dtype=torch.long)

        # Repeat for num_samples
        if num_samples > 1:
            input_ids_tensor = input_ids_tensor.repeat(num_samples, 1)
            input_lengths_tensor = input_lengths_tensor.repeat(num_samples)

        data = BatchedDataDict({
            "input_ids": input_ids_tensor,
            "input_lengths": input_lengths_tensor,
        })

        # Pad batch to dp_size if needed (VllmGeneration shards by dp_size)
        dp_size = handle.config.get("dp_size", 1)
        actual_size = num_samples
        batch_size = actual_size
        if actual_size % dp_size != 0:
            pad_count = dp_size - (actual_size % dp_size)
            batch_size = actual_size + pad_count
            data["input_ids"] = torch.cat([
                data["input_ids"],
                data["input_ids"][-1:].repeat(pad_count, 1),
            ])
            data["input_lengths"] = torch.cat([
                data["input_lengths"],
                data["input_lengths"][-1:].repeat(pad_count),
            ])

        # Pass stop strings into the data dict so vLLM workers can stop
        # generation natively. Workers read data.get("stop_strings") and
        # merge them into vLLM SamplingParams via _merge_stop_strings().
        # Must match batch dimension for shard_by_batch_size().
        if stop_strings:
            data["stop_strings"] = [stop_strings] * batch_size

        # Forward per-request sampling params via BatchedDataDict so vLLM
        # workers use them instead of static config defaults (FR-002).
        # Prefixed with _tinker_ to avoid collisions with NeMo RL internal keys.
        data["_tinker_max_new_tokens"] = [max_new_tokens] * batch_size
        data["_tinker_temperature"] = [temperature] * batch_size
        data["_tinker_top_p"] = [top_p] * batch_size

        # Serialize the offload→generate→refit cycle (BUG-002 fix).
        # Concurrent requests interleaving this stateful sequence caused
        # stop strings to be intermittently ignored.
        async with self._nemo_rl_generate_lock:
            await asyncio.to_thread(policy.offload_before_refit)
            await asyncio.to_thread(
                policy_generation.prepare_for_generation,
                tags=["weights", "kv_cache"],
            )

            try:
                result = await asyncio.to_thread(policy_generation.generate, data, greedy)
            finally:
                await asyncio.to_thread(
                    policy_generation.prepare_for_generation, tags=[],
                )
                await asyncio.to_thread(policy.offload_after_refit)

        # Convert result to TinkerCloud format
        output_ids = result["output_ids"]       # [B, padded_input_len + max_gen_len]
        gen_lengths = result["generation_lengths"]  # [B]
        logprobs_tensor = result["logprobs"]    # [B, padded_input_len + max_gen_len]

        sequences = []
        prompt_logprobs_result = None
        prompt_len = len(prompt_tokens)
        tokenizer = handle.tokenizer
        eos_id = tokenizer.eos_token_id if tokenizer else None

        for i in range(actual_size):
            gen_len = int(gen_lengths[i].item())
            out_tokens = output_ids[i, prompt_len:prompt_len + gen_len].tolist()
            # Logprobs are placed at [prompt_len, prompt_len + gen_len) by vllm_worker
            out_logprobs = logprobs_tensor[i, prompt_len:prompt_len + gen_len].tolist()
            text = tokenizer.decode(out_tokens) if tokenizer else None

            # Determine stop_reason: check EOS token, then stop strings in text
            stop_reason = "length"
            if out_tokens and eos_id is not None and out_tokens[-1] == eos_id:
                stop_reason = "stop"
            elif text and stop_strings:
                for ss in stop_strings:
                    if ss in text:
                        stop_reason = "stop"
                        break

            # Safety net: truncate output at first stop string occurrence.
            # This prevents recipe crashes when vLLM misses a stop string.
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
                        orig_token_count = len(out_tokens)
                        # Find truncation point by incremental decode to avoid
                        # tokenizer round-trip mismatch (encode != decode inverse).
                        trunc_count = orig_token_count
                        for t in range(1, orig_token_count + 1):
                            decoded = tokenizer.decode(
                                out_tokens[:t], skip_special_tokens=False,
                            )
                            if len(decoded) >= len(truncated_text):
                                trunc_count = t
                                break
                        out_tokens = out_tokens[:trunc_count]
                        out_logprobs = out_logprobs[:trunc_count]
                        text = tokenizer.decode(out_tokens)
                        stop_reason = "stop"
                        logger.warning(
                            f"[{request_id}] !!!!! STOP STRING SAFETY NET TRIGGERED: "
                            f"sequence {i} truncated from {orig_token_count} to "
                            f"{len(out_tokens)} tokens at stop string "
                            f"'{matched_stop}' !!!!!"
                        )

            sequences.append({
                "tokens": out_tokens,
                "logprobs": out_logprobs,
                "text": text,
                "stop_reason": stop_reason,
            })

        logger.info(
            f"[{request_id}] NeMo RL generated {len(sequences)} samples "
            f"(avg {sum(len(s['tokens']) for s in sequences)/max(len(sequences),1):.0f} tokens)"
        )

        return {
            "sequences": sequences,
            "prompt_logprobs": prompt_logprobs_result,
        }
