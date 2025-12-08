"""
Tinker-Slime Data Format Converter

Converts between Tinker API data formats and Slime rollout data formats.
Handles both RL (PPO/GRPO) and SFT training modes.
"""
import logging
import torch
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TinkerDataConverter:
    """
    Convert between Tinker API format and Slime rollout format.

    Tinker API uses nested dict structures with model_input/loss_fn_inputs.
    Slime expects rollout_data with torch tensors for tokens, masks, etc.
    """

    @staticmethod
    def _get_field(obj: Any, field: str) -> Any:
        """Get field from either dict or Pydantic model."""
        if hasattr(obj, field):
            return getattr(obj, field)
        elif isinstance(obj, dict):
            return obj.get(field)
        return None

    @staticmethod
    def extract_tokens_from_model_input(model_input: Any) -> List[int]:
        """
        Extract token list from flexible model_input format.

        Supports multiple formats:
        - {"chunks": [{"tokens": [1,2,3], "type": "encoded_text"}]}
        - {"tokens": [1,2,3]}
        - {"input_ids": [1,2,3]}

        Works with both dict and Pydantic model inputs.
        """
        # Try chunks first
        chunks = TinkerDataConverter._get_field(model_input, "chunks")
        if chunks:
            if not chunks:
                raise ValueError("Empty chunks in model_input")
            first_chunk = chunks[0]
            return TinkerDataConverter._get_field(first_chunk, "tokens")

        # Try tokens
        tokens = TinkerDataConverter._get_field(model_input, "tokens")
        if tokens is not None:
            return tokens

        # Try input_ids
        input_ids = TinkerDataConverter._get_field(model_input, "input_ids")
        if input_ids is not None:
            return input_ids

        raise ValueError(f"Unknown model_input format")

    @staticmethod
    def extract_tensor_data(tensor_dict: Any) -> List[Any]:
        """
        Extract data from Tinker tensor format.

        Format: {"data": [1,2,3], "shape": [3], "dtype": "int64"}
        Returns just the data list.
        Works with both dict and Pydantic model inputs.
        """
        data = TinkerDataConverter._get_field(tensor_dict, "data")
        return data if data is not None else tensor_dict

    @classmethod
    def forward_to_rollout(cls, data: List[Any]) -> Dict[str, Any]:
        """
        Convert Tinker forward data to Slime rollout_data format.

        Args:
            data: List of forward data samples (dicts or Pydantic models), each with:
                - model_input: {"chunks": [{"tokens": [...]}]}
                - loss_fn_inputs: {"target_tokens": {"data": [...]}, "mask": {"data": [...]}}

        Returns:
            Slime rollout_data dict with torch tensors
        """
        tokens_list = []
        loss_masks_list = []
        response_lengths_list = []

        for datum in data:
            # Extract input tokens
            model_input = cls._get_field(datum, "model_input")
            tokens = cls.extract_tokens_from_model_input(model_input)
            tokens_list.append(torch.tensor(tokens, dtype=torch.long))

            # Extract loss function inputs
            loss_fn_inputs = cls._get_field(datum, "loss_fn_inputs")

            # Get mask (optional)
            mask = cls._get_field(loss_fn_inputs, "mask")
            weights = cls._get_field(loss_fn_inputs, "weights")

            if mask is not None:
                mask_data = cls.extract_tensor_data(mask)
                loss_mask = torch.tensor(mask_data, dtype=torch.float32)
            elif weights is not None:
                weights_data = cls.extract_tensor_data(weights)
                loss_mask = torch.tensor(weights_data, dtype=torch.float32)
            else:
                # Default: all ones (no masking)
                loss_mask = torch.ones(len(tokens), dtype=torch.float32)

            loss_masks_list.append(loss_mask)

            # Response length is number of non-zero mask elements
            response_length = int(loss_mask.sum().item())
            response_lengths_list.append(response_length)
            print(f"[CONVERTER DEBUG SFT] Sample {len(loss_masks_list)-1}: loss_mask sum={response_length}, len={len(loss_mask)}", flush=True)

        # Build rollout_data with dummy RL fields (not used for forward-only)
        batch_size = len(data)
        max_len = max(len(t) for t in tokens_list)

        rollout_data = {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "response_lengths": response_lengths_list,
            # Dummy fields for compatibility (not used in forward_only)
            "advantages": [torch.zeros(max_len, dtype=torch.float32) for _ in range(batch_size)],
            "log_probs": [torch.zeros(max_len, dtype=torch.float32) for _ in range(batch_size)],
            "ref_log_probs": [torch.zeros(max_len, dtype=torch.float32) for _ in range(batch_size)],
            "values": [torch.zeros(max_len, dtype=torch.float32) for _ in range(batch_size)],
            "returns": [torch.zeros(max_len, dtype=torch.float32) for _ in range(batch_size)],
        }

        logger.debug(f"Converted {batch_size} forward samples to rollout_data")
        return rollout_data

    @classmethod
    def forward_backward_to_rollout(
        cls,
        data: List[Any],
        is_rl: bool = False
    ) -> Dict[str, Any]:
        """
        Convert Tinker forward_backward data to Slime rollout_data format.

        Args:
            data: List of training data samples (dicts or Pydantic models)
            is_rl: True for RL training (PPO/GRPO), False for SFT

        Returns:
            Slime rollout_data dict with torch tensors
        """
        # Auto-detect data format based on fields present
        # DPO's forward_backward_custom sends SFT-like data (target_tokens + weights)
        # even when the model was created for RL training
        if data and len(data) > 0:
            first_datum = data[0]
            loss_fn_inputs = cls._get_field(first_datum, "loss_fn_inputs")
            if loss_fn_inputs:
                has_advantages = cls._get_field(loss_fn_inputs, "advantages") is not None
                has_logprobs = cls._get_field(loss_fn_inputs, "logprobs") is not None
                has_weights = cls._get_field(loss_fn_inputs, "weights") is not None or cls._get_field(loss_fn_inputs, "weight") is not None
                has_target = cls._get_field(loss_fn_inputs, "target_tokens") is not None or cls._get_field(loss_fn_inputs, "target") is not None

                # If we have advantages or logprobs, it's RL data
                # If we have weights+target but no logprobs, it's SFT data (including DPO backward pass)
                detected_is_rl = has_advantages or has_logprobs
                if detected_is_rl != is_rl:
                    print(f"[CONVERTER] Auto-detected data format: is_rl={detected_is_rl} (was {is_rl}), "
                          f"has_advantages={has_advantages}, has_logprobs={has_logprobs}, "
                          f"has_weights={has_weights}, has_target={has_target}", flush=True)
                    is_rl = detected_is_rl

        print(f"[CONVERTER DEBUG SFT] forward_backward_to_rollout called with {len(data)} samples, is_rl={is_rl}", flush=True)
        tokens_list = []
        loss_masks_list = []
        response_lengths_list = []

        # RL-specific fields
        advantages_list = []
        log_probs_list = []
        ref_log_probs_list = [] if is_rl else None
        values_list = [] if is_rl else None
        returns_list = [] if is_rl else None

        print(f"[CONVERTER] Converting {len(data)} forward_backward samples (is_rl={is_rl})", flush=True)
        logger.info(f"Converting {len(data)} forward_backward samples (is_rl={is_rl})")

        # Handle legacy HTTP test format: empty data or [{"input": "...", "target": "..."}]
        # This is for backward compatibility with test_4_multi_step_training.py
        if not data or len(data) == 0:
            logger.warning(f"[CONVERTER] Empty data provided - cannot generate fake test data without args")
            # Return minimal rollout_data that will fail validation
            return {
                "tokens": [],
                "loss_masks": [],
                "response_lengths": [],
                "advantages": [] if is_rl else None,
                "log_probs": [] if is_rl else None,
                "ref_log_probs": [] if is_rl else None,
                "values": [] if is_rl else None,
                "returns": [] if is_rl else None
            }

        for idx, datum in enumerate(data):
            print(f"[CONVERTER] Processing datum {idx}, type={type(datum)}", flush=True)
            # Extract input tokens
            model_input = cls._get_field(datum, "model_input")
            print(f"[CONVERTER] model_input type={type(model_input)}, value={model_input}", flush=True)
            tokens = cls.extract_tokens_from_model_input(model_input)
            print(f"[CONVERTER] Extracted {len(tokens)} input tokens: {tokens}", flush=True)
            logger.debug(f"Extracted {len(tokens)} input tokens: {tokens}")
            tokens_list.append(torch.tensor(tokens, dtype=torch.long))

            # Extract loss function inputs
            loss_fn_inputs = cls._get_field(datum, "loss_fn_inputs")

            if is_rl:
                # RL mode: Extract logprobs, mask, advantages, values, returns
                # CRITICAL: response_length must match the lengths of loss_mask and all RL fields.
                # Miles splits entropy/logprob tensors using response_lengths, so mismatches cause
                # RuntimeError in get_sum_of_sample_mean(): split_with_sizes expects sum to match tensor size.

                # Step 1: Determine response_length from mask or logprobs
                # For RL, the mask determines which tokens to compute loss on (the response portion).
                # If mask is not provided (tinker-cookbook removes mask via remove_mask()), we infer
                # the response length from logprobs, which represents per-response-token values.
                logprobs = cls._get_field(loss_fn_inputs, "logprobs")
                logprobs_data = cls.extract_tensor_data(logprobs) if logprobs is not None else None

                mask = cls._get_field(loss_fn_inputs, "mask")
                if mask is not None:
                    mask_data = cls.extract_tensor_data(mask)
                    loss_mask = torch.tensor(mask_data, dtype=torch.float32)
                    response_len = len(mask_data)
                elif logprobs_data is not None:
                    # No mask provided but logprobs available - use logprobs length as response length
                    # This is the correct interpretation: logprobs covers only the response tokens
                    response_len = len(logprobs_data)
                    loss_mask = torch.ones(response_len, dtype=torch.float32)
                    print(f"[CONVERTER RL] Sample {idx}: No mask, using logprobs length={response_len} for response_length (tokens={len(tokens)})", flush=True)
                else:
                    # Fallback: no mask and no logprobs - use full token length
                    response_len = len(tokens)
                    loss_mask = torch.ones(response_len, dtype=torch.float32)
                    print(f"[CONVERTER RL] Sample {idx}: No mask and no logprobs, using full token length={response_len} for response_length", flush=True)

                # Step 2: Process logprobs - must match response_len
                # Handle None values in logprobs (SGLang returns None for first token)
                if logprobs_data is None:
                    logprobs_data_clean = [0.0] * response_len
                else:
                    logprobs_data_clean = [
                        0.0 if lp is None else float(lp) for lp in logprobs_data
                    ]
                log_probs_list.append(torch.tensor(logprobs_data_clean, dtype=torch.float32))

                # Step 3: Process advantages - must match response_len
                # Note: DPO doesn't use advantages - it's optional for preference methods
                advantages = cls._get_field(loss_fn_inputs, "advantages")
                if advantages is not None:
                    advantages_data = cls.extract_tensor_data(advantages)
                    advantages_list.append(torch.tensor(advantages_data, dtype=torch.float32))
                else:
                    # Create zeros with matching response_len (not len(tokens))
                    advantages_list.append(torch.zeros(response_len, dtype=torch.float32))

                # Step 4: Process ref_log_probs - must match response_len
                ref_logprobs = cls._get_field(loss_fn_inputs, "ref_logprobs")
                if ref_logprobs is not None:
                    ref_logprobs_data = cls.extract_tensor_data(ref_logprobs)
                    ref_log_probs_list.append(torch.tensor(ref_logprobs_data, dtype=torch.float32))
                else:
                    ref_log_probs_list.append(torch.zeros(response_len, dtype=torch.float32))

                # Step 5: Process values - must match response_len
                values = cls._get_field(loss_fn_inputs, "values")
                if values is not None:
                    values_data = cls.extract_tensor_data(values)
                    values_list.append(torch.tensor(values_data, dtype=torch.float32))
                else:
                    values_list.append(torch.zeros(response_len, dtype=torch.float32))

                # Step 6: Process returns - must match response_len
                returns = cls._get_field(loss_fn_inputs, "returns")
                if returns is not None:
                    returns_data = cls.extract_tensor_data(returns)
                    returns_list.append(torch.tensor(returns_data, dtype=torch.float32))
                else:
                    returns_list.append(torch.zeros(response_len, dtype=torch.float32))

            else:
                # SFT mode: Extract target and weights
                # Look for "target_tokens" or "target" field
                target = cls._get_field(loss_fn_inputs, "target_tokens")
                if target is None:
                    target = cls._get_field(loss_fn_inputs, "target")

                weights = cls._get_field(loss_fn_inputs, "weights")
                if weights is None:
                    weights = cls._get_field(loss_fn_inputs, "weight")

                if not weights or not target:
                    raise ValueError("SFT loss_fn_inputs must contain weights and target_tokens/target")

                # Parse TensorData format
                weights_data = cls.extract_tensor_data(weights)
                target_data = cls.extract_tensor_data(target)

                print(f"[CONVERTER DEBUG SFT] Sample {idx}: weights_data length = {len(weights_data)}, target_data length = {len(target_data)}", flush=True)

                # Build full token sequence by appending last target token
                # Input: [1,2,3,4,5], Target: [2,3,4,5,6]
                # Full tokens: [1,2,3,4,5,6] (input + last_target)
                input_tokens_tensor = torch.tensor(tokens, dtype=torch.long)
                target_tensor = torch.tensor(target_data, dtype=torch.long)
                full_tokens = torch.cat([input_tokens_tensor, target_tensor[-1:]], dim=0)

                print(f"[CONVERTER DEBUG SFT] Sample {idx}: full_tokens length = {len(full_tokens)}", flush=True)

                # Update tokens list with full sequence
                tokens_list[-1] = full_tokens  # Replace the one we added earlier

                # Loss mask and response length
                loss_mask = torch.tensor(weights_data, dtype=torch.float32)
                print(f"[CONVERTER DEBUG SFT] Sample {idx}: loss_mask length = {len(loss_mask)}, sum = {loss_mask.sum().item()}", flush=True)

                # Dummy RL fields
                advantages_list.append(torch.zeros(len(loss_mask), dtype=torch.float32))
                log_probs_list.append(torch.zeros(len(loss_mask), dtype=torch.float32))

            loss_masks_list.append(loss_mask)
            # For standard SFT: response_length = count of non-zero mask elements
            # For DPO weighted loss: weights can be negative, so use mask LENGTH not sum
            # Miles uses response_length to slice log_probs, so it must equal mask length
            response_length = len(loss_mask)

            # CRITICAL FIX: Handle causal LM shift when response_length == total_length
            #
            # When the entire sequence is treated as "response" (no separate prompt),
            # Miles' get_responses() in loss.py returns N-1 elements because:
            #   - logits[i] predicts tokens[i+1] (causal LM)
            #   - First token has no logit predicting it
            #   - So entropy/log_probs tensors have N-1 elements
            #
            # If we pass response_length=N, sum_of_sample_mean() will try to split
            # an (N-1) tensor using response_lengths that sum to N, causing:
            #   RuntimeError: split_with_sizes expects split_sizes to sum exactly to X
            #
            # Fix: When response_length == token_length, adjust to N-1 and trim all
            # per-token tensors to match.
            # See: miles/backends/megatron_utils/loss.py lines 100-108
            token_length = len(tokens_list[-1])
            if is_rl and response_length == token_length and token_length > 1:
                response_length = token_length - 1
                # Trim loss_mask to match (skip first token)
                loss_masks_list[-1] = loss_mask[1:]
                # Trim all RL per-token tensors to match
                if advantages_list:
                    advantages_list[-1] = advantages_list[-1][1:]
                if log_probs_list:
                    log_probs_list[-1] = log_probs_list[-1][1:]
                if ref_log_probs_list:
                    ref_log_probs_list[-1] = ref_log_probs_list[-1][1:]
                if values_list:
                    values_list[-1] = values_list[-1][1:]
                if returns_list:
                    returns_list[-1] = returns_list[-1][1:]
                print(f"[CONVERTER RL] Sample {idx}: Adjusted response_length from {token_length} to {response_length} (N-1 for causal LM shift)", flush=True)

            response_lengths_list.append(response_length)

        # Build rollout_data
        rollout_data = {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "response_lengths": response_lengths_list,
            "advantages": advantages_list,
            "log_probs": log_probs_list,
        }

        if is_rl:
            rollout_data["ref_log_probs"] = ref_log_probs_list
            rollout_data["values"] = values_list
            rollout_data["returns"] = returns_list
        else:
            # SFT-like data detected (including DPO backward pass)
            # Override loss type to use sft_loss instead of policy_loss
            # NOTE: Stored as scalar metadata (like _actual_global_batch_size),
            # accessed directly from rollout_data in Miles get_batch()
            rollout_data["_loss_type_override"] = "sft_loss"

        logger.info(f"Converted {len(data)} {'RL' if is_rl else 'SFT'} samples to rollout_data with {len(tokens_list)} token sequences")
        logger.info(f"rollout_data keys: {list(rollout_data.keys())}")
        if "_loss_type_override" in rollout_data:
            logger.info(f"_loss_type_override set to: {rollout_data['_loss_type_override']}")
        return rollout_data

    @staticmethod
    def _extract_response_lengths_from_original(original_data: Optional[List[Any]]) -> List[int]:
        """Extract response lengths (mask lengths) from original Tinker payload.

        For RL training, tinker-cookbook removes the 'mask' field before sending to the API
        (see remove_mask() in train.py). In this case, we use the 'logprobs' field length
        as the response length since both mask and logprobs have the same length (per-response-token).
        """
        response_lengths_list: List[int] = []
        if not original_data:
            return response_lengths_list

        for idx, datum in enumerate(original_data):
            loss_fn_inputs = (
                datum.get("loss_fn_inputs")
                if isinstance(datum, dict)
                else getattr(datum, "loss_fn_inputs", None)
            )
            if not loss_fn_inputs:
                response_lengths_list.append(0)
                continue

            # Try to get response length from weights/weight/mask first (SFT path)
            weights_data = None
            if isinstance(loss_fn_inputs, dict):
                weights_data = loss_fn_inputs.get("weights") or loss_fn_inputs.get("weight") or loss_fn_inputs.get("mask")
            else:
                weights_data = (
                    getattr(loss_fn_inputs, "weights", None)
                    or getattr(loss_fn_inputs, "weight", None)
                    or getattr(loss_fn_inputs, "mask", None)
                )

            if weights_data is not None:
                weights = TinkerDataConverter.extract_tensor_data(weights_data)
                response_lengths_list.append(len(weights))
                print(f"[CONVERTER DEBUG] Original weights[{idx}] length = {len(weights)}", flush=True)
                continue

            # RL path: mask is removed by tinker-cookbook's remove_mask()
            # Use logprobs field length instead (same length as mask - per-response-token)
            logprobs_data = None
            if isinstance(loss_fn_inputs, dict):
                logprobs_data = loss_fn_inputs.get("logprobs")
            else:
                logprobs_data = getattr(loss_fn_inputs, "logprobs", None)

            if logprobs_data is not None:
                logprobs = TinkerDataConverter.extract_tensor_data(logprobs_data)
                response_lengths_list.append(len(logprobs))
                print(f"[CONVERTER DEBUG] Original logprobs[{idx}] length = {len(logprobs)} (using as response_length)", flush=True)
                continue

            # No length info available
            response_lengths_list.append(0)
            print(f"[CONVERTER DEBUG] Sample[{idx}] no weights/mask/logprobs found, using 0", flush=True)

        return response_lengths_list

    @staticmethod
    def rollout_to_forward_result(
        results: List[Dict[str, Any]],
        loss_fn: str = "cross_entropy",
        rollout_data: Optional[Dict[str, Any]] = None,
        original_data: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert Slime forward_only results to Tinker forward result format.

        Args:
            results: List of results from Slime (per-GPU results)
            loss_fn: Loss function type

        Returns:
            Tinker forward result dict
        """
        loss_fn_outputs: List[Dict[str, Any]] = []
        all_logprobs: List[float] = []
        total_loss = 0.0
        response_lengths_list: List[int] = TinkerDataConverter._extract_response_lengths_from_original(original_data)
        if not response_lengths_list and rollout_data and rollout_data.get("response_lengths"):
            response_lengths_list = [int(length) for length in rollout_data.get("response_lengths", [])]

        # Miles returns one result per (data-parallel, pipeline) shard.
        # Only the pipeline-last shard includes log_probs, and it nests them
        # under result["loss"]["log_probs"] as a list of per-sample tensors.
        sample_index = 0

        for result in results:
            loss_dict = result.get("loss") or {}
            raw_logprobs = []
            loss_value = 0.0

            if isinstance(loss_dict, dict):
                raw_logprobs = loss_dict.get("log_probs") or []
                loss_value = TinkerDataConverter._extract_scalar_loss(loss_dict.get("loss", 0.0))

            # Fallback for legacy payloads where logprobs sit at the top level
            if not raw_logprobs:
                raw_logprobs = result.get("logprobs") or result.get("log_probs") or []

            if not isinstance(raw_logprobs, list):
                raw_logprobs = []

            for tensor in raw_logprobs:
                if hasattr(tensor, "cpu"):
                    logprob_list = tensor.cpu().tolist()
                elif isinstance(tensor, (list, tuple)):
                    logprob_list = list(tensor)
                else:
                    logprob_list = [float(tensor)]

                original_logprobs_length = len(logprob_list)

                # Trim/pad to match response mask length if available
                if sample_index < len(response_lengths_list):
                    response_length = response_lengths_list[sample_index]
                    if response_length > 0:
                        logprobs_length = len(logprob_list)
                        if logprobs_length > response_length:
                            logprob_list = logprob_list[-response_length:]
                            print(f"[CONVERTER DEBUG] Sample {sample_index}: trimmed logprobs from {original_logprobs_length} to {response_length}", flush=True)
                        elif logprobs_length < response_length:
                            padding = [0.0] * (response_length - logprobs_length)
                            logprob_list = padding + logprob_list
                            print(f"[CONVERTER DEBUG] Sample {sample_index}: padded logprobs from {original_logprobs_length} to {response_length}", flush=True)
                        else:
                            print(f"[CONVERTER DEBUG] Sample {sample_index}: logprobs {original_logprobs_length} matches response_length {response_length}", flush=True)
                else:
                    print(f"[CONVERTER DEBUG] Sample {sample_index}: no response_length, using raw logprobs length {original_logprobs_length}", flush=True)

                # Use logprobs as-is - no [0.0] prepend needed here
                # The [0.0] prepend for DPO [1:] slice compensation is done in sglang_client.py
                # for the compute_logprobs_async path (reference model), not here
                payload_logprobs = logprob_list
                print(f"[CONVERTER DEBUG] Sample {sample_index}: final payload_logprobs length = {len(payload_logprobs)}", flush=True)

                loss_fn_outputs.append({
                    "loss": {
                        "data": [loss_value],
                        "shape": [1],
                        "dtype": "float32"
                    },
                    "logprobs": {
                        "data": payload_logprobs,
                        "shape": [len(payload_logprobs)],
                        "dtype": "float32"
                    }
                })
                all_logprobs.extend(logprob_list)

                total_loss += loss_value
                sample_index += 1

        if not loss_fn_outputs:
            logger.warning("forward_only returned no log_probs; loss_fn_outputs will be empty")

        metrics_dict = {
            "total_loss:sum": float(total_loss),
            "grad_norm:mean": 0.0,
            "num_tokens:sum": float(len(all_logprobs)),
        }

        return {
            "type": "forward",
            "loss_fn_output_type": loss_fn,
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics_dict,
            "logprobs": {
                "data": all_logprobs,
                "shape": [len(all_logprobs)],
                "dtype": "float32"
            }
        }

    @staticmethod
    def rollout_to_forward_backward_result(
        results: List[Dict[str, Any]],
        loss_fn: str = "cross_entropy",
        rollout_data: Optional[Dict[str, Any]] = None,
        original_data: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert Slime forward_backward_only results to Tinker result format.

        Args:
            results: List of results from Slime (per-GPU results)
            loss_fn: Loss function type
            rollout_data: Optional rollout_data dict with tokens, loss_masks, response_lengths

        Returns:
            Tinker forward_backward result dict
        """
        # Find the result with actual loss values (pipeline last stage)
        # With DP>1, only one rank (pipeline last stage) has populated loss_dict
        result_with_loss = None
        for result in results:
            if result.get("loss"):  # Non-empty loss dict
                result_with_loss = result
                break

        # Fallback to first result if none have loss
        if result_with_loss is None:
            result_with_loss = results[0] if results else {}

        loss_dict = result_with_loss.get("loss", {})
        grad_norm = result_with_loss.get("grad_norm", 0.0)

        # Extract per-sample logprobs from loss_dict
        # Slime returns this as "log_probs" (with underscore) containing list of tensors
        per_sample_logprobs = loss_dict.get("log_probs", None)
        print(f"[CONVERTER DEBUG] per_sample_logprobs type: {type(per_sample_logprobs)}, is None: {per_sample_logprobs is None}", flush=True)
        if per_sample_logprobs:
            print(f"[CONVERTER DEBUG] per_sample_logprobs length: {len(per_sample_logprobs)}", flush=True)

        # Determine batch_size from original_data (actual samples from Tinker)
        # Don't use rollout_data as it may be padded for data parallel
        batch_size = 0
        if original_data:
            batch_size = len(original_data)
        elif per_sample_logprobs:
            batch_size = len(per_sample_logprobs)
        elif rollout_data and "tokens" in rollout_data:
            # Fallback to rollout_data if no original_data
            batch_size = len(rollout_data["tokens"])
        else:
            # Fallback: assume batch_size=1
            batch_size = 1
            logger.warning("Could not determine batch_size - defaulting to 1")

        print(f"[CONVERTER DEBUG] batch_size: {batch_size}", flush=True)

        # Get tokens for fallback
        tokens_list = rollout_data.get("tokens", []) if rollout_data else []

        # Extract response_lengths from ORIGINAL Tinker data (before Slime padding)
        response_lengths_list = TinkerDataConverter._extract_response_lengths_from_original(original_data)
        if not response_lengths_list and rollout_data and rollout_data.get("response_lengths"):
            response_lengths_list = [int(length) for length in rollout_data.get("response_lengths", [])]
        if not response_lengths_list:
            print(f"[CONVERTER DEBUG] No mask lengths available; using zero-length defaults", flush=True)

        print(f"[CONVERTER DEBUG] response_lengths_list: {response_lengths_list}", flush=True)

        # Build per-sample loss_fn_outputs
        loss_fn_outputs = []
        total_loss = TinkerDataConverter._extract_scalar_loss(loss_dict.get("loss", 0.0))

        for i in range(batch_size):
            output_entry = {
                "loss": {
                    "data": [total_loss],
                    "shape": [1],
                    "dtype": "float32"
                }
            }

            # Add per-sample logprobs if available
            if per_sample_logprobs and i < len(per_sample_logprobs):
                print(f"[CONVERTER DEBUG] Processing sample {i}", flush=True)
                # Convert tensor to list for JSON serialization
                logprobs_tensor = per_sample_logprobs[i]
                if hasattr(logprobs_tensor, 'cpu'):
                    response_logprobs = logprobs_tensor.cpu().tolist()
                else:
                    response_logprobs = logprobs_tensor if isinstance(logprobs_tensor, list) else list(logprobs_tensor)

                print(f"[CONVERTER DEBUG] Sample {i}: response_logprobs length = {len(response_logprobs)}", flush=True)

                # Extract RESPONSE LENGTH (non-zero mask count) for proper trimming
                # Slime returns full sequence logprobs, but Tinker expects RESPONSE portion only
                # Use response_lengths (count of non-zero mask elements), NOT len(loss_mask)
                if i < len(response_lengths_list):
                    response_length = response_lengths_list[i]  # Number of non-zero mask elements
                    logprobs_length = len(response_logprobs)
                    print(f"[CONVERTER DEBUG] Sample {i}: response_length = {response_length}, logprobs_length = {logprobs_length}", flush=True)

                    if logprobs_length > response_length:
                        # Slime returned full sequence logprobs - extract only the LAST response_length elements
                        final_logprobs = response_logprobs[-response_length:]
                        print(f"[CONVERTER DEBUG] Sample {i}: Trimmed from {logprobs_length} to {len(final_logprobs)}", flush=True)
                    elif logprobs_length == response_length:
                        # Slime returned exactly the response portion - use as-is
                        final_logprobs = response_logprobs
                        print(f"[CONVERTER DEBUG] Sample {i}: Using as-is (lengths match)", flush=True)
                    else:
                        # Slime returned fewer logprobs than response - pad with zeros
                        padding_length = response_length - logprobs_length
                        final_logprobs = [0.0] * padding_length + response_logprobs
                        print(f"[CONVERTER DEBUG] Sample {i}: Padded from {logprobs_length} to {len(final_logprobs)}", flush=True)
                else:
                    # No mask info - use logprobs as-is
                    final_logprobs = response_logprobs
                    print(f"[CONVERTER DEBUG] Sample {i}: No mask info, using logprobs as-is", flush=True)

                output_entry["logprobs"] = {
                    "data": final_logprobs,
                    "shape": [len(final_logprobs)],
                    "dtype": "float32"
                }
                print(f"[CONVERTER DEBUG] Sample {i}: FINAL logprobs length in output_entry = {len(final_logprobs)}", flush=True)
            else:
                print(f"[CONVERTER DEBUG] Sample {i}: No logprobs available", flush=True)
                # Fallback: return zeros padded to RESPONSE length (not full sequence)
                # Tinker-cookbook expects logprobs to match response length (non-zero mask count)
                response_length = response_lengths_list[i] if i < len(response_lengths_list) else 0
                print(f"[CONVERTER DEBUG] Sample {i}: Using zero-padding with response_length {response_length}", flush=True)
                output_entry["logprobs"] = {
                    "data": [0.0] * response_length,
                    "shape": [response_length],
                    "dtype": "float32"
                }

            loss_fn_outputs.append(output_entry)

        # Calculate total number of tokens (sum of all response lengths)
        total_num_tokens = sum(response_lengths_list) if response_lengths_list else 0.0

        # Build metrics dict matching original API format
        metrics_dict = {
            "total_loss:sum": TinkerDataConverter._extract_scalar_loss(loss_dict.get("loss", 0.0)),
            "pg_loss:sum": TinkerDataConverter._extract_scalar_loss(loss_dict.get("pg_loss", 0.0)),
            "entropy_loss:sum": TinkerDataConverter._extract_scalar_loss(loss_dict.get("entropy_loss", 0.0)),
            "pg_clipfrac:mean": TinkerDataConverter._extract_scalar_loss(loss_dict.get("pg_clipfrac", 0.0)),
            "ppo_kl:sum": TinkerDataConverter._extract_scalar_loss(loss_dict.get("ppo_kl", 0.0)),
            "grad_norm:mean": TinkerDataConverter._extract_scalar_loss(grad_norm),
            "num_tokens:sum": float(total_num_tokens),
        }

        # Add optional metrics if present
        if "kl_loss" in loss_dict:
            metrics_dict["kl_loss:sum"] = TinkerDataConverter._extract_scalar_loss(loss_dict.get("kl_loss", 0.0))
        if "value_loss" in loss_dict:
            metrics_dict["value_loss:sum"] = TinkerDataConverter._extract_scalar_loss(loss_dict.get("value_loss", 0.0))
        if "value_clipfrac" in loss_dict:
            metrics_dict["value_clipfrac:mean"] = TinkerDataConverter._extract_scalar_loss(loss_dict.get("value_clipfrac", 0.0))

        # Build top-level logprobs (flattened across all samples)
        # Required by Tinker client for RL training
        all_logprobs = []
        for idx, output in enumerate(loss_fn_outputs):
            if output.get("logprobs") and output["logprobs"].get("data"):
                print(f"[CONVERTER DEBUG] loss_fn_outputs[{idx}] logprobs data length: {len(output['logprobs']['data'])}", flush=True)
                all_logprobs.extend(output["logprobs"]["data"])

        print(f"[CONVERTER DEBUG] Returning {len(loss_fn_outputs)} loss_fn_outputs, total logprobs: {len(all_logprobs)}", flush=True)

        return {
            "loss_fn_output_type": loss_fn,
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": metrics_dict,
            "logprobs": {
                "data": all_logprobs,
                "shape": [len(all_logprobs)],
                "dtype": "float32"
            } if all_logprobs else {
                "data": [],
                "shape": [0],
                "dtype": "float32"
            }
        }

    @staticmethod
    def _extract_scalar_loss(loss_entry: Any) -> float:
        """Extract a scalar float from various Miles/Slime loss formats."""
        if isinstance(loss_entry, (int, float)):
            return float(loss_entry)
        if hasattr(loss_entry, "item"):
            try:
                return float(loss_entry.item())
            except Exception:
                return 0.0
        if isinstance(loss_entry, dict):
            for key in ("loss", "pg_loss", "value_loss", "kl_loss", "entropy_loss"):
                val = loss_entry.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
                if hasattr(val, "item"):
                    try:
                        return float(val.item())
                    except Exception:
                        continue
            for val in loss_entry.values():
                if isinstance(val, (int, float)):
                    return float(val)
                if hasattr(val, "item"):
                    try:
                        return float(val.item())
                    except Exception:
                        continue
        return 0.0

    @staticmethod
    def _extract_logprob_list(logprobs_entry: Any) -> List[float]:
        """Normalize logprob outputs into a flat list of floats."""
        if not logprobs_entry:
            return []

        normalized: List[float] = []
        for item in logprobs_entry:
            value = None
            if isinstance(item, (int, float)):
                value = float(item)
            elif hasattr(item, "item"):
                try:
                    value = float(item.item())
                except Exception:
                    value = None
            elif isinstance(item, (list, tuple)) and item:
                first = item[0]
                if isinstance(first, (int, float)):
                    value = float(first)
                elif hasattr(first, "item"):
                    try:
                        value = float(first.item())
                    except Exception:
                        value = None
            if value is not None:
                normalized.append(value)
        return normalized
