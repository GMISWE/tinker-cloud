"""
Tinker-Slime Data Format Converter

Converts between Tinker API data formats and Slime rollout data formats.
Handles both RL (PPO/GRPO) and SFT training modes.
"""
import logging
import torch
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TinkerDataConverter:
    """
    Convert between Tinker API format and Slime rollout format.

    Tinker API uses nested dict structures with model_input/loss_fn_inputs.
    Slime expects rollout_data with torch tensors for tokens, masks, etc.
    """

    @staticmethod
    def _get_field(obj: Any, field: str) -> Any:
        """Get field from either dict or Pydantic model. Dict lookup FIRST:
        hasattr-first returns bound methods for key names that collide with
        dict methods (e.g. "values" -> dict.values, hit by the RL path)."""
        if isinstance(obj, dict):
            return obj.get(field)
        if hasattr(obj, field):
            return getattr(obj, field)
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
        # Try chunks first. A ModelInput may carry MULTIPLE chunks (the SDK
        # splits inputs); concatenate them all — taking only chunks[0]
        # silently truncates the sample (e.g. 4-token tokens vs 438-token
        # weights, crashing the miles loss on shape mismatch).
        chunks = TinkerDataConverter._get_field(model_input, "chunks")
        if chunks:
            tokens: List[int] = []
            for chunk in chunks:
                chunk_tokens = TinkerDataConverter._get_field(chunk, "tokens")
                if chunk_tokens is None:
                    raise ValueError(
                        "model_input chunk without tokens (non-text chunks are "
                        "not supported by the miles backend)"
                    )
                tokens.extend(chunk_tokens)
            return tokens

        # Try tokens
        tokens = TinkerDataConverter._get_field(model_input, "tokens")
        if tokens is not None:
            return tokens

        # Try input_ids
        input_ids = TinkerDataConverter._get_field(model_input, "input_ids")
        if input_ids is not None:
            return input_ids

        raise ValueError("Unknown model_input format")

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
        loss_weights_list = []
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

            # loss_mask is Miles' 0/1 response region; client per-token weights
            # travel beside it as loss_weights (the loss multiplies them in).
            loss_weights = None
            if weights is not None:
                loss_weights = torch.tensor(cls.extract_tensor_data(weights), dtype=torch.float32)
            if mask is not None:
                loss_mask = torch.tensor(cls.extract_tensor_data(mask), dtype=torch.float32)
            elif loss_weights is not None:
                loss_mask = (loss_weights != 0).to(torch.float32)
            else:
                # Default: all ones (no masking)
                loss_mask = torch.ones(len(tokens), dtype=torch.float32)

            # Miles convention: the mask covers the response REGION, so
            # response_length is the mask's length (zeros inside are allowed);
            # counting nonzeros breaks prompt/response alignment in get_batch.
            # Same layout as the forward_backward RL path: append the final
            # target so the response is the T wire targets and the returned
            # logprobs are T-long, entry k = logprob of target k.
            target = cls._get_field(loss_fn_inputs, "target_tokens")
            if target is None:
                target = cls._get_field(loss_fn_inputs, "target")
            target_data = cls.extract_tensor_data(target) if target is not None else None
            if target_data and len(loss_mask) == len(tokens) and len(tokens) > 0:
                tokens_list[-1] = torch.cat([tokens_list[-1], torch.tensor(target_data[-1:], dtype=torch.long)])
            elif len(loss_mask) == len(tokens) and len(tokens) > 1:
                loss_mask = loss_mask[:-1]  # no target on the wire: the last target is unknowable
                if loss_weights is not None:
                    loss_weights = loss_weights[:-1]
            loss_masks_list.append(loss_mask)
            loss_weights_list.append(loss_weights if loss_weights is not None else torch.ones_like(loss_mask))
            response_lengths_list.append(len(loss_mask))
            # print(f"[CONVERTER DEBUG SFT] Sample {len(loss_masks_list)-1}: loss_mask sum={response_length}, len={len(loss_mask)}", flush=True)

        # Build rollout_data with dummy RL fields (not used for forward-only)
        batch_size = len(data)
        max_len = max(len(t) for t in tokens_list)

        rollout_data = {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "loss_weights": loss_weights_list,
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
                    # print(f"[CONVERTER] Auto-detected data format: is_rl={detected_is_rl} (was {is_rl}), "
                    #       f"has_advantages={has_advantages}, has_logprobs={has_logprobs}, "
                    #       f"has_weights={has_weights}, has_target={has_target}", flush=True)
                    is_rl = detected_is_rl

        # print(f"[CONVERTER DEBUG SFT] forward_backward_to_rollout called with {len(data)} samples, is_rl={is_rl}", flush=True)
        tokens_list = []
        loss_masks_list = []
        loss_weights_list = []
        response_lengths_list = []

        # RL-specific fields
        advantages_list = []
        log_probs_list = []
        ref_log_probs_list = [] if is_rl else None
        values_list = [] if is_rl else None
        returns_list = [] if is_rl else None

        # print(f"[CONVERTER] Converting {len(data)} forward_backward samples (is_rl={is_rl})", flush=True)
        logger.info(f"Converting {len(data)} forward_backward samples (is_rl={is_rl})")

        # Handle legacy HTTP test format: empty data or [{"input": "...", "target": "..."}]
        # This is for backward compatibility with test_4_multi_step_training.py
        if not data or len(data) == 0:
            logger.warning("[CONVERTER] Empty data provided - cannot generate fake test data without args")
            # Return minimal rollout_data that will fail validation
            return {
                "tokens": [],
                "loss_masks": [],
                "loss_weights": [],
                "response_lengths": [],
                "advantages": [] if is_rl else None,
                "log_probs": [] if is_rl else None,
                "ref_log_probs": [] if is_rl else None,
                "values": [] if is_rl else None,
                "returns": [] if is_rl else None
            }

        for idx, datum in enumerate(data):
            # print(f"[CONVERTER] Processing datum {idx}, type={type(datum)}", flush=True)
            # Extract input tokens
            model_input = cls._get_field(datum, "model_input")
            # print(f"[CONVERTER] model_input type={type(model_input)}, value={model_input}", flush=True)
            tokens = cls.extract_tokens_from_model_input(model_input)
            # print(f"[CONVERTER] Extracted {len(tokens)} input tokens: {tokens}", flush=True)
            logger.debug(f"Extracted {len(tokens)} input tokens: {tokens}")
            tokens_list.append(torch.tensor(tokens, dtype=torch.long))

            # Extract loss function inputs
            loss_fn_inputs = cls._get_field(datum, "loss_fn_inputs")

            if is_rl:
                # RL mode: Extract logprobs, mask, advantages, values, returns
                #
                # Key invariant: response_length must match the size of per-token tensors
                # (loss_mask, log_probs, advantages, etc.) that Miles uses in loss computation.

                # Step 1: Extract raw data from loss_fn_inputs
                logprobs = cls._get_field(loss_fn_inputs, "logprobs")
                logprobs_data = cls.extract_tensor_data(logprobs) if logprobs is not None else None
                # Loss mask: `mask` when the client sends one; otherwise the
                # target-aligned `weights` (custom-loss datums); otherwise all ones
                # (the cookbook's RL datums strip the mask and rely on zero
                # advantages outside the response).
                mask = cls._get_field(loss_fn_inputs, "mask")
                weights = cls._get_field(loss_fn_inputs, "weights")
                weights_data = cls.extract_tensor_data(weights) if weights is not None else None
                mask_from_weights = mask is None and weights is not None
                mask_data = cls.extract_tensor_data(mask) if mask is not None else weights_data

                # Step 2: Determine response_len from mask or logprobs
                if mask_data is not None:
                    response_len = len(mask_data)
                elif logprobs_data is not None:
                    response_len = len(logprobs_data)
                else:
                    response_len = len(tokens)

                # Step 3: Causal shift. The wire datum is pre-shifted: model_input
                # has T tokens and every per-token tensor has T entries, entry k
                # describing target k = model_input[k+1] (k < T-1) or the final
                # target (k = T-1). Miles computes one logprob per position after
                # the first token, so with model_input alone it would see only
                # T-1 targets: the final response token would get no loss or
                # logprob, and any trim shifts the tensors against the positions
                # they describe. Append the final target instead, as the SFT path
                # and the NeMo RL converter do, so the response is exactly the T
                # targets and every tensor lines up untouched.
                token_length = len(tokens)
                target = cls._get_field(loss_fn_inputs, "target_tokens")
                if target is None:
                    target = cls._get_field(loss_fn_inputs, "target")
                target_data = cls.extract_tensor_data(target) if target is not None else None
                needs_causal_trim = False
                if target_data and response_len == token_length and token_length > 0:
                    tokens_list[-1] = torch.cat([
                        tokens_list[-1], torch.tensor(target_data[-1:], dtype=torch.long),
                    ])
                elif response_len == token_length and token_length > 1:
                    # No target on the wire (not a Tinker RL datum): the last
                    # target is unknowable, so drop the LAST entry of each tensor
                    # (the one describing it) and train on T-1 positions.
                    logger.warning("RL datum %d carries no target_tokens; training on %d of %d targets", idx, token_length - 1, token_length)
                    needs_causal_trim = True
                    response_len = token_length - 1

                def maybe_trim(data: list) -> list:
                    return data[:-1] if needs_causal_trim and data else data

                # Step 4: Build per-token tensors (all must have length = response_len)
                if mask_data is not None:
                    loss_mask = torch.tensor(maybe_trim(mask_data), dtype=torch.float32)
                    if mask_from_weights:
                        loss_mask = (loss_mask != 0).to(torch.float32)
                else:
                    loss_mask = torch.ones(response_len, dtype=torch.float32)
                if weights_data is not None:
                    loss_weights_list.append(torch.tensor(maybe_trim(weights_data), dtype=torch.float32))
                else:
                    loss_weights_list.append(torch.ones(response_len, dtype=torch.float32))

                if logprobs_data is not None:
                    logprobs_clean = [0.0 if lp is None else float(lp) for lp in logprobs_data]
                    log_probs_list.append(torch.tensor(maybe_trim(logprobs_clean), dtype=torch.float32))
                else:
                    log_probs_list.append(torch.zeros(response_len, dtype=torch.float32))

                advantages = cls._get_field(loss_fn_inputs, "advantages")
                if advantages is not None:
                    adv_data = cls.extract_tensor_data(advantages)
                    advantages_list.append(torch.tensor(maybe_trim(adv_data), dtype=torch.float32))
                else:
                    advantages_list.append(torch.zeros(response_len, dtype=torch.float32))

                ref_logprobs = cls._get_field(loss_fn_inputs, "ref_logprobs")
                if ref_logprobs is not None:
                    ref_data = cls.extract_tensor_data(ref_logprobs)
                    ref_log_probs_list.append(torch.tensor(maybe_trim(ref_data), dtype=torch.float32))
                else:
                    # Use sampling logprobs as reference (policy at sampling time = "frozen" reference)
                    # log_probs_list[-1] contains the sampling logprobs we just added at line 268
                    # This enables proper KL penalty computation in Miles
                    ref_log_probs_list.append(log_probs_list[-1].clone())

                values = cls._get_field(loss_fn_inputs, "values")
                if values is not None:
                    val_data = cls.extract_tensor_data(values)
                    values_list.append(torch.tensor(maybe_trim(val_data), dtype=torch.float32))
                else:
                    values_list.append(torch.zeros(response_len, dtype=torch.float32))

                returns = cls._get_field(loss_fn_inputs, "returns")
                if returns is not None:
                    ret_data = cls.extract_tensor_data(returns)
                    returns_list.append(torch.tensor(maybe_trim(ret_data), dtype=torch.float32))
                else:
                    returns_list.append(torch.zeros(response_len, dtype=torch.float32))

                # Append to shared lists
                loss_masks_list.append(loss_mask)
                response_lengths_list.append(response_len)

                # DEBUG: Print per-sample lengths to diagnose mismatch
                # print(f"[CONVERTER DEBUG] Sample {idx}: response_len={response_len}, "
                #       f"advantages_len={len(advantages_list[-1])}, "
                #       f"logprobs_len={len(log_probs_list[-1])}, "
                #       f"loss_mask_len={len(loss_mask)}, "
                #       f"token_len={len(tokens)}, "
                #       f"needs_causal_trim={needs_causal_trim}", flush=True)

            else:
                # SFT mode: Extract target and weights
                target = cls._get_field(loss_fn_inputs, "target_tokens")
                if target is None:
                    target = cls._get_field(loss_fn_inputs, "target")

                weights = cls._get_field(loss_fn_inputs, "weights")
                if weights is None:
                    weights = cls._get_field(loss_fn_inputs, "weight")

                if not weights or not target:
                    raise ValueError("SFT loss_fn_inputs must contain weights and target_tokens/target")

                weights_data = cls.extract_tensor_data(weights)
                target_data = cls.extract_tensor_data(target)

                # Build full token sequence: input + last target token
                # Input: [1,2,3,4,5], Target: [2,3,4,5,6] -> Full: [1,2,3,4,5,6]
                input_tokens_tensor = torch.tensor(tokens, dtype=torch.long)
                target_tensor = torch.tensor(target_data, dtype=torch.long)
                full_tokens = torch.cat([input_tokens_tensor, target_tensor[-1:]], dim=0)
                tokens_list[-1] = full_tokens  # Replace the one we added earlier

                loss_weights = torch.tensor(weights_data, dtype=torch.float32)
                loss_mask = (loss_weights != 0).to(torch.float32)
                response_len = len(loss_mask)

                # Append to shared lists
                loss_masks_list.append(loss_mask)
                loss_weights_list.append(loss_weights)
                response_lengths_list.append(response_len)
                advantages_list.append(torch.zeros(response_len, dtype=torch.float32))
                log_probs_list.append(torch.zeros(response_len, dtype=torch.float32))

        # Build rollout_data
        rollout_data = {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "loss_weights": loss_weights_list,
            "response_lengths": response_lengths_list,
            "advantages": advantages_list,
            "log_probs": log_probs_list,
        }

        if is_rl:
            rollout_data["ref_log_probs"] = ref_log_probs_list
            rollout_data["values"] = values_list
            rollout_data["returns"] = returns_list
            # rollout_log_probs = sampling logprobs, needed for TIS (Truncated Importance Sampling)
            rollout_data["rollout_log_probs"] = [lp.clone() for lp in log_probs_list]
            # Flag for CP handling: Tinker sends full-size tensors
            # (logprobs, advantages computed client-side, not CP-split)
            rollout_data["_with_tinker"] = True
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

        # DEBUG: Print totals to diagnose mismatch
        if is_rl:
            total_response_len = sum(response_lengths_list)
            total_advantages_len = sum(len(a) for a in advantages_list)
            total_logprobs_len = sum(len(lp) for lp in log_probs_list)
            total_tokens_len = sum(len(t) for t in tokens_list)
            # print(f"[CONVERTER DEBUG] TOTALS: num_samples={len(tokens_list)}, "
            #       f"response_lengths_sum={total_response_len}, "
            #       f"advantages_sum={total_advantages_len}, "
            #       f"logprobs_sum={total_logprobs_len}, "
            #       f"tokens_sum={total_tokens_len}", flush=True)
            # print(f"[CONVERTER DEBUG] response_lengths_list={response_lengths_list}", flush=True)
            # print(f"[CONVERTER DEBUG] advantages_lens={[len(a) for a in advantages_list]}", flush=True)

        return rollout_data
