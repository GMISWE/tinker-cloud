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
                # RL mode: Extract advantages, logprobs, mask
                advantages = cls._get_field(loss_fn_inputs, "advantages")
                advantages_data = cls.extract_tensor_data(advantages)
                advantages_list.append(torch.tensor(advantages_data, dtype=torch.float32))

                logprobs = cls._get_field(loss_fn_inputs, "logprobs")
                logprobs_data = cls.extract_tensor_data(logprobs)
                log_probs_list.append(torch.tensor(logprobs_data, dtype=torch.float32))

                # Optional: ref_log_probs, values, returns
                ref_logprobs = cls._get_field(loss_fn_inputs, "ref_logprobs")
                if ref_logprobs is not None:
                    ref_logprobs_data = cls.extract_tensor_data(ref_logprobs)
                    ref_log_probs_list.append(torch.tensor(ref_logprobs_data, dtype=torch.float32))
                else:
                    ref_log_probs_list.append(torch.zeros(len(tokens), dtype=torch.float32))

                values = cls._get_field(loss_fn_inputs, "values")
                if values is not None:
                    values_data = cls.extract_tensor_data(values)
                    values_list.append(torch.tensor(values_data, dtype=torch.float32))
                else:
                    values_list.append(torch.zeros(len(tokens), dtype=torch.float32))

                returns = cls._get_field(loss_fn_inputs, "returns")
                if returns is not None:
                    returns_data = cls.extract_tensor_data(returns)
                    returns_list.append(torch.tensor(returns_data, dtype=torch.float32))
                else:
                    returns_list.append(torch.zeros(len(tokens), dtype=torch.float32))

                # Mask
                mask = cls._get_field(loss_fn_inputs, "mask")
                if mask is not None:
                    mask_data = cls.extract_tensor_data(mask)
                    loss_mask = torch.tensor(mask_data, dtype=torch.float32)
                else:
                    loss_mask = torch.ones(len(tokens), dtype=torch.float32)

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
            response_length = int(loss_mask.sum().item())
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

        logger.info(f"Converted {len(data)} {'RL' if is_rl else 'SFT'} samples to rollout_data with {len(tokens_list)} token sequences")
        return rollout_data

    @staticmethod
    def rollout_to_forward_result(
        results: List[Dict[str, Any]],
        loss_fn: str = "cross_entropy"
    ) -> Dict[str, Any]:
        """
        Convert Slime forward_only results to Tinker forward result format.

        Args:
            results: List of results from Slime (per-GPU results)
            loss_fn: Loss function type

        Returns:
            Tinker forward result dict
        """
        loss_fn_outputs = []

        for result in results:
            loss_value = TinkerDataConverter._extract_scalar_loss(result.get("loss", 0.0))
            logprobs = TinkerDataConverter._extract_logprob_list(result.get("logprobs", []))
            loss_fn_outputs.append({
                "loss": {
                    "data": [loss_value],
                    "shape": [1],
                    "dtype": "float32"
                },
                "logprobs": {
                    "data": logprobs,
                    "shape": [len(logprobs)],
                    "dtype": "float32"
                } if logprobs else None
            })

        return {
            "type": "forward",
            "loss_fn_output_type": loss_fn,
            "loss_fn_outputs": loss_fn_outputs
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
        response_lengths_list = []
        if original_data:
            for idx, datum in enumerate(original_data):
                # Extract weights from original Tinker request
                loss_fn_inputs = datum.get("loss_fn_inputs") if isinstance(datum, dict) else getattr(datum, "loss_fn_inputs", None)
                if loss_fn_inputs:
                    weights_data = loss_fn_inputs.get("weights") if isinstance(loss_fn_inputs, dict) else getattr(loss_fn_inputs, "weights", None)
                    if weights_data:
                        # Extract tensor data - handle TensorData Pydantic model
                        weights = TinkerDataConverter.extract_tensor_data(weights_data)
                        response_length = len(weights)
                        response_lengths_list.append(response_length)
                        print(f"[CONVERTER DEBUG] Original weights[{idx}] length = {response_length}", flush=True)
                    else:
                        response_lengths_list.append(0)
                else:
                    response_lengths_list.append(0)
        else:
            print(f"[CONVERTER DEBUG] No original_data provided, using empty response_lengths", flush=True)

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
                if val is not None:
                    return TinkerDataConverter._extract_scalar_loss(val)
        return 0.0

    @staticmethod
    def _extract_logprob_list(logprobs_entry: Any) -> List[float]:
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
                head = item[0]
                if isinstance(head, (int, float)):
                    value = float(head)
                elif hasattr(head, "item"):
                    try:
                        value = float(head.item())
                    except Exception:
                        value = None
            if value is not None:
                normalized.append(value)
        return normalized
