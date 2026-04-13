"""
NeMo RL data converter — converts Tinker Datum format to NeMo RL's
BatchedDataDict format for both RL and SFT training.

RL path (ClippedPGLossFn):
  tokens           → input_ids         [B, S]
  loss_masks       → token_mask        [B, S]
  advantages       → advantages        [B, S]
  log_probs        → prev_logprobs     [B, S]
  rollout_log_probs→ generation_logprobs[B, S]
  ref_log_probs    → reference_policy_logprobs [B, S]
  (computed)       → input_lengths     [B]
  (all ones)       → sample_mask       [B]

SFT path (NLLLoss, loss_fn="cross_entropy"):
  model_input.tokens + target_tokens[-1] → input_ids    [B, S]  (reconstructed)
  [0.0] + weights                        → token_mask   [B, S]  (reconstructed)
  (computed)                             → input_lengths [B]
  (all ones)                             → sample_mask   [B]
"""
import logging
from typing import Any, Dict, List

import torch

from ..base import DataConverter

logger = logging.getLogger(__name__)


class NemoRLDataConverter(DataConverter):
    """Converts between Tinker Datum format and NeMo RL BatchedDataDict."""

    def forward_to_backend(
        self,
        data: List[Dict],
        args: Any,
    ) -> Any:
        """
        Convert Tinker data to NeMo RL format for forward-only pass.

        Only includes fields needed for logprob computation:
        input_ids, input_lengths, token_mask.
        """
        if not data:
            return self._empty_batched_data_dict()

        max_seq_len = _compute_max_seq_len(data)
        batch_size = len(data)

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)

        for i, datum in enumerate(data):
            tokens = _extract_tokens(datum)
            seq_len = len(tokens)
            input_ids[i, :seq_len] = tokens
            input_lengths[i] = seq_len

            masks = _extract_loss_masks(datum, seq_len)
            token_mask[i, :seq_len] = masks

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        return BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
        })

    def forward_backward_to_backend(
        self,
        data: List[Dict],
        loss_fn: str,
        args: Any,
    ) -> Any:
        """
        Convert Tinker data to NeMo RL BatchedDataDict for training.

        Two paths:
        - loss_fn == "cross_entropy" (SFT): Builds BatchedDataDict for NLLLoss
          with reconstructed full sequence (input_ids, token_mask, sample_mask,
          input_lengths). No RL-specific fields.
        - Other loss_fn values (RL): Maps all fields needed by ClippedPGLossDataDict.
        """
        if not data:
            return self._empty_batched_data_dict()

        # SFT path — reconstruct full sequence for NLLLoss
        if loss_fn == "cross_entropy":
            return self._forward_backward_sft(data)

        max_seq_len = _compute_max_seq_len(data)
        batch_size = len(data)

        # Initialize tensors with padding (zeros)
        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        advantages = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        prev_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        generation_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        reference_policy_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        sample_mask = torch.ones(batch_size, dtype=torch.float32)

        for i, datum in enumerate(data):
            tokens = _extract_tokens(datum)
            seq_len = len(tokens)
            input_ids[i, :seq_len] = tokens
            input_lengths[i] = seq_len

            # Loss masks → token_mask (full sequence length, 0 for prompt, 1 for response)
            masks = _extract_loss_masks(datum, seq_len)
            token_mask[i, :seq_len] = masks

            # Response-length fields need to be placed in the response portion
            response_length = int(masks.sum().item())
            prompt_length = seq_len - response_length

            if i == 0:
                logger.info(
                    "RL converter: seq_len=%d, prompt_len=%d, resp_len=%d",
                    seq_len, prompt_length, response_length,
                )

            # Advantages (response-length → placed at response positions)
            adv = _extract_field(datum, "advantages", response_length)
            if adv is not None:
                advantages[i, prompt_length:seq_len] = adv

            # Prev logprobs (log_probs from behavior policy)
            lp = _extract_field(datum, "log_probs", response_length)
            if lp is not None:
                prev_logprobs[i, prompt_length:seq_len] = lp

            # Generation logprobs (rollout_log_probs from SGLang sampling)
            rlp = _extract_field(datum, "rollout_log_probs", response_length)
            if rlp is not None:
                generation_logprobs[i, prompt_length:seq_len] = rlp

            # Reference policy logprobs
            ref_lp = _extract_field(datum, "ref_log_probs", response_length)
            if ref_lp is not None:
                reference_policy_logprobs[i, prompt_length:seq_len] = ref_lp

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        return BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "advantages": advantages,
            "prev_logprobs": prev_logprobs,
            "generation_logprobs": generation_logprobs,
            "reference_policy_logprobs": reference_policy_logprobs,
            "sample_mask": sample_mask,
        })

    def _forward_backward_sft(self, data: List[Dict]) -> Any:
        """Convert SFT data to NeMo RL BatchedDataDict for NLLLoss.

        SFT datum provides (from datum_from_tokens_weights in tinker-cookbook):
          model_input.tokens       = tokens[:-1]   (length N-1)
          loss_fn_inputs["target_tokens"] = tokens[1:]    (length N-1)
          loss_fn_inputs["weights"]       = weights[1:]   (length N-1)

        NLLLoss expects:
          input_ids  = full unshifted sequence [B, S]  (length N)
          token_mask = full-length mask [B, S]          (NLLLoss slices [:, 1:])
          sample_mask = [B]
          input_lengths = [B]

        Reconstruction:
          input_ids  = concat(model_input.tokens, [target_tokens[-1]])  → length N
          token_mask = concat([0.0], weights)                           → length N

        After NLLLoss slices [:, 1:], token_mask becomes weights and
        input_ids becomes target_tokens — correct alignment.
        """
        batch_size = len(data)

        # Compute max reconstructed sequence length (N-1 input tokens + 1 = N)
        max_seq_len = 0
        for datum in data:
            tokens = _extract_tokens(datum)
            max_seq_len = max(max_seq_len, len(tokens) + 1)  # +1 for reconstruction

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        sample_mask = torch.ones(batch_size, dtype=torch.float32)

        for i, datum in enumerate(data):
            # Extract model_input.tokens (length N-1)
            input_tokens = _extract_tokens(datum)

            # Extract target_tokens (length N-1) — need last element
            target_tokens = _extract_sft_target_tokens(datum)

            # Reconstruct full sequence: concat(input_tokens, [target_tokens[-1]])
            if target_tokens is not None and len(target_tokens) > 0:
                last_token = target_tokens[-1].unsqueeze(0)
                full_tokens = torch.cat([input_tokens, last_token])
            else:
                # Malformed SFT datum — exclude from loss via sample_mask=0
                full_tokens = input_tokens
                sample_mask[i] = 0.0
                logger.warning(
                    "SFT datum %d: no target_tokens found, excluding from training", i,
                )

            seq_len = len(full_tokens)
            input_ids[i, :seq_len] = full_tokens
            input_lengths[i] = seq_len

            # Extract weights (length N-1)
            weights = _extract_sft_weights(datum)

            # Reconstruct full mask: concat([0.0], weights) → length N
            if weights is not None:
                full_mask = torch.cat([torch.zeros(1, dtype=torch.float32), weights])
            else:
                # Fallback: mask entire response (all ones except first position)
                full_mask = torch.ones(seq_len, dtype=torch.float32)
                full_mask[0] = 0.0
                logger.warning(
                    "SFT datum %d: no weights found, using all-ones mask", i,
                )

            mask_len = min(len(full_mask), seq_len)
            token_mask[i, :mask_len] = full_mask[:mask_len]

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        logger.info(
            "SFT conversion: batch_size=%d, max_seq_len=%d (reconstructed from N-1 input tokens)",
            batch_size, max_seq_len,
        )

        return BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        })

    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Dict],
        loss_fn: str = "",
    ) -> Dict[str, Any]:
        """
        Convert NeMo RL forward-only result to Tinker format.

        NeMo RL get_logprobs() returns BatchedDataDict with key "logprobs" [B, S].
        Response must match ForwardBackwardOutput schema (shared by forward and
        forward_backward on the SDK side).
        """
        logprobs_tensor = None
        # BatchedDataDict inherits from UserDict (not dict), so use
        # hasattr+get to handle both dict and UserDict results.
        if hasattr(result, "get"):
            logprobs_tensor = result.get("logprobs", None)
        elif hasattr(result, "logprobs"):
            logprobs_tensor = result.logprobs

        loss_fn_outputs = []
        if logprobs_tensor is not None:
            for i in range(len(data)):
                lp = logprobs_tensor[i].detach().cpu()
                # Trim padded logprobs to actual sequence length
                seq_len = len(_extract_tokens(data[i]))
                lp = lp[:seq_len]
                lp_list = lp.tolist()
                loss_fn_outputs.append({
                    "logprobs": {
                        "data": lp_list,
                        "shape": [len(lp_list)],
                        "dtype": "float32",
                    },
                })
        else:
            for _ in data:
                loss_fn_outputs.append({
                    "logprobs": {
                        "data": [],
                        "shape": [0],
                        "dtype": "float32",
                    },
                })

        return {
            "loss_fn_output_type": loss_fn,
            "loss_fn_outputs": loss_fn_outputs,
            "metrics": {},
        }

    def backend_to_forward_backward_result(
        self,
        result: Any,
        data: List[Dict],
        loss_fn: str = "",
    ) -> Dict[str, Any]:
        """
        Convert NeMo RL training result to Tinker format.

        NeMo RL policy.train() returns dict with:
        - "loss": scalar loss value
        - "grad_norm": gradient norm
        - "all_mb_metrics": dict of per-microbatch metrics
        """
        if result is None:
            # Deferred result (buffering mode)
            return {
                "loss_fn_output_type": loss_fn,
                "loss": None,
                "grad_norm": None,
                "metrics": {},
                "deferred": True,
                "loss_fn_outputs": [],
            }

        loss = _to_python_scalar(result.get("loss", 0.0))
        grad_norm = _to_python_scalar(result.get("grad_norm", 0.0))

        # Extract per-microbatch metrics
        all_mb_metrics = result.get("all_mb_metrics", {})
        metrics = {
            "total_loss": loss,
            "grad_norm": grad_norm,
        }

        # Map NeMo RL metric names to normalized schema
        metric_mapping = {
            "policy_loss": "pg_loss",
            "entropy": "entropy_loss",
            "kl": "ppo_kl",
            "ratio_mean": "importance_ratio_mean",
            "ratio_clipped_fraction": "clip_fraction",
        }
        for nemo_key, tinker_key in metric_mapping.items():
            if nemo_key in all_mb_metrics:
                vals = all_mb_metrics[nemo_key]
                if isinstance(vals, list) and vals:
                    metrics[tinker_key] = _to_python_scalar(
                        sum(_to_python_scalar(v) for v in vals) / len(vals)
                    )
                else:
                    metrics[tinker_key] = _to_python_scalar(vals)

        return {
            "loss_fn_output_type": loss_fn,
            "loss": loss,
            "grad_norm": grad_norm,
            "metrics": metrics,
            "deferred": False,
            "loss_fn_outputs": [],
        }

    @staticmethod
    def _empty_batched_data_dict():
        """Return an empty BatchedDataDict."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        return BatchedDataDict({
            "input_ids": torch.zeros(0, 0, dtype=torch.long),
            "input_lengths": torch.zeros(0, dtype=torch.long),
            "token_mask": torch.zeros(0, 0, dtype=torch.float32),
        })


def _to_python_scalar(val):
    """Convert a torch.Tensor or numpy scalar to a Python float/int."""
    if isinstance(val, (int, float)):
        return val
    if hasattr(val, "item"):  # torch.Tensor or numpy scalar
        return val.item()
    return float(val)


# ---------------------------------------------------------------------------
# Helper functions for extracting fields from Tinker Datum dicts / objects
#
# Datum can arrive in two formats:
#   1. Pydantic ForwardBackwardDatum: datum.model_input has tokens,
#      datum.loss_fn_inputs is Dict[str, TensorData]
#   2. Flat dict: datum["tokens"], datum["log_probs"] (raw lists)
# ---------------------------------------------------------------------------


def _get_attr_or_key(obj, field: str):
    """Get a field from a dict or object (used for datum-level access)."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _extract_tokens(datum) -> torch.Tensor:
    """Extract token IDs from datum (Pydantic ForwardBackwardDatum, dict, or flat object)."""
    # Try nested model_input first (Pydantic ForwardBackwardDatum format)
    model_input = _get_attr_or_key(datum, "model_input")
    if model_input is not None:
        # Try chunks format: model_input.chunks[0].tokens
        chunks = _get_attr_or_key(model_input, "chunks")
        if chunks:
            tokens = _get_attr_or_key(chunks[0], "tokens")
            if tokens is not None:
                if isinstance(tokens, torch.Tensor):
                    return tokens.detach().cpu().long()
                return torch.tensor(tokens, dtype=torch.long)

        # Try direct tokens: model_input.tokens
        tokens = _get_attr_or_key(model_input, "tokens")
        if tokens is not None:
            if isinstance(tokens, torch.Tensor):
                return tokens.detach().cpu().long()
            return torch.tensor(tokens, dtype=torch.long)

        # Try input_ids: model_input.input_ids
        input_ids = _get_attr_or_key(model_input, "input_ids")
        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                return input_ids.detach().cpu().long()
            return torch.tensor(input_ids, dtype=torch.long)

    # Fall back to flat format: datum.tokens or datum.input_ids
    if isinstance(datum, dict):
        tokens = datum.get("tokens", datum.get("input_ids", []))
    else:
        tokens = getattr(datum, "tokens", getattr(datum, "input_ids", []))

    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().long()
    return torch.tensor(tokens, dtype=torch.long)


def _extract_loss_masks(datum, seq_len: int) -> torch.Tensor:
    """Extract loss masks from datum, padded to seq_len.

    Tries loss_fn_inputs["mask"] first, then datum.loss_masks / datum.loss_mask.
    Returns all-ones if no mask found (requires "mask" to flow through SDK).
    """
    masks = None

    # Try loss_fn_inputs["mask"] (Dict[str, TensorData] format)
    loss_fn_inputs = _get_attr_or_key(datum, "loss_fn_inputs")
    if isinstance(loss_fn_inputs, dict):
        mask_obj = loss_fn_inputs.get("mask")
        if mask_obj is not None:
            masks = mask_obj.data if hasattr(mask_obj, "data") else mask_obj

    # Fall back to flat format: datum.loss_masks or datum.loss_mask
    if masks is None:
        if isinstance(datum, dict):
            masks = datum.get("loss_masks", datum.get("loss_mask", None))
        else:
            masks = getattr(datum, "loss_masks", getattr(datum, "loss_mask", None))

    if masks is None:
        return torch.ones(seq_len, dtype=torch.float32)

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().float()
    else:
        masks = torch.tensor(masks, dtype=torch.float32)

    # If masks are response-length, expand to full sequence length
    if len(masks) < seq_len:
        full = torch.zeros(seq_len, dtype=torch.float32)
        full[seq_len - len(masks):] = masks
        return full

    return masks[:seq_len]


# Mapping from converter field names to SDK dict key names
_FIELD_NAME_MAP = {
    "advantages": "advantages",
    "log_probs": "logprobs",          # Tinker flat "log_probs" → SDK dict "logprobs"
    "ref_log_probs": "ref_logprobs",  # Tinker flat "ref_log_probs" → SDK dict "ref_logprobs"
    # "rollout_log_probs" has no SDK dict equivalent — generation logprobs are
    # computed by the sampling service, not passed in forward_backward requests.
}


def _extract_field(datum, field_name: str, expected_len: int):
    """Extract a response-length tensor field from datum.

    Handles both:
      - Pydantic ForwardBackwardDatum: datum.loss_fn_inputs[sdk_key].data
      - Flat dict: datum[field_name] (raw list or tensor)
    """
    value = None

    # Try loss_fn_inputs dict (Dict[str, TensorData] format)
    loss_fn_inputs = _get_attr_or_key(datum, "loss_fn_inputs")
    if isinstance(loss_fn_inputs, dict):
        sdk_key = _FIELD_NAME_MAP.get(field_name, field_name)
        tensor_obj = loss_fn_inputs.get(sdk_key)
        if tensor_obj is not None:
            value = tensor_obj.data if hasattr(tensor_obj, "data") else tensor_obj

    # Fall back to flat format: datum[field_name]
    if value is None:
        if isinstance(datum, dict):
            value = datum.get(field_name, None)
        else:
            value = getattr(datum, field_name, None)

    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().float()
    elif isinstance(value, (list, tuple)):
        t = torch.tensor(value, dtype=torch.float32)
    else:
        return None

    # Truncate or pad to expected length.
    # The cookbook prepends zeros for prompt tokens, so full-sequence fields
    # have structure [0, 0, ..., 0, val_0, val_1, ..., val_R].
    # Take the LAST expected_len elements to get response-only values.
    if expected_len == 0:
        return torch.zeros(0, dtype=torch.float32)
    if len(t) >= expected_len:
        return t[-expected_len:]
    else:
        padded = torch.zeros(expected_len, dtype=torch.float32)
        padded[:len(t)] = t
        return padded


def _extract_sft_target_tokens(datum) -> torch.Tensor:
    """Extract target_tokens from SFT datum's loss_fn_inputs.

    SFT datums store target_tokens = tokens[1:] in loss_fn_inputs["target_tokens"].
    Handles both Pydantic ForwardBackwardDatum and flat dict formats.

    Returns:
        torch.Tensor of token IDs (dtype=long), or None if not found.
    """
    loss_fn_inputs = _get_attr_or_key(datum, "loss_fn_inputs")
    if isinstance(loss_fn_inputs, dict):
        target_obj = loss_fn_inputs.get("target_tokens")
        if target_obj is not None:
            data = target_obj.data if hasattr(target_obj, "data") else target_obj
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().long()
            return torch.tensor(data, dtype=torch.long)

    # Fall back to flat format
    if isinstance(datum, dict):
        value = datum.get("target_tokens")
    else:
        value = getattr(datum, "target_tokens", None)

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().long()
    return torch.tensor(value, dtype=torch.long)


def _extract_sft_weights(datum) -> torch.Tensor:
    """Extract weights from SFT datum's loss_fn_inputs.

    SFT datums store weights = weights[1:] in loss_fn_inputs["weights"].
    Handles both Pydantic ForwardBackwardDatum and flat dict formats.

    Returns:
        torch.Tensor of float weights, or None if not found.
    """
    loss_fn_inputs = _get_attr_or_key(datum, "loss_fn_inputs")
    if isinstance(loss_fn_inputs, dict):
        weights_obj = loss_fn_inputs.get("weights")
        if weights_obj is not None:
            data = weights_obj.data if hasattr(weights_obj, "data") else weights_obj
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().float()
            return torch.tensor(data, dtype=torch.float32)

    # Fall back to flat format
    if isinstance(datum, dict):
        value = datum.get("weights")
    else:
        value = getattr(datum, "weights", None)

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.tensor(value, dtype=torch.float32)


def _compute_max_seq_len(data: List) -> int:
    """Compute maximum sequence length across all data samples."""
    max_len = 0
    for datum in data:
        tokens = _extract_tokens(datum)
        max_len = max(max_len, len(tokens))
    return max_len
