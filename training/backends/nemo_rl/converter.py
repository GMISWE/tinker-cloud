"""
NeMo RL data converter — converts Tinker Datum format to NeMo RL's
BatchedDataDict[ClippedPGLossDataDict] format.

Field mapping (Tinker → NeMo RL):
  tokens           → input_ids         [B, S]
  loss_masks       → token_mask        [B, S]
  advantages       → advantages        [B, S]
  log_probs        → prev_logprobs     [B, S]
  rollout_log_probs→ generation_logprobs[B, S]
  ref_log_probs    → reference_policy_logprobs [B, S]
  (computed)       → input_lengths     [B]
  (all ones)       → sample_mask       [B]
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

        Maps all fields needed by ClippedPGLossDataDict:
        input_ids, advantages, prev_logprobs, generation_logprobs,
        reference_policy_logprobs, token_mask, sample_mask, input_lengths.
        """
        if not data:
            return self._empty_batched_data_dict()

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

    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Convert NeMo RL forward-only result to Tinker format.

        NeMo RL get_logprobs() returns BatchedDataDict with key "logprobs" [B, S].
        """
        logprobs_tensor = None
        if isinstance(result, dict):
            logprobs_tensor = result.get("logprobs", None)
        elif hasattr(result, "logprobs"):
            logprobs_tensor = result.logprobs

        loss_fn_outputs = []
        if logprobs_tensor is not None:
            for i in range(len(data)):
                lp = logprobs_tensor[i].detach().cpu()
                loss_fn_outputs.append({
                    "logprobs": {"data": lp.tolist()},
                })
        else:
            for _ in data:
                loss_fn_outputs.append({"logprobs": {"data": []}})

        return {
            "loss_fn_outputs": loss_fn_outputs,
        }

    def backend_to_forward_backward_result(
        self,
        result: Any,
        data: List[Dict],
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
#   1. Pydantic ForwardBackwardDatum: datum.model_input.tokens,
#      datum.loss_fn_inputs.logprobs.data (TensorData objects)
#   2. Flat dict: datum["tokens"], datum["log_probs"] (raw lists)
# ---------------------------------------------------------------------------

def _get_field(obj, field: str):
    """Get a field from a dict or object."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _extract_tensor_data(tensor_obj) -> list:
    """Extract raw data from a TensorData object or passthrough for raw lists."""
    if tensor_obj is None:
        return None
    # TensorData has a .data field containing the raw list
    data = _get_field(tensor_obj, "data")
    if data is not None:
        return data
    # Already a raw list/tensor
    return tensor_obj


def _extract_tokens(datum) -> torch.Tensor:
    """Extract token IDs from datum (Pydantic ForwardBackwardDatum, dict, or flat object)."""
    # Try nested model_input first (Pydantic ForwardBackwardDatum format)
    model_input = _get_field(datum, "model_input")
    if model_input is not None:
        # Try chunks format: model_input.chunks[0].tokens
        chunks = _get_field(model_input, "chunks")
        if chunks:
            tokens = _get_field(chunks[0], "tokens")
            if tokens is not None:
                if isinstance(tokens, torch.Tensor):
                    return tokens.detach().cpu().long()
                return torch.tensor(tokens, dtype=torch.long)

        # Try direct tokens: model_input.tokens
        tokens = _get_field(model_input, "tokens")
        if tokens is not None:
            if isinstance(tokens, torch.Tensor):
                return tokens.detach().cpu().long()
            return torch.tensor(tokens, dtype=torch.long)

        # Try input_ids: model_input.input_ids
        input_ids = _get_field(model_input, "input_ids")
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


def _get_loss_fn_inputs(datum):
    """Get the loss_fn_inputs container from datum."""
    return _get_field(datum, "loss_fn_inputs")


def _extract_loss_masks(datum, seq_len: int) -> torch.Tensor:
    """Extract loss masks from datum, padded to seq_len."""
    masks = None

    # Try nested loss_fn_inputs.mask (Pydantic format — TensorData)
    loss_fn_inputs = _get_loss_fn_inputs(datum)
    if loss_fn_inputs is not None:
        mask_obj = _get_field(loss_fn_inputs, "mask")
        if mask_obj is not None:
            masks = _extract_tensor_data(mask_obj)

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


# Mapping from NeMo RL converter field names to Pydantic loss_fn_inputs field names
_FIELD_TO_PYDANTIC = {
    "advantages": "advantages",
    "log_probs": "logprobs",          # Tinker flat "log_probs" → Pydantic "logprobs"
    "ref_log_probs": "ref_logprobs",  # Tinker flat "ref_log_probs" → Pydantic "ref_logprobs"
    # "rollout_log_probs" has no Pydantic equivalent — generation logprobs are
    # computed by the sampling service, not passed in forward_backward requests.
}


def _extract_field(datum, field_name: str, expected_len: int):
    """Extract a response-length tensor field from datum.

    Handles both:
      - Pydantic ForwardBackwardDatum: datum.loss_fn_inputs.<pydantic_field>.data
      - Flat dict: datum[field_name] (raw list or tensor)
    """
    value = None

    # Try nested loss_fn_inputs (Pydantic ForwardBackwardDatum path)
    loss_fn_inputs = _get_loss_fn_inputs(datum)
    if loss_fn_inputs is not None:
        pydantic_field = _FIELD_TO_PYDANTIC.get(field_name, field_name)
        tensor_obj = _get_field(loss_fn_inputs, pydantic_field)
        if tensor_obj is not None:
            value = _extract_tensor_data(tensor_obj)

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

    # Truncate or pad to expected length
    if len(t) >= expected_len:
        return t[:expected_len]
    else:
        padded = torch.zeros(expected_len, dtype=torch.float32)
        padded[:len(t)] = t
        return padded


def _compute_max_seq_len(data: List) -> int:
    """Compute maximum sequence length across all data samples."""
    max_len = 0
    for datum in data:
        tokens = _extract_tokens(datum)
        max_len = max(max_len, len(tokens))
    return max_len
