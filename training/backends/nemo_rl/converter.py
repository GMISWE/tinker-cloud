"""
NeMo RL data converter — converts Tinker Datum format to NeMo RL's
BatchedDataDict format for both RL and SFT training.

RL path (ClippedPGLossFn):
  model_input.tokens + target_tokens[-1] → input_ids  [B, S]  (reconstructed)
  mask (right-aligned)                   → token_mask [B, S]
  advantages (right-aligned)             → advantages [B, S]
  log_probs (right-aligned)              → prev_logprobs [B, S]
  rollout_log_probs (right-aligned)      → generation_logprobs [B, S]
  ref_log_probs (right-aligned)          → reference_policy_logprobs [B, S]
  (computed)                             → input_lengths [B]
  (all ones)                             → sample_mask   [B]

  Position convention: NeMo RL losses slice [B, S] fields [:, 1:], so
  position p must describe target token x_p. Alignment details and the
  wire/flat layout cases: specs/001-dual-backend-support/bugs/BUG-012.

SFT path (NLLLoss, loss_fn="cross_entropy"):
  model_input.tokens + target_tokens[-1] → input_ids    [B, S]  (reconstructed)
  [0.0] + weights                        → token_mask   [B, S]  (reconstructed)
  (computed)                             → input_lengths [B]
  (all ones)                             → sample_mask   [B]
"""
import logging
from typing import Any, Dict, List, Optional

import torch

from ..base import DataConverter
from ...models.requests import Datum

logger = logging.getLogger(__name__)


class NemoRLDataConverter(DataConverter):
    """Converts between Tinker Datum format and NeMo RL BatchedDataDict."""

    def forward_to_backend(
        self,
        data: List[Datum],
        args: Any,
    ) -> Any:
        """
        Convert Tinker data to NeMo RL format for forward-only pass.

        Only includes fields needed for logprob computation (input_ids,
        input_lengths, token_mask) plus sample_mask for dp-size padding.
        Datums with target_tokens are reconstructed to the full sequence;
        backend_to_forward_result() drops position 0 to keep alignment.
        """
        if not data:
            return self._empty_batched_data_dict()

        full_tokens_B = [_full_sequence_tokens(datum) for datum in data]
        max_seq_len = max(len(t) for t in full_tokens_B)
        batch_size = len(data)

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        sample_mask = torch.ones(batch_size, dtype=torch.float32)

        for i, (datum, full_tokens) in enumerate(zip(data, full_tokens_B)):
            seq_len = len(full_tokens)
            input_ids[i, :seq_len] = full_tokens
            input_lengths[i] = seq_len

            masks = _extract_loss_masks(datum, seq_len)
            token_mask[i, :seq_len] = masks

        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        return BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        })

    def forward_backward_to_backend(
        self,
        data: List[Datum],
        loss_fn: str,
        args: Any,
        image_preprocessor=None,
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

        if loss_fn == "cross_entropy":
            return self._forward_backward_sft(data, image_preprocessor)

        full_tokens_B = [_full_sequence_tokens(datum) for datum in data]

        max_seq_len = max((len(t) for t in full_tokens_B), default=0)
        batch_size = len(data)

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        advantages = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        prev_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        generation_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        reference_policy_logprobs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        sample_mask = torch.ones(batch_size, dtype=torch.float32)

        for i, (datum, full_tokens) in enumerate(zip(data, full_tokens_B)):
            seq_len = len(full_tokens)
            input_ids[i, :seq_len] = full_tokens
            input_lengths[i] = seq_len

            masks = _extract_loss_masks(datum, seq_len)
            token_mask[i, :seq_len] = masks

            if i == 0:
                logger.info(
                    "RL converter: seq_len=%d, masked_positions=%d",
                    seq_len, int(masks.sum().item()),
                )

            for field_name, dest in (
                ("advantages", advantages),
                ("log_probs", prev_logprobs),
                ("rollout_log_probs", generation_logprobs),
                ("ref_log_probs", reference_policy_logprobs),
            ):
                values = _extract_field(datum, field_name)
                if values is not None:
                    _place_right_aligned(dest[i], values, seq_len)

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

    def _forward_backward_sft(self, data: List[Datum], image_preprocessor=None) -> Any:
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
        image_token_id = (
            image_preprocessor.image_token_id if image_preprocessor is not None else None
        )

        # VLM: expand image chunks to placeholder tokens; text-only: concat text chunks.
        def _full_input_tokens(datum):
            if image_token_id is not None:
                expanded = _expand_chunks_to_full_sequence(datum, image_token_id)
                if len(expanded) > 0:
                    return expanded
            return _extract_tokens(datum)

        max_seq_len = 0
        for datum in data:
            max_seq_len = max(max_seq_len, len(_full_input_tokens(datum)) + 1)

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        input_lengths = torch.zeros(batch_size, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        sample_mask = torch.ones(batch_size, dtype=torch.float32)

        for i, datum in enumerate(data):
            input_tokens = _full_input_tokens(datum)

            target_tokens = _extract_target_tokens(datum)

            if target_tokens is not None and len(target_tokens) > 0:
                last_token = target_tokens[-1].unsqueeze(0)
                full_tokens = torch.cat([input_tokens, last_token])
            else:
                full_tokens = input_tokens
                sample_mask[i] = 0.0
                logger.warning(
                    "SFT datum %d: no target_tokens found, excluding from training", i,
                )

            seq_len = len(full_tokens)
            input_ids[i, :seq_len] = full_tokens
            input_lengths[i] = seq_len

            weights = _extract_sft_weights(datum)

            if weights is not None:
                full_mask = torch.cat([torch.zeros(1, dtype=torch.float32), weights])
            else:
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


        batch_dict = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
        }

        if image_preprocessor is not None:
            has_images = any(_extract_images(datum) for datum in data)
            if has_images:
                from nemo_rl.data.multimodal_utils import PackedTensor, get_dim_to_pack_along

                multimodal_fields = {key: [] for key in image_preprocessor.multimodal_keys}
                for datum in data:
                    images = _extract_images(datum)
                    if images:
                        processed = image_preprocessor.process_images(images)
                        for key in image_preprocessor.multimodal_keys:
                            multimodal_fields[key].append(processed.get(key))
                    else:
                        for key in image_preprocessor.multimodal_keys:
                            multimodal_fields[key].append(None)

                for key, tensors_list in multimodal_fields.items():
                    dim = get_dim_to_pack_along(image_preprocessor.processor, key)
                    batch_dict[key] = PackedTensor(tensors_list, dim_to_pack=dim)

        return BatchedDataDict(batch_dict)


    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Datum],
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
                tokens_len = len(_extract_tokens(data[i]))
                target_tokens = _extract_target_tokens(data[i])
                if target_tokens is not None and len(target_tokens) > 0:
                    # lp[1:] element k = logprob of target_tokens[k]
                    lp = lp[1:tokens_len + 1]
                else:
                    lp = lp[:tokens_len]
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
        data: List[Datum],
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
# Field access on the validated wire Datum (models.requests.Datum): model_input
# is a ModelInput, loss_fn_inputs a Dict[str, TensorData]. Optional loss inputs
# are a real domain state (an SFT datum has no advantages), hence .get on the
# dict; the model fields themselves are read directly.
# ---------------------------------------------------------------------------


def _extract_tokens(datum: Datum) -> torch.Tensor:
    """Token ids from model_input: the text chunks concatenated (image chunks
    skipped: their placeholder tokens already sit in adjacent text chunks),
    else the flat tokens / input_ids form."""
    mi = datum.model_input
    if mi.chunks:
        parts = [torch.tensor(c.tokens, dtype=torch.long) for c in mi.chunks
                 if c.type != "image" and c.tokens is not None]
        if parts:
            return torch.cat(parts)
    if mi.tokens is not None:
        return torch.tensor(mi.tokens, dtype=torch.long)
    if mi.input_ids is not None:
        return torch.tensor(mi.input_ids, dtype=torch.long)
    raise ValueError("model_input carries no chunks, tokens or input_ids")


def _expand_chunks_to_full_sequence(datum: Datum, image_token_id: int) -> torch.Tensor:
    """Expand model_input.chunks into a dense token sequence.

    Text chunks contribute their tokens directly. Image chunks contribute
    `expected_tokens` copies of `image_token_id` (e.g. <|image_pad|>=151655
    for Qwen3-VL). The resulting sequence matches the full expanded sequence
    that the cookbook used to compute target_tokens and weights.
    """
    all_tokens: list = []
    for chunk in datum.model_input.chunks or []:
        if chunk.type == "image":
            all_tokens.extend([image_token_id] * int(chunk.expected_tokens or 0))
        else:
            all_tokens.extend(chunk.tokens or [])
    return torch.tensor(all_tokens, dtype=torch.long)


def _extract_images(datum: Datum) -> list:
    """Decoded image bytes, one per image chunk (empty when there are none)."""
    import base64
    return [base64.b64decode(chunk.data)
            for chunk in datum.model_input.chunks or []
            if chunk.type == "image" and chunk.data is not None]


def _extract_loss_masks(datum: Datum, seq_len: int) -> torch.Tensor:
    """loss_fn_inputs["mask"] padded to seq_len; all-ones when the client sent
    none (the cookbook's RL datums strip the mask and rely on zero advantages)."""
    mask = datum.loss_fn_inputs.get("mask")
    if mask is None:
        return torch.ones(seq_len, dtype=torch.float32)
    masks = torch.tensor(mask.data, dtype=torch.float32)
    # If masks are response-length, expand to full sequence length
    if len(masks) < seq_len:
        full = torch.zeros(seq_len, dtype=torch.float32)
        full[seq_len - len(masks):] = masks
        return full
    return masks[:seq_len]


# Mapping from converter field names to SDK dict key names
_FIELD_NAME_MAP = {
    "advantages": "advantages",
    "log_probs": "logprobs",          # converter name -> SDK dict "logprobs"
    "ref_log_probs": "ref_logprobs",  # converter name -> SDK dict "ref_logprobs"
    # "rollout_log_probs" has no SDK dict equivalent — generation logprobs are
    # computed by the sampling service, not passed in forward_backward requests.
}


def _full_sequence_tokens(datum: Datum) -> torch.Tensor:
    """Full unshifted sequence: appends target_tokens[-1] when the datum is
    pre-shifted (wire format)."""
    tokens = _extract_tokens(datum)
    target_tokens = _extract_target_tokens(datum)
    if target_tokens is not None and len(target_tokens) > 0:
        return torch.cat([tokens, target_tokens[-1:]])
    return tokens


def _place_right_aligned(row: torch.Tensor, values: torch.Tensor, seq_len: int) -> None:
    """Write a 1-D field into row[:seq_len], aligned to the sequence end, so
    position p holds the value for target token x_p (BUG-012)."""
    n = min(len(values), seq_len)
    if n > 0:
        row[seq_len - n:seq_len] = values[-n:]


def _extract_field(datum: Datum, field_name: str) -> Optional[torch.Tensor]:
    """A per-token float tensor from loss_fn_inputs (unresized), or None when
    the client did not send that input."""
    tensor = datum.loss_fn_inputs.get(_FIELD_NAME_MAP.get(field_name, field_name))
    if tensor is None:
        return None
    return torch.tensor(tensor.data, dtype=torch.float32)


def _extract_target_tokens(datum: Datum) -> Optional[torch.Tensor]:
    """loss_fn_inputs["target_tokens"] as a long tensor, or None when absent."""
    tensor = datum.loss_fn_inputs.get("target_tokens")
    if tensor is None:
        return None
    return torch.tensor(tensor.data, dtype=torch.long)


def _extract_sft_weights(datum: Datum) -> Optional[torch.Tensor]:
    """loss_fn_inputs["weights"] (= weights[1:] of the cookbook's SFT datum) as
    a float tensor, or None when absent."""
    tensor = datum.loss_fn_inputs.get("weights")
    if tensor is None:
        return None
    return torch.tensor(tensor.data, dtype=torch.float32)
