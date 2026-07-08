"""
Classification data converter — Tinker Datum -> {input_ids, attention_mask,
labels} batch dict for HF-encoder classification (ESM2, Nucleotide Transformer).

Unlike the language-modeling converters there is NO target_tokens = tokens[1:] shift:
classification consumes labels directly from loss_fn_inputs["labels"].

  seq-cls:   labels [B]      (one class id per sample)
  token-cls: labels [B, S]   (per-token class id, -100 ignore on pads)

Shared contract for both classification backends (Constitution P1/P4). The
megatron_bridge backend subclasses this for a NeMo2-batch variant.
See specs/004-bionemo-classification/plan.md.
"""
import logging
from typing import Any, Dict, List

import torch

from ..base import DataConverter
from ..objectives import Objective

logger = logging.getLogger(__name__)

# HF cross-entropy ignore index for masked / padded token positions.
IGNORE_INDEX = -100


class ClassificationDataConverter(DataConverter):
    """Converts Tinker classification Datums to a padded HF batch dict."""

    def forward_to_backend(self, data: List[Dict], args: Any) -> Any:
        """Forward-only batch (input_ids + attention_mask; labels if present)."""
        return self._to_batch(data, args)

    def forward_backward_to_backend(
        self, data: List[Dict], loss_fn: str, args: Any,
    ) -> Any:
        """Training batch: input_ids, attention_mask, labels."""
        return self._to_batch(data, args)

    def _to_batch(self, data: List[Dict], args: Any) -> Dict[str, torch.Tensor]:
        if not data:
            return {
                "input_ids": torch.zeros(0, 0, dtype=torch.long),
                "attention_mask": torch.zeros(0, 0, dtype=torch.long),
                "labels": torch.zeros(0, dtype=torch.long),
            }

        tokens_B = [_extract_tokens(d) for d in data]
        labels_B = [_extract_labels(d) for d in data]

        max_seq_len = max(len(t) for t in tokens_B)
        batch_size = len(data)

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for i, tokens in enumerate(tokens_B):
            seq_len = len(tokens)
            input_ids[i, :seq_len] = tokens
            attention_mask[i, :seq_len] = 1

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Forward-only inputs may carry no labels — emit input_ids/attention_mask
        # only so a logits pass doesn't require a labels tensor.
        if not any(lbl.numel() > 0 for lbl in labels_B):
            return batch

        token_cls = _is_token_classification(args, tokens_B, labels_B)
        if token_cls:
            labels = torch.full(
                (batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long,
            )
        else:
            labels = torch.zeros(batch_size, dtype=torch.long)

        for i, (tokens, lbl) in enumerate(zip(tokens_B, labels_B)):
            if lbl.numel() == 0:
                continue
            if token_cls:
                n = min(len(lbl), len(tokens))
                labels[i, :n] = lbl[:n]
            else:
                # seq-cls: single class id (scalar or length-1 tensor).
                labels[i] = lbl.reshape(-1)[0]

        batch["labels"] = labels
        return batch

    def backend_to_forward_result(
        self, result: Any, data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert classification forward result (logits) to Tinker format."""
        logits = result.get("logits") if hasattr(result, "get") else result
        outputs = []
        if logits is not None:
            for i in range(len(data)):
                row = logits[i].detach().cpu()
                outputs.append({"logits": row.tolist()})
        return {"loss_fn_outputs": outputs, "metrics": {}}

    def backend_to_forward_backward_result(
        self, result: Any, data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert classification training result (loss/metrics) to Tinker format."""
        if result is None:
            return {"loss": None, "metrics": {}, "deferred": True, "loss_fn_outputs": []}
        loss = _to_python_scalar(result.get("loss", 0.0))
        return {
            "loss": loss,
            "metrics": {"total_loss": loss},
            "deferred": False,
            "loss_fn_outputs": [],
        }


def _is_token_classification(args: Any, tokens_B, labels_B) -> bool:
    """Prefer explicit objective from args; else infer from label shape."""
    objective = args.get("objective") if isinstance(args, dict) else None
    if objective is not None:
        return Objective(objective) == Objective.TOKEN_CLASSIFICATION
    # Fallback inference: per-token labels are as long as their sequence.
    return any(len(lbl) > 1 for lbl in labels_B)


def _to_python_scalar(val):
    if isinstance(val, (int, float)):
        return val
    if hasattr(val, "item"):
        return val.item()
    return float(val)


def _get(obj, field):
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _extract_tokens(datum) -> torch.Tensor:
    """Token IDs from model_input (chunks / tokens / input_ids) or flat datum."""
    model_input = _get(datum, "model_input")
    if model_input is not None:
        chunks = _get(model_input, "chunks")
        if chunks:
            all_tokens = []
            for chunk in chunks:
                if _get(chunk, "type") == "image":
                    continue
                toks = _get(chunk, "tokens")
                if toks is not None:
                    all_tokens.append(_as_long(toks))
            if all_tokens:
                return torch.cat(all_tokens)
        for key in ("tokens", "input_ids"):
            toks = _get(model_input, key)
            if toks is not None:
                return _as_long(toks)
    toks = _get(datum, "tokens")
    if toks is None:
        toks = _get(datum, "input_ids") or []
    return _as_long(toks)


def _extract_labels(datum) -> torch.Tensor:
    """Labels from loss_fn_inputs["labels"] (TensorData or raw), or flat datum."""
    loss_fn_inputs = _get(datum, "loss_fn_inputs")
    labels = None
    if loss_fn_inputs is not None:
        obj = _get(loss_fn_inputs, "labels")
        if obj is not None:
            labels = _tensor_data(obj)
    if labels is None:
        labels = _get(datum, "labels")
    if labels is None:
        return torch.zeros(0, dtype=torch.long)
    return _as_long(labels)


def _tensor_data(obj):
    """Unwrap a TensorData-like value: Pydantic (.data), dict ({"data": ...}), raw."""
    if hasattr(obj, "data"):
        return obj.data
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    return obj


def _as_long(val) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().long().reshape(-1)
    return torch.tensor(val, dtype=torch.long).reshape(-1)
