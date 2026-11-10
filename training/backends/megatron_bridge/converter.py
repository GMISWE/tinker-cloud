"""
Megatron-Bridge classification converter — Tinker Datum -> the evo2_classifier
recipe batch {input_ids, pool_mask, labels}.

Unlike the language-modeling converters there is NO target_tokens = tokens[1:] shift:
classification consumes labels directly from loss_fn_inputs["labels"].
"""
from typing import Any, Dict, List

import torch

from ..base import DataConverter
from ...models.requests import Datum

PAD_TOKEN_ID = 1  # recipe default (Evo2ClassifierDataset)


class NeMo2ClassificationDataConverter(DataConverter):
    """Emits the evo2_classifier batch layout (input_ids / pool_mask / labels)."""

    def forward_backward_to_backend(self, data: List, loss_fn: str, args: Any) -> Any:
        # The label is the training target, so each datum must carry exactly one
        # valid class: nothing is filled in, cut or rounded. args["num_labels"] is
        # the model's class count. The backend always passes it, so a caller that
        # does not gets a KeyError and not a range that is silently unchecked.
        num_labels = int(args["num_labels"])
        classes = [_class_label(i, d, num_labels) for i, d in enumerate(data)]
        return self._to_recipe_batch(data, args, torch.tensor(classes, dtype=torch.long))

    def forward_to_backend(self, data: List, args: Any) -> Any:
        # Forward-only inputs may carry no label; the first element is taken when
        # there is one, and 0 otherwise. This is separate from training on purpose.
        labels = torch.zeros(len(data), dtype=torch.long)
        for i, lbl in enumerate(_extract_labels(d) for d in data):
            if lbl.numel() > 0:
                labels[i] = lbl.reshape(-1)[0]
        return self._to_recipe_batch(data, args, labels)

    def _to_recipe_batch(self, data: List, args: Any, labels: torch.Tensor) -> dict:
        tokens_B = [_extract_tokens(d) for d in data]
        bsz = len(tokens_B)
        # pad to the model's fixed seq_length if given, else per-batch max
        seq_length = int((args or {}).get("seq_length", 0)) or max(len(t) for t in tokens_B)
        input_ids = torch.full((bsz, seq_length), PAD_TOKEN_ID, dtype=torch.long)
        pool_mask = torch.zeros(bsz, seq_length, dtype=torch.float32)
        for i, tokens in enumerate(tokens_B):
            n = min(len(tokens), seq_length)
            input_ids[i, :n] = tokens[:n] if torch.is_tensor(tokens) else torch.tensor(tokens[:n])
            pool_mask[i, :n] = 1.0
        return {"input_ids": input_ids, "pool_mask": pool_mask, "labels": labels}

    def backend_to_forward_result(
        self, result: Any, data: List[Datum],
    ) -> Dict[str, Any]:
        """Convert classification forward result (logits) to the SDK's
        ForwardBackwardOutput shape: loss_fn_outputs[i]["logits"] is a
        TensorData dict ({data, shape, dtype}) of per-position class logits."""
        logits = result["logits"]
        outputs = []
        if logits is not None:
            for i in range(len(data)):
                row = logits[i].detach().cpu().float()      # [S, C]
                outputs.append({"logits": {
                    "data": row.flatten().tolist(),
                    "shape": list(row.shape),
                    "dtype": "float32",
                }})
        return {
            "loss_fn_output_type": "classification",
            "loss_fn_outputs": outputs,
            "metrics": {},
        }

    def backend_to_forward_backward_result(
        self, result: Any, data: List[Datum],
    ) -> Dict[str, Any]:
        """Convert classification training result to the SDK's
        ForwardBackwardOutput shape. metrics stays EMPTY: the SDK combiner
        weights metrics by len(loss_fn_outputs) (0 here) and would zero-divide
        on a non-empty metrics dict. `loss` is kept as an extra key (SDK ignores
        it) for the raw-HTTP path."""
        if result is None:
            return {"loss": None, "loss_fn_output_type": "classification_ce",
                    "loss_fn_outputs": [], "metrics": {}, "deferred": True}
        loss = _to_python_scalar(result.get("loss", 0.0))
        return {
            "loss": loss,
            "loss_fn_output_type": "classification_ce",
            "loss_fn_outputs": [],
            "metrics": {},
            "deferred": False,
        }


def _to_python_scalar(val):
    if isinstance(val, (int, float)):
        return val
    if hasattr(val, "item"):
        return val.item()
    return float(val)


def _extract_tokens(datum: Datum) -> torch.Tensor:
    """Token IDs from model_input: text chunks concatenated, else tokens / input_ids."""
    mi = datum.model_input
    if mi.chunks:
        parts = [_as_long(c.tokens) for c in mi.chunks if c.type != "image" and c.tokens is not None]
        if parts:
            return torch.cat(parts)
    if mi.tokens is not None:
        return _as_long(mi.tokens)
    if mi.input_ids is not None:
        return _as_long(mi.input_ids)
    raise ValueError("model_input carries no chunks, tokens or input_ids")


def _extract_labels(datum: Datum) -> torch.Tensor:
    """loss_fn_inputs["labels"], or an empty tensor when the datum carries none
    (forward-only inputs may)."""
    labels = datum.loss_fn_inputs.get("labels")
    if labels is None:
        return torch.zeros(0, dtype=torch.long)
    return _as_long(labels.data)


def _class_label(index: int, datum: Datum, num_labels: int) -> int:
    """The one class of a sequence-classification datum: exactly one whole number in
    loss_fn_inputs["labels"], inside [0, num_labels). A bool is not a class, and the
    error names the datum and says what was found, not the label list itself."""
    labels = datum.loss_fn_inputs.get("labels")
    values = [] if labels is None else labels.data
    if len(values) != 1:
        raise ValueError(
            f"datum {index}: classification training needs exactly one integer class in "
            f"loss_fn_inputs['labels'], got {len(values)} value(s)"
            + (f" starting with {values[0]!r}" if values else ""))
    value = values[0]
    # int is checked before float, so an integer too large for a float is a class out of range and not an OverflowError.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            isinstance(value, float) and not value.is_integer()):
        raise ValueError(
            f"datum {index}: classification training needs a whole number as the class, got {type(value).__name__} {value!r}")
    label = int(value)
    if not 0 <= label < num_labels:
        shown = str(label) if len(str(label)) <= 20 else f"{str(label)[:20]}... ({len(str(label))} digits)"
        raise ValueError(f"datum {index}: class {shown} is outside [0, {num_labels}) for this model")
    return label


def _as_long(val) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().long().reshape(-1)
    return torch.tensor(val, dtype=torch.long).reshape(-1)
