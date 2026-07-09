"""
Megatron-Bridge classification converter — Tinker Datum -> the bionemo-recipes
evo2_classifier batch: {input_ids [B,S] long, pool_mask [B,S] float, labels [B]}.

This is what ``classifier_forward_step`` (recipes/evo2_megatron/examples/
evo2_classifier.py) consumes: ``model(input_ids=..., pool_mask=...)`` then CE on
``labels``. Pooling is masked-mean over pool_mask, so pad positions must be 0.
No target_tokens shift (classification reads labels from loss_fn_inputs["labels"]).
See specs/004-bionemo-classification/P5-TINKER-BACKEND.md.
"""
from typing import Any, List

import torch

from ..automodel.converter import (
    ClassificationDataConverter,
    _extract_labels,
    _extract_tokens,
)

PAD_TOKEN_ID = 1  # recipe default (Evo2ClassifierDataset)


class NeMo2ClassificationDataConverter(ClassificationDataConverter):
    """Emits the evo2_classifier batch layout (input_ids / pool_mask / labels)."""

    def _to_recipe_batch(self, data: List, args: Any) -> dict:
        tokens_B = [_extract_tokens(d) for d in data]
        labels_B = [_extract_labels(d) for d in data]
        bsz = len(tokens_B)
        # pad to the model's fixed seq_length if given, else per-batch max
        seq_length = int((args or {}).get("seq_length", 0)) or max(len(t) for t in tokens_B)

        input_ids = torch.full((bsz, seq_length), PAD_TOKEN_ID, dtype=torch.long)
        pool_mask = torch.zeros((bsz, seq_length), dtype=torch.float32)
        for i, tokens in enumerate(tokens_B):
            n = min(len(tokens), seq_length)
            input_ids[i, :n] = tokens[:n] if torch.is_tensor(tokens) else torch.tensor(tokens[:n])
            pool_mask[i, :n] = 1.0

        labels = torch.zeros(bsz, dtype=torch.long)
        for i, lbl in enumerate(labels_B):
            if lbl.numel() > 0:
                labels[i] = lbl.reshape(-1)[0]
        return {"input_ids": input_ids, "pool_mask": pool_mask, "labels": labels}

    def forward_backward_to_backend(self, data: List, loss_fn: str, args: Any) -> Any:
        return self._to_recipe_batch(data, args)

    def forward_to_backend(self, data: List, args: Any) -> Any:
        return self._to_recipe_batch(data, args)
