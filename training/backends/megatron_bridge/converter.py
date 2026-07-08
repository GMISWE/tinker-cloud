"""
Megatron-Bridge classification converter.

Reuses the shared ClassificationDataConverter contract (input_ids,
attention_mask, labels) from the automodel package (Constitution P1/P4).
Megatron/NeMo2 may need a distinct batch layout (packed THD, position ids,
NeMo2 microbatch dicts) — NeMo2ClassificationDataConverter is the stub seam
for that variant.
See specs/004-bionemo-classification/plan.md.
"""
from typing import Any, List

from ..automodel.converter import ClassificationDataConverter


class NeMo2ClassificationDataConverter(ClassificationDataConverter):
    """NeMo2-batch variant of the shared classification converter (scaffold).

    Inherits the {input_ids, attention_mask, labels} contract; override to emit
    NeMo2/Megatron microbatch dicts when the Evo2/StripedHyena path lands.
    """

    def forward_backward_to_backend(
        self, data: List, loss_fn: str, args: Any,
    ) -> Any:
        # TODO(004-P5): emit NeMo2 microbatch layout (packed THD / position_ids)
        # for Megatron-Bridge + StripedHyena. Falls back to the HF-style batch.
        return super().forward_backward_to_backend(data, loss_fn, args)
