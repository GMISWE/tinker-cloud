"""
Training-objective axis for the backend abstraction (feature 004).

Additive: language_modeling is the default and leaves existing behavior
unchanged. See specs/004-bionemo-classification/plan.md ("Architecture:
additive OBJECTIVE axis").
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class Objective(str, Enum):
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"


# Loss-registry name for classification cross-entropy.
CLASSIFICATION_CE = "classification_ce"

CLASSIFICATION_OBJECTIVES = frozenset({
    Objective.SEQUENCE_CLASSIFICATION,
    Objective.TOKEN_CLASSIFICATION,
})


def is_classification(objective) -> bool:
    """True if objective is a classification variant (seq- or token-cls)."""
    return Objective(objective) in CLASSIFICATION_OBJECTIVES


@dataclass(frozen=True)
class ClassificationSpec:
    """The head a base model's own config declares (specs/025 single-source rule)."""
    objective: Objective
    num_labels: int


_HEAD_ARCHITECTURES = {
    "ForSequenceClassification": Objective.SEQUENCE_CLASSIFICATION,
    "ForTokenClassification": Objective.TOKEN_CLASSIFICATION,
}


def classification_spec(raw_config: Dict[str, Any]) -> Optional[ClassificationSpec]:
    """None for a language model. For a classification architecture the label
    count must be declared (num_labels or id2label in the file): transformers'
    default of 2 is not a declaration, which is why the caller passes the raw
    config.json rather than a loaded AutoConfig."""
    architectures = raw_config.get("architectures") or []
    heads = {
        objective
        for arch in architectures
        for suffix, objective in _HEAD_ARCHITECTURES.items()
        if arch.endswith(suffix)
    }
    if not heads:
        return None
    if len(heads) > 1:
        raise ValueError(f"architectures {architectures} declare more than one head type")
    objective = heads.pop()
    if "num_labels" in raw_config:
        num_labels = int(raw_config["num_labels"])
    elif isinstance(raw_config.get("id2label"), dict) and raw_config["id2label"]:
        num_labels = len(raw_config["id2label"])
    else:
        raise ValueError(
            f"architectures {architectures} declare a classification head but the "
            f"config.json has neither num_labels nor id2label"
        )
    if num_labels < 2:
        raise ValueError(f"num_labels must be >= 2 for classification, got {num_labels}")
    return ClassificationSpec(objective=objective, num_labels=num_labels)
