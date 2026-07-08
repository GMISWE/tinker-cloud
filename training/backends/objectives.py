"""
Training-objective axis for the backend abstraction (feature 004).

Additive: language_modeling is the default and leaves existing behavior
unchanged. See specs/004-bionemo-classification/plan.md ("Architecture:
additive OBJECTIVE axis").
"""
from enum import Enum


class Objective(str, Enum):
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"


# Loss-registry name for classification cross-entropy (plan.md:113).
CLASSIFICATION_CE = "classification_ce"

CLASSIFICATION_OBJECTIVES = frozenset({
    Objective.SEQUENCE_CLASSIFICATION,
    Objective.TOKEN_CLASSIFICATION,
})


def is_classification(objective) -> bool:
    """True if objective is a classification variant (seq- or token-cls)."""
    return Objective(objective) in CLASSIFICATION_OBJECTIVES
