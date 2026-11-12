"""Whether a value is a numpy array, asked without importing numpy: the gateway process must not
load it (tests/unit/gateway/test_imports.py). This module imports nothing of ours, so the storage,
model and proto packages can each import it first."""
import sys
from typing import Any


def is_array(values: Any) -> bool:
    numpy = sys.modules.get("numpy")   # None: numpy is not loaded here, so no array exists
    return numpy is not None and isinstance(values, numpy.ndarray)
