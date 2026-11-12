"""A wire tensor's values as they arrived or as a backend returned them: a list, or a 1-D numpy array
whose bytes are never spelled out as Python numbers. This module does not import numpy: the gateway
process imports the models and must not load it (tests/unit/gateway/test_imports.py). An array can
exist only where numpy already is."""
from typing import Annotated, Any, Iterable, Sequence, Tuple

from pydantic import PlainValidator, TypeAdapter, WithJsonSchema

from ..core.array_values import is_array


# The dtypes a wire tensor may state: the ones the SDK's proto messages carry.
PROTO_TENSOR_DTYPES = frozenset({"float32", "int64"})
DEFAULT_TENSOR_DTYPE = "float32"   # of a result tensor that states none


def batched_layout(name: str, tensors: Iterable[Tuple[str, Sequence[int]]]) -> Tuple[str, Tuple[int, ...]]:
    """The (dtype, trailing shape) under which result field `name` of several datums, each given as
    (dtype, shape), is one batched proto tensor: datums may differ in the leading dimension only.

    Raises:
        ValueError: the datums state more than one dtype or trailing shape, or a dtype the proto lacks.
    """
    dtypes, trailing = zip(*((dtype, tuple(shape[1:])) for dtype, shape in tensors), strict=True)
    if len(set(dtypes)) > 1:
        raise ValueError(f"field {name!r} has mixed dtypes across datums: {sorted(set(dtypes))}")
    if dtypes[0] not in PROTO_TENSOR_DTYPES:
        raise ValueError(f"field {name!r}: dtype {dtypes[0]!r} has no proto value")
    if len(set(trailing)) > 1:
        raise ValueError(f"field {name!r} has mixed trailing shapes across datums: {sorted(set(trailing))}")
    return dtypes[0], trailing[0]


def list_or_array(list_type: Any) -> Any:
    """A field type: a value validated as `list_type`, or a numpy array taken as it is."""
    adapter = TypeAdapter(list_type)

    def validate(values: Any) -> Any:
        return values if is_array(values) else adapter.validate_python(values)

    return Annotated[Any, PlainValidator(validate), WithJsonSchema(adapter.json_schema())]
