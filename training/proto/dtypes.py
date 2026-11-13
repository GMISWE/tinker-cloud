"""The tensor dtypes the proto schema carries, in the three spellings the conversions need."""
import numpy as np

from ..core.tensor_values import PROTO_TENSOR_DTYPES
from . import tinker_public_pb2 as pb

DTYPE_TO_PROTO = {"float32": pb.DTYPE_FLOAT32, "int64": pb.DTYPE_INT64}
DTYPE_TO_NUMPY = {"float32": np.float32, "int64": np.int64}
assert set(DTYPE_TO_PROTO) == set(DTYPE_TO_NUMPY) == PROTO_TENSOR_DTYPES
# The SDK collapses request tensors to {float32, int64} before writing.
PROTO_TO_DTYPE = {pb.DTYPE_FLOAT32: "float32", pb.DTYPE_INT64: "int64"}
