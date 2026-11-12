"""A field of a wire model has one name on the wire."""
import inspect

import pytest
from pydantic import BaseModel

from tinkercloud.training.models import requests, responses

WIRE_MODELS = sorted(
    {cls for module in (requests, responses)
     for _, cls in inspect.getmembers(module, inspect.isclass)
     if issubclass(cls, BaseModel) and cls.__module__ == module.__name__},
    key=lambda cls: cls.__qualname__)


@pytest.mark.parametrize("model", WIRE_MODELS, ids=lambda cls: cls.__qualname__)
def test_an_aliased_field_is_not_also_read_by_its_python_name(model):
    """`populate_by_name` on a request model once let a JSON tensor keyed `values` through beside
    `data`: a second wire name that no client was told of and no schema shows."""
    aliased = [name for name, field in model.model_fields.items() if field.alias not in (None, name)]
    by_name = [key for key in ("populate_by_name", "validate_by_name") if key in model.model_config]
    assert not (aliased and by_name), f"{model.__qualname__}: {aliased} are also read by name ({by_name})"


def test_the_guard_sees_the_aliased_tensor():
    """The guard above is empty if no model it walks has an alias: the request tensor has one."""
    assert requests.TensorData in WIRE_MODELS
    assert requests.TensorData.model_fields["values"].alias == "data"
