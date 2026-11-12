"""A refused classification label is reported by its datum, its type and the count of values, never by
a value the client sent: a label can be megabytes, and the message goes to the client and the log."""
import pytest

from tinkercloud.training.backends.megatron_bridge.converter import NeMo2ClassificationDataConverter
from tinkercloud.training.backends.objectives import CLASSIFICATION_CE
from tinkercloud.training.models.requests import Datum


def _refusal(labels) -> str:
    datum = Datum.model_validate({"model_input": {"tokens": [5, 6, 7]}, "loss_fn_inputs": {"labels": {"data": labels}}})
    with pytest.raises(ValueError) as caught:
        NeMo2ClassificationDataConverter().forward_backward_to_backend([datum], CLASSIFICATION_CE, {"num_labels": 2})
    return str(caught.value)


@pytest.mark.parametrize("labels, said, hidden", [
    (["A" * 1_000_000], "a str", "AAAA"),            # one value of the wrong type: echoed whole before
    ([[0] * 5000], "a list", "0, 0"),                # one nested list: echoed whole before
    ([10 ** 400, 0], "2 value(s)", "1000000000"),    # several values: the first was echoed whole before
])
def test_a_refused_label_is_reported_without_its_value(labels, said, hidden):
    """The message embedded `repr` of the value: megabytes for a long string, a nested list, or a huge first label."""
    message = _refusal(labels)
    assert message.startswith("datum 0: ") and said in message
    assert hidden not in message and len(message) < 300


def test_a_fraction_is_named_as_one():
    """With the value gone from the message, "a float" alone would not say why 1.9 is refused and 1.0 is not."""
    assert "a float that is not whole" in _refusal([1.9])
