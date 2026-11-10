"""The verl converter on cross_entropy targets of shape [N] and [N, K]: where
each value lands in the padded tensors, what comes back per datum, and the
shapes it refuses by field name."""
import pytest
import torch

from tinkercloud.training.backends.verl.converter import VerlDataConverter
from tinkercloud.training.models.requests import Datum


def nk_datum(tokens, targets, weights):
    """A cross-entropy datum; `targets` and `weights` are nested lists in wire shape."""
    def wire(values):
        t = torch.tensor(values)
        return {"data": t.reshape(-1).tolist(), "shape": list(t.shape)}
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": wire(targets), "weights": wire(weights)}})


def convert(data, loss_fn="cross_entropy"):
    return VerlDataConverter().forward_backward_to_backend(data, loss_fn, None)


LONG = nk_datum([5, 6, 7], [[10, 11], [20, 21], [30, 31]], [[1.0, -2.0], [0.0, 3.0], [4.0, 5.0]])
SHORT = nk_datum([8, 9], [[40, 41], [50, 51]], [[0.5, 0.25], [-1.0, 0.0]])


def test_nk_values_land_at_their_position_and_column():
    """A flattening converter (the bug this replaces) read [N, K] as N * K positions."""
    out = convert([LONG, SHORT])
    assert out["target_tokens"].tolist() == [[[10, 11], [20, 21], [30, 31]], [[40, 41], [50, 51], [0, 0]]]
    assert out["weights"].tolist() == [[[1.0, -2.0], [0.0, 3.0], [4.0, 5.0]], [[0.5, 0.25], [-1.0, 0.0], [0.0, 0.0]]]
    # N target positions need N + 1 input tokens: the model input plus one more
    assert out["attention_mask"].sum(dim=1).tolist() == [4, 3]
    assert out["input_ids"][:, :3].tolist() == [[5, 6, 7], [8, 9, 51]]


def test_extract_logprobs_returns_nk_in_row_order():
    padded = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)   # [B, Rmax, K]
    long, short = VerlDataConverter.extract_logprobs(padded, [LONG, SHORT])
    assert long["logprobs"]["shape"] == [3, 2] and long["logprobs"]["data"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert short["logprobs"]["shape"] == [2, 2] and short["logprobs"]["data"] == [6.0, 7.0, 8.0, 9.0]


def test_an_n_datum_converts_as_it_did_before_nk():
    datum = nk_datum([5, 6, 7], [6, 7, 9], [1.0, 0.0, 2.0])
    out = convert([datum])
    assert "target_tokens" not in out            # the engine scores the shifted input ids
    assert out["input_ids"].tolist() == [[5, 6, 7, 9]] and out["responses"].tolist() == [[6, 7, 9]]
    assert out["weights"].tolist() == [[1.0, 0.0, 2.0]]
    (result,) = VerlDataConverter.extract_logprobs(torch.tensor([[-1.0, -2.0, -3.0]]), [datum])
    assert result["logprobs"] == {"data": [-1.0, -2.0, -3.0], "dtype": "float32", "shape": [3]}


def test_weights_shaped_unlike_target_tokens_are_refused():
    datum = nk_datum([5, 6, 7], [[10, 11], [20, 21], [30, 31]], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"weights has shape \[3\], target_tokens \[3, 2\]"):
        convert([datum])


@pytest.mark.parametrize("targets, weights", [
    ([[10, 11], [20, 21]], [[1.0, 1.0], [1.0, 1.0]]),   # [2, K] for 3 positions
    ([10, 20], [1.0, 1.0]),                             # [2] for 3 positions
])
def test_a_first_dimension_that_is_not_the_target_count_is_refused(targets, weights):
    with pytest.raises(ValueError, match="target_tokens has 2 rows for 3 target positions"):
        convert([nk_datum([5, 6, 7], targets, weights)])


def test_datums_of_different_k_in_one_call_are_refused():
    wider = nk_datum([8, 9], [[40, 41, 42], [50, 51, 52]], [[1.0] * 3] * 2)
    with pytest.raises(ValueError, match="target_tokens differs across the datums"):
        convert([LONG, wider])


def test_a_rank_above_two_is_refused():
    datum = nk_datum([5, 6], [[[1, 2]], [[3, 4]]], [[[1.0, 1.0]], [[1.0, 1.0]]])
    with pytest.raises(ValueError, match="target_tokens has rank 3"):
        convert([datum])


def test_nk_targets_outside_cross_entropy_are_refused():
    """importance_sampling and ppo score one target per position; [N, K] there would be misread."""
    with pytest.raises(ValueError, match="target_tokens has rank 2.*cross_entropy only, not ppo"):
        convert([LONG], loss_fn="ppo")


def test_a_datum_without_nk_targets_in_an_nk_call_is_refused():
    """Its [N] weights would broadcast across the K columns, or fail in torch with no field named."""
    no_targets = Datum.model_validate({"model_input": {"tokens": [8, 9, 7]},
                                       "loss_fn_inputs": {"weights": {"data": [1.0, 1.0], "shape": [2]}}})
    with pytest.raises(ValueError, match="target_tokens differs across the datums"):
        convert([LONG, no_targets])


def test_nk_weights_without_target_tokens_are_refused():
    datum = Datum.model_validate({"model_input": {"tokens": [8, 9, 7]}, "loss_fn_inputs": {
        "weights": {"data": [1.0] * 4, "shape": [2, 2]}}})
    with pytest.raises(ValueError, match="weights has shape .2, 2. and no target_tokens"):
        convert([datum])
