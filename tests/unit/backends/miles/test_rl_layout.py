"""Miles RL datum layout: the response is exactly the T wire targets, tensors untouched."""
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

from tinkercloud.training.backends.miles.rollout_data import TinkerDataConverter
from tinkercloud.training.models.requests import Datum


def _datum(T=6, with_target=True):
    tokens = list(range(10, 10 + T))
    target = [t + 1 for t in tokens]
    inputs = {
        "weights": {"data": [0.0, 0.0] + [1.0] * (T - 2), "dtype": "float32", "shape": [T]},
        "logprobs": {"data": [-0.1 * (k + 1) for k in range(T)], "dtype": "float32", "shape": [T]},
        "advantages": {"data": [float(k) for k in range(T)], "dtype": "float32", "shape": [T]},
    }
    if with_target:
        inputs["target_tokens"] = {"data": target, "dtype": "int64", "shape": [T]}
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": inputs}), tokens, target


def test_rl_datum_appends_final_target_and_keeps_tensors_aligned():
    d, tokens, target = _datum()
    rd = TinkerDataConverter().forward_backward_to_rollout([d], is_rl=True)
    assert rd["tokens"][0].tolist() == tokens + [target[-1]]
    assert rd["response_lengths"] == [len(tokens)]
    assert rd["advantages"][0].tolist() == [float(k) for k in range(len(tokens))]
    assert rd["log_probs"][0].tolist() == pytest.approx([-0.1 * (k + 1) for k in range(len(tokens))])
    assert rd["loss_masks"][0].tolist() == [0.0, 0.0] + [1.0] * (len(tokens) - 2)


def test_rl_datum_without_target_drops_the_last_entry():
    d, tokens, _ = _datum(with_target=False)
    rd = TinkerDataConverter().forward_backward_to_rollout([d], is_rl=True)
    assert rd["tokens"][0].tolist() == tokens
    assert rd["response_lengths"] == [len(tokens) - 1]
    assert rd["advantages"][0].tolist() == [float(k) for k in range(len(tokens) - 1)]  # drop-last, not drop-first
