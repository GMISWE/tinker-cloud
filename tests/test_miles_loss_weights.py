"""Client per-token weights reach Miles beside the 0/1 mask, never inside it."""
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)

from tinkercloud.training.backends.miles.converter import MilesDataConverter
from tinkercloud.training.backends.miles.rollout_data import TinkerDataConverter


def _td(vals, dtype="float32"):
    return {"data": vals, "dtype": dtype, "shape": [len(vals)]}


def _sft_datum(weights, T=6):
    tokens = list(range(10, 10 + T))
    return {"model_input": {"tokens": tokens},
            "loss_fn_inputs": {"weights": _td(weights), "target_tokens": _td([t + 1 for t in tokens], "int64")}}


def test_sft_weights_split_into_binary_mask_and_weights():
    w = [0.0, 0.5, 1.0, 2.0, 0.0, 0.25]
    rd = TinkerDataConverter().forward_backward_to_rollout([_sft_datum(w)], is_rl=False)
    assert rd["loss_masks"][0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert rd["loss_weights"][0].tolist() == w


def test_rl_weights_split_and_trim_together():
    T = 6
    tokens = list(range(10, 10 + T))
    w = [0.0, 0.0, 0.5, 1.0, 1.5, 1.0]
    d = {"model_input": {"tokens": tokens},
         "loss_fn_inputs": {"weights": _td(w), "logprobs": _td([-0.1] * T), "advantages": _td([1.0] * T)}}
    rd = TinkerDataConverter().forward_backward_to_rollout([d], is_rl=True)  # no target: causal trim
    assert rd["response_lengths"] == [T - 1]
    assert rd["loss_masks"][0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert rd["loss_weights"][0].tolist() == w[:-1]


def test_rl_without_weights_gets_unit_weights():
    T = 4
    d = {"model_input": {"tokens": list(range(T))},
         "loss_fn_inputs": {"logprobs": _td([-0.1] * T), "advantages": _td([1.0] * T),
                            "target_tokens": _td(list(range(1, T + 1)), "int64")}}
    rd = TinkerDataConverter().forward_backward_to_rollout([d], is_rl=True)
    assert rd["loss_weights"][0].tolist() == [1.0] * T
    assert rd["loss_masks"][0].tolist() == [1.0] * T


def test_forward_path_carries_weights_beside_mask():
    w = [0.0, 0.5, 1.0, 0.0, 2.0, 1.0]
    rd = TinkerDataConverter().forward_to_rollout([_sft_datum(w)])
    assert rd["loss_masks"][0].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert rd["loss_weights"][0].tolist() == w


def test_pad_zeroes_weights_and_merge_concatenates():
    rd = TinkerDataConverter().forward_backward_to_rollout([_sft_datum([1.0] * 6)] * 3, is_rl=False)
    rd["dynamic_global_batch_size"] = 3
    n_pad = MilesDataConverter.pad_rollout_data_to_dp(rd, 2)
    assert n_pad == 1 and len(rd["loss_weights"]) == 4
    assert rd["loss_weights"][-1].tolist() == [0.0] * 6 and rd["loss_masks"][-1].tolist() == [0.0] * 6
    a = TinkerDataConverter().forward_backward_to_rollout([_sft_datum([0.5] * 6)], is_rl=False)
    b = TinkerDataConverter().forward_backward_to_rollout([_sft_datum([2.0] * 6)] * 2, is_rl=False)
    a["dynamic_global_batch_size"], b["dynamic_global_batch_size"] = 1, 2
    m = MilesDataConverter.merge_forward_backward_batches([a, b])
    assert [x[0].item() for x in m["loss_weights"]] == [0.5, 2.0, 2.0]
