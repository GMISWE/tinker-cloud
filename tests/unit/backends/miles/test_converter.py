"""Miles converter and rollout_data: DP-alignment padding, per-token weights beside the 0/1 mask, and the RL datum layout.

Padding restores admission for a batch whose sample count is not divisible by the data-parallel width; without it the ranks
disagree on num_steps_per_rollout and deadlock on the gradient collective (specs/008-q3-abstraction-tax/HANDOFF.md). A pad
must be invisible twice over: semantically (zero loss mask and advantages contribute exactly zero under the pure-sum
gradient) and observationally (no observation key, so it sits at the tail and is truncated off before any client output).
Client per-token weights reach Miles beside the 0/1 mask, never inside it. An RL datum's response is exactly the T wire
targets, tensors untouched."""
import pytest

from tinkercloud.training.backends.miles.converter import MilesDataConverter
from tinkercloud.training.backends.miles.rollout_data import TinkerDataConverter
from tinkercloud.training.models.requests import Datum


pad = MilesDataConverter.pad_rollout_data_to_dp


def _rollout(n, tok_len=4, slot=None):
    """A rollout_data shaped like the miles SFT path produces."""
    d = {
        "tokens": [[10 + i] * tok_len for i in range(n)],
        "loss_masks": [[1] * tok_len for _ in range(n)],
        "advantages": [[0.5] * tok_len for _ in range(n)],
        "log_probs": [[-0.1] * tok_len for _ in range(n)],
        "response_lengths": [tok_len for _ in range(n)],
        "dynamic_global_batch_size": n,
        "_loss_norm_total": 1,
        "_loss_type_override": "sft_loss",
    }
    if slot is not None:
        d["adapter_slots"] = [slot] * n
    return d


@pytest.mark.parametrize("n,dp,want", [(8, 2, 0), (4, 2, 0), (3, 2, 1), (1, 2, 1),
                                       (5, 4, 3), (100, 8, 4), (7, 1, 0), (6, 3, 0)])
def test_pad_count(n, dp, want):
    d = _rollout(n)
    assert pad(d, dp) == want
    assert d["dynamic_global_batch_size"] == n + want
    assert (n + want) % dp == 0 or dp <= 1


def test_pads_are_inert():
    """The whole justification: a pad must add exactly zero to the gradient."""
    d = _rollout(3)
    before_mask = sum(sum(m) for m in d["loss_masks"])
    before_adv = sum(sum(a) for a in d["advantages"])
    pad(d, 2)
    assert sum(sum(m) for m in d["loss_masks"]) == before_mask, "pad changed loss mass"
    assert sum(sum(a) for a in d["advantages"]) == before_adv, "pad changed advantage mass"
    assert all(v == 0 for v in d["loss_masks"][-1])
    assert all(v == 0 for v in d["advantages"][-1])


def test_pads_are_at_the_tail_and_reals_untouched():
    """Truncation by slice is only valid if pads never land mid-batch."""
    d = _rollout(3)
    reals = [list(t) for t in d["tokens"]]
    pad(d, 2)
    assert d["tokens"][:3] == reals
    assert len(d["tokens"]) == 4


def test_pad_is_well_formed_not_empty():
    """A pad still has to be a valid sample: real token ids, matching lengths."""
    d = _rollout(3, tok_len=5)
    pad(d, 2)
    for key in ("tokens", "loss_masks", "advantages", "log_probs"):
        assert len(d[key]) == 4
        assert len(d[key][-1]) == 5, f"{key} pad has wrong token length"
    assert d["response_lengths"][-1] == 5


def test_pad_does_not_alias_source_sample():
    d = _rollout(3)
    pad(d, 2)
    d["tokens"][-1][0] = 999
    assert d["tokens"][2][0] != 999, "pad aliases the last real sample's storage"


def test_scalars_and_non_per_sample_keys_untouched():
    d = _rollout(3)
    pad(d, 2)
    assert d["_loss_norm_total"] == 1
    assert d["_loss_type_override"] == "sft_loss"


def test_adapter_slots_extended_for_pool_routing():
    """Pool mode routes by slot; a pad with no slot would be unroutable."""
    d = _rollout(3, slot=7)
    pad(d, 2)
    assert len(d["adapter_slots"]) == 4
    assert d["adapter_slots"][-1] == 7


def test_idempotent_once_aligned():
    d = _rollout(3)
    assert pad(d, 2) == 1
    assert pad(d, 2) == 0, "second pad must be a no-op"
    assert d["dynamic_global_batch_size"] == 4


def test_per_sample_key_rule_matches_merge():
    """Padding and merging must agree on what a per-sample key is, or a
    co-batched request would be sliced across a pad."""
    d = _rollout(3)
    n = d["dynamic_global_batch_size"]
    per_sample = {k for k, v in d.items() if isinstance(v, list) and len(v) == n}
    pad(d, 2)
    for k in per_sample:
        assert len(d[k]) == 4, f"per-sample key {k} not padded"


def test_dp1_is_noop():
    d = _rollout(3)
    assert pad(d, 1) == 0
    assert d["dynamic_global_batch_size"] == 3


def _td(vals, dtype="float32"):
    return {"data": vals, "dtype": dtype, "shape": [len(vals)]}


def _sft_datum(weights, T=6):
    tokens = list(range(10, 10 + T))
    return Datum.model_validate({"model_input": {"tokens": tokens},
            "loss_fn_inputs": {"weights": _td(weights), "target_tokens": _td([t + 1 for t in tokens], "int64")}})


def test_sft_weights_split_into_binary_mask_and_weights():
    w = [0.0, 0.5, 1.0, 2.0, 0.0, 0.25]
    rd = TinkerDataConverter().forward_backward_to_rollout([_sft_datum(w)], is_rl=False)
    assert rd["loss_masks"][0].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert rd["loss_weights"][0].tolist() == w


def test_rl_weights_split_and_trim_together():
    T = 6
    tokens = list(range(10, 10 + T))
    w = [0.0, 0.0, 0.5, 1.0, 1.5, 1.0]
    d = Datum.model_validate({"model_input": {"tokens": tokens},
         "loss_fn_inputs": {"weights": _td(w), "logprobs": _td([-0.1] * T), "advantages": _td([1.0] * T)}})
    rd = TinkerDataConverter().forward_backward_to_rollout([d], is_rl=True)  # no target: causal trim
    assert rd["response_lengths"] == [T - 1]
    assert rd["loss_masks"][0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert rd["loss_weights"][0].tolist() == w[:-1]


def test_rl_without_weights_gets_unit_weights():
    T = 4
    d = Datum.model_validate({"model_input": {"tokens": list(range(T))},
         "loss_fn_inputs": {"logprobs": _td([-0.1] * T), "advantages": _td([1.0] * T),
                            "target_tokens": _td(list(range(1, T + 1)), "int64")}})
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
