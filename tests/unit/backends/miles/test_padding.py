"""Unit tests for DP alignment padding on the miles path.

Padding is what restores ADMISSION for a batch whose sample count is not
divisible by the data-parallel width. Without it those calls deadlock: miles
derives num_steps_per_rollout per rank from that rank's local sample count,
the ranks disagree, and they block on the gradient collective
(specs/008-q3-abstraction-tax/HANDOFF.md).

Padding is only legitimate if it is invisible in two senses, and both are
tested here:
  * semantically — gradients are pure-sum (_loss_norm_total=1), so a pad whose
    loss mask and advantages are zero contributes exactly zero;
  * observationally — a pad has no observation key, so it must sit at the TAIL
    and be truncated off before any client-visible output.
"""

import pytest


def _load_converter_module():
    from tinkercloud.training.backends.miles import converter
    return converter


CONV = _load_converter_module()
pad = CONV.MilesDataConverter.pad_rollout_data_to_dp


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
