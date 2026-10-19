"""Unit tests for RequestValidator's DP-divisibility contract.

The load-bearing case: an indivisible batch must be REJECTED, not attempted,
and not waivable by ALLOW_PARTIAL_BATCHES. Miles derives num_steps_per_rollout
per rank from that rank's local sample count, so an unbalanced split gives the
ranks different micro-step counts and they deadlock on the gradient collective
-- the future never resolves and the actors stay wedged until restart.
Measured 2026-08-04 with 3 samples at dp=2; see
specs/008-q3-abstraction-tax/HANDOFF.md.

The flag still governs the genuinely partial cases (RL group alignment), where
the consequence is normalization rather than liveness.
"""
from argparse import Namespace

import pytest

from tinkercloud.training.core.validators import RequestValidator


def _args(dp=2, gbs=8, balance_data=False, n_per_prompt=1):
    return Namespace(
        data_parallel_size=dp,
        global_batch_size=gbs,
        balance_data=balance_data,
        n_samples_per_prompt=n_per_prompt,
    )


def _v(allow_partial=False, **kw):
    return RequestValidator(_args(**kw), allow_partial_batches=allow_partial)


@pytest.mark.parametrize("allow_partial", [False, True])
@pytest.mark.parametrize("n", [3, 5, 7, 9])
def test_indivisible_rejected_regardless_of_partial_flag(n, allow_partial):
    """The regression this file exists for: dp-indivisible => reject, always.

    Before the fix, allow_partial_batches waived this and the run deadlocked.
    """
    err = _v(allow_partial=allow_partial, dp=2).validate_sample_count(n)
    assert err is not None, f"{n} samples at dp=2 must be rejected"
    assert "divisible by 2" in err
    assert "deadlock" in err, "the error must say WHY, since the old failure was a hang"


@pytest.mark.parametrize("allow_partial", [False, True])
@pytest.mark.parametrize("n", [2, 4, 6, 8, 128])
def test_divisible_accepted(n, allow_partial):
    assert _v(allow_partial=allow_partial, dp=2).validate_sample_count(n) is None


def test_indivisible_at_wider_dp():
    """dp=4: 100 samples splits 25/25/25/25 (ok) but 102 does not."""
    assert _v(dp=4).validate_sample_count(100) is None
    err = _v(dp=4, allow_partial=True).validate_sample_count(102)
    assert err is not None and "divisible by 4" in err


def test_below_dp_size_still_rejected():
    """Check 1 was already unconditional; keep it that way."""
    err = _v(allow_partial=True, dp=4).validate_sample_count(3)
    assert err is not None and "At least 4 samples" in err


def test_suggestions_are_actually_valid():
    """A suggestion that is itself indivisible would be worse than none."""
    err = _v(dp=4).validate_sample_count(10)
    assert err is not None
    line = next(s for s in err.splitlines() if s.startswith("Suggestion:"))
    suggested = [int(t.strip(".,")) for t in line.split() if t.strip(".,").isdigit()]
    assert suggested, f"no numeric suggestion in {line!r}"
    for s in suggested:
        assert s % 4 == 0, f"suggested {s} is not divisible by dp=4"


def test_partial_flag_still_waives_rl_group_alignment():
    """The flag must keep its legitimate purpose, or we have over-corrected.

    RL group alignment is a normalization concern, not a liveness one, so it
    stays waivable -- unlike dp divisibility.
    """
    kw = dict(dp=2, balance_data=True, n_per_prompt=4)
    # 8 is divisible by dp=2 but not by (n_per_prompt * dp) = 8 -> it is, use 12
    assert _v(allow_partial=False, **kw).validate_sample_count(12, is_rl=True) is not None
    assert _v(allow_partial=True, **kw).validate_sample_count(12, is_rl=True) is None


def test_global_batch_size_mismatch_is_a_warning_not_an_error():
    """Check 3 only warns: gradient accumulation is well-defined here."""
    assert _v(dp=2, gbs=4096).validate_sample_count(128) is None
