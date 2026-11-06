"""gspo against Tinker's reference implementation (tinker-docs /tinker/losses/gspo/),
transcribed per datum below: value, gradient and the two diagnostics."""
import math

import pytest
import torch

from tinkercloud.training.backends.automodel.converter import coalesce, convert_request
from tinkercloud.training.backends.automodel.losses import compute_loss
from tinkercloud.training.core.loss_registry import clip_thresholds
from tests.datums import rl_datum

from .test_losses import Toy, reference_lp


def tinker_gspo(target_logprobs, sampling_logprobs, advantages, sequence_ratio_mask, low, high):
    """Tinker's published code for one datum."""
    in_ratio = sequence_ratio_mask != 0
    log_ratio = (target_logprobs.detach() - sampling_logprobs)[in_ratio].mean()
    ratio = torch.exp(log_ratio + target_logprobs - target_logprobs.detach())
    clipped_ratio = torch.clamp(ratio, low, high)
    gspo_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    s = float(torch.exp(log_ratio))
    return -gspo_objective.sum(), float(log_ratio), (s < low or s > high)


def _rows(seed):
    """Rows of different lengths; one marks a single token; one has a prompt with A=0."""
    g = torch.Generator().manual_seed(seed)
    shapes = [(6, [0, 0, 1, 1, 1, 1]), (3, [0, 0, 1]), (5, [1, 1, 1, 1, 1])]
    rows = []
    for n, mask in shapes:
        tokens = torch.randint(0, 11, (n,), generator=g).tolist()
        targets = torch.randint(0, 11, (n,), generator=g).tolist()
        adv = [0.0 if m == 0 else float(a) for m, a in zip(mask, torch.randn(n, generator=g))]
        rows.append((tokens, targets, adv, [float(m) for m in mask]))
    return rows


@pytest.mark.parametrize("shift", [0.0, 2e-4, -5e-3])   # inside the default range, at its edge, far outside
def test_gspo_matches_tinkers_reference(shift):
    m = Toy(seed=4)
    rows = _rows(7)
    with torch.no_grad():
        lps = [reference_lp(m(torch.tensor([t])), torch.tensor([y]))[0] for t, y, _, _ in rows]
    q = [(lp - shift).tolist() for lp in lps]
    datums = [rl_datum(t, y, qi, a, sequence_ratio_mask=mk) for (t, y, a, mk), qi in zip(rows, q)]
    b = coalesce([convert_request(datums, "gspo")], 2)        # a padding row too
    low, high = clip_thresholds("gspo", None)
    assert (low, high) == (1 - 3e-4, 1 + 4e-4)

    out = compute_loss(m(b.input_ids), b, clip=(low, high))
    out.loss.backward()
    got_grad = m.E.grad.clone()
    m.E.grad = None

    ref_total, ref_log_ratios, ref_clipped = torch.zeros((), dtype=torch.float64), [], []
    for (t, y, a, mk), qi in zip(rows, q):
        lp = reference_lp(m(torch.tensor([t])), torch.tensor([y]))[0]
        loss, lr, c = tinker_gspo(lp, torch.tensor(qi, dtype=torch.float64), torch.tensor(a, dtype=torch.float64),
                                  torch.tensor(mk), low, high)
        ref_total = ref_total + loss
        ref_log_ratios.append(lr)
        ref_clipped.append(float(c))
    ref_total.backward()

    # token_logprobs computes in float32; the reference is float64
    assert math.isclose(float(out.loss), float(ref_total), rel_tol=1e-5, abs_tol=1e-6)
    assert torch.allclose(got_grad, m.E.grad, rtol=1e-4, atol=1e-6)
    assert out.seq_log_ratio[:3].tolist() == pytest.approx(ref_log_ratios, abs=1e-6)
    assert out.seq_clipped[:3].tolist() == ref_clipped
    assert float(out.seq_clipped[3]) == 0.0 and float(out.seq_log_ratio[3]) == 0.0   # the padding row


def test_gspo_needs_a_marked_token_and_its_mask():
    with pytest.raises(ValueError, match="marks no token"):
        convert_request([rl_datum([1, 2], [2, 3], [-1.0, -1.0], [1.0, 1.0], sequence_ratio_mask=[0.0, 0.0])], "gspo")
    with pytest.raises(ValueError, match="missing sequence_ratio_mask"):
        convert_request([rl_datum([1, 2], [2, 3], [-1.0, -1.0], [1.0, 1.0])], "gspo")
    with pytest.raises(ValueError, match="not read: sequence_ratio_mask"):
        convert_request([rl_datum([1, 2], [2, 3], [-1.0, -1.0], [1.0, 1.0], sequence_ratio_mask=[1.0, 1.0])], "ppo")
