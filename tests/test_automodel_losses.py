"""Automodel losses: Tinker's sum contract from logits, exact gradients, inert padding."""
import math

import pytest
import torch

from tinkercloud.training.backends.automodel.converter import coalesce, convert_request, shard
from tinkercloud.training.backends.automodel.losses import compute_loss, token_logprobs
from tinkercloud.training.models.requests import Datum

V = 11


def ce(tokens, targets, weights):
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": targets}, "weights": {"data": weights}}})


def rl(tokens, targets, q, adv):
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": targets}, "logprobs": {"data": q}, "advantages": {"data": adv}}})


class Toy(torch.nn.Module):
    """logits[r, s] = E[input_ids[r, s]]: one learnable row per token id."""

    def __init__(self, seed=0, dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(V, V, generator=g, dtype=dtype))

    def forward(self, input_ids):
        return self.E[input_ids]


def reference_lp(logits, targets):
    return torch.log_softmax(logits.double(), -1).gather(-1, targets[..., None]).squeeze(-1)


def test_token_logprobs_value_and_grad_match_log_softmax():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 5, V, dtype=torch.float64, generator=g, requires_grad=True)
    t = torch.randint(0, V, (3, 5), generator=g)
    w = torch.randn(3, 5, dtype=torch.float64, generator=g)
    lp = token_logprobs(x, t, chunk_tokens=4)            # chunk boundary inside the batch
    (lp.double() * w).sum().backward()
    got = x.grad.clone()
    x.grad = None
    ref = reference_lp(x, t)
    (ref * w).sum().backward()
    assert torch.allclose(lp.double(), ref, atol=1e-6)
    assert torch.allclose(got, x.grad, atol=1e-6)


def test_bf16_logits_get_bf16_grads_close_to_fp32():
    g = torch.Generator().manual_seed(2)
    x32 = torch.randn(2, 4, V, generator=g)
    t = torch.randint(0, V, (2, 4), generator=g)
    xb = x32.to(torch.bfloat16).requires_grad_(True)
    token_logprobs(xb, t).sum().backward()
    xr = xb.detach().float().requires_grad_(True)
    reference_lp(xr, t).sum().backward()
    assert xb.grad.dtype == torch.bfloat16
    assert torch.allclose(xb.grad.float(), xr.grad.float(), atol=1e-2)


def batch_of(datums, loss_fn, ranks=1):
    return coalesce([convert_request(datums, loss_fn)], ranks)


def test_cross_entropy_is_weighted_sum_of_neg_logprobs():
    b = batch_of([ce([1, 2, 3], [2, 3, 4], [0.0, 1.0, 2.0]), ce([5], [6], [1.0])], "cross_entropy")
    m = Toy()
    logits = m(b.input_ids)
    out = compute_loss(logits, b)
    lp = reference_lp(logits.detach(), b.inputs["target_tokens"])
    want_rows = [-(lp[0, 1] + 2 * lp[0, 2]), -lp[1, 0]]
    assert torch.allclose(out.row_loss.double(), torch.stack(want_rows), atol=1e-6)
    assert math.isclose(float(out.loss), float(sum(want_rows)), rel_tol=1e-6)
    assert out.logprobs[1, 1:].abs().sum() == 0                     # past length 1
    assert out.elementwise[0, 0] == 0                               # weight 0


@pytest.mark.parametrize("loss_fn,clip", [("importance_sampling", None), ("ppo", (0.8, 1.2))])
def test_rl_losses_match_the_bug017_formulas(loss_fn, clip):
    m = Toy(seed=3)
    tokens, targets = [1, 4, 7, 2], [4, 7, 2, 9]
    with torch.no_grad():
        lp_true = reference_lp(m(torch.tensor([tokens])), torch.tensor([targets]))[0]
    q = (lp_true - 3.0).tolist()                 # ratio ~ e^3: the clip must bind for ppo, never for IS
    adv = [1.0, -0.5, 0.0, 2.0]
    b = batch_of([rl(tokens, targets, q, adv)], loss_fn)
    out = compute_loss(m(b.input_ids), b, clip=clip)
    r = torch.exp(lp_true - torch.tensor(q, dtype=torch.float64))
    a = torch.tensor(adv, dtype=torch.float64)
    term = r * a if clip is None else torch.minimum(r * a, r.clamp(*clip) * a)
    assert math.isclose(float(out.loss), float(-term.sum()), rel_tol=1e-6)


def test_padding_is_inert_even_with_extreme_values():
    # row 2 is short; its padding and the padding row would overflow exp if computed
    b = batch_of([rl([1, 2, 3], [2, 3, 4], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                  rl([5], [6], [-1.0], [1.0])], "importance_sampling", ranks=3)
    b.inputs["logprobs"][1, 1:] = -1e9
    b.inputs["logprobs"][2, :] = -1e9
    m = Toy()
    out = compute_loss(m(b.input_ids), b)
    out.loss.backward()
    assert torch.isfinite(out.loss) and torch.isfinite(m.E.grad).all()
    assert out.row_loss[2] == 0


def test_shard_gradients_average_to_the_full_batch_gradient():
    """The worker's rule: each rank backprops loss * num_ranks, ranks average
    gradients; that must equal one pass over every row."""
    datums = [ce([i % V, (i + 3) % V], [(i + 1) % V, (i + 5) % V], [1.0, 0.5]) for i in range(5)]
    full = batch_of(datums, "cross_entropy", ranks=2)                  # 5 rows + 1 padding row
    m = Toy(seed=4)
    out = compute_loss(m(full.input_ids), full)
    out.loss.backward()
    want_grad, want_loss = m.E.grad.clone(), float(out.loss)

    ranks, grads, loss = 2, [], 0.0
    for part in shard(full, ranks):
        m.E.grad = None
        o = compute_loss(m(part.input_ids), part)
        (o.loss * ranks).backward()
        grads.append(m.E.grad.clone())
        loss += float(o.loss)
    # fp32 losses: equal up to summation order
    assert math.isclose(loss, want_loss, rel_tol=1e-6)
    assert torch.allclose(sum(grads) / ranks, want_grad, atol=1e-6)


def test_clip_is_required_for_ppo_and_refused_elsewhere():
    b = batch_of([rl([1], [2], [-1.0], [1.0])], "ppo")
    with pytest.raises(ValueError, match="required by ppo"):
        compute_loss(Toy()(b.input_ids), b)
    b = batch_of([ce([1], [2], [1.0])], "cross_entropy")
    with pytest.raises(ValueError, match="not taken by cross_entropy"):
        compute_loss(Toy()(b.input_ids), b, clip=(0.8, 1.2))
