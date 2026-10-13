"""
Tinker's losses from logits (R10), on one rank's shard of a converter Batch.

Contract (API-CONTRACT F2, BUG-017): a per-token SUM, no mean, at T=1.
  cross_entropy:        -sum_t w_t * lp_t
  importance_sampling:  -sum_t exp(lp_t - q_t) * A_t
  ppo:                  -sum_t min(r_t * A_t, clip(r_t, low, high) * A_t),  r_t = exp(lp_t - q_t)
lp_t is the model's logprob of target_tokens[t] from the logits at position t
(the converter's layout: no shift). q_t are the sampler's logprobs.

Positions past a row's length (and padding rows) are selected out with
torch.where before exp, so they add exactly 0 to the loss and the gradient and
cannot turn into NaN. Real positions follow the formula exactly: no ratio clamp.

compute_loss returns this shard's sum. Ranks average gradients (FSDP2), so the
worker scales the loss by the rank count before backward; the average of the
scaled shard gradients is then the gradient of the one sum over all rows.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .converter import Batch

# Rows of [S, V] logits upcast to fp32 at a time in token_logprobs.
DEFAULT_CHUNK_TOKENS = 1024


class _TokenLogprob(torch.autograd.Function):
    """lp[n] = x[n, t_n] - logsumexp(x[n]) over [N, V] logits, in fp32, chunked.

    Saves the logits as given and the [N] logsumexp; backward recomputes the
    softmax per chunk (grad x = g * (onehot(t) - softmax(x))), so no fp32
    [N, V] tensor outlives a chunk."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
        n = logits.shape[0]
        lp = torch.empty(n, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n, dtype=torch.float32, device=logits.device)
        for s in range(0, n, chunk):
            x = logits[s:s + chunk].float()
            lse[s:s + chunk] = torch.logsumexp(x, dim=-1)
            lp[s:s + chunk] = x.gather(1, targets[s:s + chunk, None]).squeeze(1) - lse[s:s + chunk]
        ctx.save_for_backward(logits, targets, lse)
        ctx.chunk = chunk
        return lp

    @staticmethod
    def backward(ctx, grad_lp: torch.Tensor):
        logits, targets, lse = ctx.saved_tensors
        grad = torch.empty_like(logits)
        for s in range(0, logits.shape[0], ctx.chunk):
            g = grad_lp[s:s + ctx.chunk, None]
            p = torch.exp(logits[s:s + ctx.chunk].float() - lse[s:s + ctx.chunk, None])
            p.mul_(-g)
            p.scatter_add_(1, targets[s:s + ctx.chunk, None], g)
            grad[s:s + ctx.chunk] = p.to(logits.dtype)
        return grad, None, None


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor,
                   chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> torch.Tensor:
    """[R, S, V] logits, [R, S] targets -> [R, S] fp32 logprobs of the targets."""
    r, s, v = logits.shape
    lp = _TokenLogprob.apply(logits.reshape(r * s, v), targets.reshape(r * s), chunk_tokens)
    return lp.reshape(r, s)


@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor           # scalar, this shard's sum, carries the graph
    row_loss: torch.Tensor       # [R] detached: each row's sum (padding rows 0)
    logprobs: torch.Tensor       # [R, S] detached: lp_t (0 past each length)
    elementwise: torch.Tensor    # [R, S] detached: each position's term of the sum


def compute_loss(logits: torch.Tensor, shard: Batch,
                 clip: Optional[Tuple[float, float]] = None,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> LossOutput:
    """The shard's loss for shard.loss_fn. `clip` = (low, high) ratio bounds,
    required for ppo and refused for the others."""
    if (shard.loss_fn == "ppo") != (clip is not None):
        raise ValueError(f"clip bounds are {'required' if shard.loss_fn == 'ppo' else 'not taken'} "
                         f"by {shard.loss_fn}")
    width = shard.input_ids.shape[1]
    device = logits.device
    valid = torch.arange(width, device=device)[None, :] < shard.lengths.to(device)[:, None]
    inputs = {k: v.to(device) for k, v in shard.inputs.items()}
    lp = token_logprobs(logits, inputs["target_tokens"], chunk_tokens)
    zero = torch.zeros((), dtype=torch.float32, device=device)

    if shard.loss_fn == "cross_entropy":
        terms = -inputs["weights"] * lp
    elif shard.loss_fn in ("importance_sampling", "ppo"):
        ratio = torch.exp(torch.where(valid, lp - inputs["logprobs"], zero))
        adv = inputs["advantages"]
        terms = -(ratio * adv)
        if shard.loss_fn == "ppo":
            low, high = clip
            terms = -torch.minimum(ratio * adv, ratio.clamp(low, high) * adv)
    else:
        raise ValueError(f"loss_fn {shard.loss_fn!r} has no automodel implementation")

    terms = torch.where(valid, terms, zero)
    row = terms.sum(dim=1)
    return LossOutput(
        loss=row.sum(),
        row_loss=row.detach(),
        logprobs=torch.where(valid, lp, zero).detach(),
        elementwise=terms.detach(),
    )
