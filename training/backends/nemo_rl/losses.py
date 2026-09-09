"""Tinker-owned loss functions for the NeMo RL backend."""

import torch
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn, NLLLoss

from .loss_config import NO_CLIP, NO_GRAD_CLIP, TINKER_PG_LOSS_DEFAULTS  # noqa: F401  (re-exported)


class TinkerSumCELoss(NLLLoss):
    """Pure-sum cross-entropy: L = sum(-logprob * weight), no token-count mean.

    Tinker's `cross_entropy` contract (and forward_backward_custom, which routes
    DPO/custom losses through it) requires a pure sum so the client owns
    normalization; Miles honors this via `_loss_norm_total=1`. NeMo RL's NLLLoss
    divides by `global_valid_toks` (a mean), which mis-scales the gradient — and
    for the custom-loss path that divisor is a sum of real-valued gradient
    coefficients, not a token count (BUG-015). We reuse NLLLoss's logprob gather
    verbatim and only force the normalization factor to 1 (pure sum).
    """

    def __call__(
        self, next_token_logits, data, global_valid_seqs, global_valid_toks,
        *args, **kwargs,
    ):
        one = (
            torch.ones_like(global_valid_toks)
            if torch.is_tensor(global_valid_toks)
            else 1.0
        )
        return super().__call__(
            next_token_logits, data, global_valid_seqs, one, *args, **kwargs,
        )


class TinkerSumPGLoss(ClippedPGLossFn):
    """Pure-sum policy-gradient loss: -sum(ratio * A) over masked tokens.

    Same reason as TinkerSumCELoss: the contract is a sum the client owns;
    ClippedPGLossFn divides by the global token count. Force both
    normalization factors to 1 and reuse everything else (ratio, clip, KL).
    """

    def __call__(
        self, next_token_logits, data, global_valid_seqs, global_valid_toks,
        *args, **kwargs,
    ):
        one_s = torch.ones_like(global_valid_seqs) if torch.is_tensor(global_valid_seqs) else 1.0
        one_t = torch.ones_like(global_valid_toks) if torch.is_tensor(global_valid_toks) else 1.0
        return super().__call__(
            next_token_logits, data, one_s, one_t, *args, **kwargs,
        )
