"""Tinker-owned loss functions for the NeMo RL backend."""

import torch
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn, NLLLossFn

from .loss_config import NO_CLIP, NO_GRAD_CLIP, TINKER_PG_LOSS_DEFAULTS  # noqa: F401  (re-exported)


class TinkerSumCELoss(NLLLossFn):
    """Pure-sum cross-entropy: L = sum(-logprob * weight), no token-count mean.

    Tinker's `cross_entropy` contract (and forward_backward_custom, which routes
    DPO/custom losses through it) requires a pure sum so the client owns
    normalization; Miles honors this via `_loss_norm_total=1`. NeMo RL's NLLLossFn
    divides by `global_valid_toks` (a mean), which mis-scales the gradient — and
    for the custom-loss path that divisor is a sum of real-valued gradient
    coefficients, not a token count (BUG-015). We reuse NLLLossFn's logprob gather
    verbatim and only force the normalization factor to 1 (pure sum). The
    parameter names are the parent's: NeMo RL calls the loss by keyword.
    """

    def __call__(self, next_token_logprobs, data, global_valid_seqs, global_valid_toks, **kwargs):
        one = torch.ones_like(global_valid_toks) if torch.is_tensor(global_valid_toks) else 1.0
        return super().__call__(next_token_logprobs=next_token_logprobs, data=data,
                                global_valid_seqs=global_valid_seqs, global_valid_toks=one, **kwargs)


class TinkerSumPGLoss(ClippedPGLossFn):
    """Pure-sum policy-gradient loss: -sum(ratio * A) over masked tokens.

    Same reason as TinkerSumCELoss: the contract is a sum the client owns;
    ClippedPGLossFn divides by the global token count. Force both
    normalization factors to 1 and reuse everything else (ratio, clip, KL).
    `cfg` is the ClippedPGLossConfig field dict the builder writes.
    """

    def __init__(self, cfg: dict):
        super().__init__(ClippedPGLossConfig(**cfg))

    def __call__(self, next_token_logprobs, data, global_valid_seqs, global_valid_toks):
        one_s = torch.ones_like(global_valid_seqs) if torch.is_tensor(global_valid_seqs) else 1.0
        one_t = torch.ones_like(global_valid_toks) if torch.is_tensor(global_valid_toks) else 1.0
        return super().__call__(next_token_logprobs=next_token_logprobs, data=data,
                                global_valid_seqs=one_s, global_valid_toks=one_t)
