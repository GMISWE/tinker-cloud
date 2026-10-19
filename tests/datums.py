"""Wire-shaped Datum builders shared by the unit and GPU suites."""
from tinkercloud.training.models.requests import Datum


def ce_datum(tokens, targets=None, weights=None):
    """A cross-entropy datum; targets default to the tokens, weights to 1."""
    targets = tokens if targets is None else targets
    weights = [1.0] * len(tokens) if weights is None else weights
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": targets}, "weights": {"data": weights}}})


def rl_datum(tokens, targets, logprobs, advantages, **extra):
    """An RL datum (target_tokens, logprobs, advantages, plus any extra per-token input)."""
    inputs = {"target_tokens": {"data": targets}, "logprobs": {"data": logprobs}, "advantages": {"data": advantages}}
    inputs.update({k: {"data": v} for k, v in extra.items()})
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": inputs})


def random_ce_datum(g, length, weight=1.0):
    """Random tokens; each position predicts the next (the last wraps); the
    first third of the positions is weighted 0."""
    import torch
    tokens = torch.randint(100, 20000, (length,), generator=g).tolist()
    return ce_datum(tokens, tokens[1:] + [tokens[0]], [0.0] * (length // 3) + [weight] * (length - length // 3))
