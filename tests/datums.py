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


def make_synthetic_datum(seq_len: int = 128, response_len: int = 64):
    """A synthetic wire datum (what the router hands the backends): a full
    sequence of `seq_len` tokens sent pre-shifted, with response-length RL
    inputs. Per-token inputs are right-aligned by the converters."""
    import torch

    prompt_len = seq_len - response_len
    tokens = torch.randint(0, 32000, (seq_len,), dtype=torch.long).tolist()
    loss_masks = [0.0] * prompt_len + [1.0] * response_len

    def td(vals, dtype="float32"):
        return {"data": vals, "shape": [len(vals)], "dtype": dtype}

    return Datum.model_validate({
        "model_input": {"tokens": tokens[:-1]},
        "loss_fn_inputs": {
            "target_tokens": td(tokens[1:], "int64"),
            "mask": td(loss_masks[1:]),
            "advantages": td([0.1] * response_len),
            "logprobs": td([-5.0 + 0.1 * i for i in range(response_len)]),
            "ref_logprobs": td([-5.2 + 0.1 * i for i in range(response_len)]),
        },
    })


def make_synthetic_batch(batch_size: int = 4, seq_len: int = 128) -> list:
    """A batch of synthetic wire datums."""
    return [make_synthetic_datum(seq_len=seq_len) for _ in range(batch_size)]
