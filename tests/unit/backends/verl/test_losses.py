"""The verl cross_entropy loss as the engine calls it: once as the logits hook
(the [N, K] path), once as the loss. Checked against -sum(w * logprob) taken
straight from log_softmax, value and gradient, for weights of both signs.
Logits are float64 so that summation order stays below the tolerances."""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("verl", reason="the loss reads verl's padding helpers")

from tinkercloud.training.backends.verl.backend import VerlBackend, VerlHandle, _loss_dispatch  # noqa: E402
from .datums import nk_datum, wire_values  # noqa: E402

VOCAB = 50
ENGINE = {"micro_batch_size_per_gpu": 1, "infer_micro_batch_size_per_gpu": 1, "ulysses_sequence_parallel_size": 1}


def engine_loss(data, logits):
    """What verl's FSDP engine does around the loss function: `logits` is one
    row per input token of the batch, sequences back to back."""
    backend = VerlBackend()
    handle = VerlHandle(model_id="m", backend_type="verl", config={"engine": ENGINE}, trainer_config=None)  # unread here
    td = backend._to_tensordict(handle, data, "cross_entropy")
    offsets = td["input_ids"].offsets()
    if "target_tokens" in td.keys():
        columns = _loss_dispatch(student_logits=logits.unsqueeze(0), data=td, loss_name="cross_entropy")
        per_position = {key: column.squeeze(0) for key, column in columns.items()}
    else:
        next_ids = torch.roll(td["input_ids"].values(), shifts=-1)
        per_position = {"log_probs": F.log_softmax(logits, dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)}
    model_output = {key: torch.nested.nested_tensor_from_jagged(v, offsets) for key, v in per_position.items()}
    loss, _ = _loss_dispatch(model_output=model_output, data=td, loss_name="cross_entropy")
    return loss


def reference_loss(data, logits):
    """-sum(w * log_softmax(logits)[position, id]) read from the datums themselves."""
    total, row = logits.new_zeros(()), 0
    for d in data:
        targets, weights = wire_values(d, "target_tokens"), wire_values(d, "weights")
        n = targets.shape[0]
        log_probs = F.log_softmax(logits[row:row + n], dim=-1)
        picked = log_probs.gather(-1, targets.reshape(n, -1)).reshape(targets.shape)
        total = total - (weights * picked).sum()
        row += n + 1   # the appended input token has no target
    return total


def loss_and_grad(fn, data, logits):
    logits = logits.clone().requires_grad_(True)
    loss = fn(data, logits)
    (grad,) = torch.autograd.grad(loss, logits)
    return loss.detach(), grad


def nk_batch():
    g = torch.Generator().manual_seed(0)
    lengths, k = (5, 3), 4
    data = []
    for n in lengths:
        targets = torch.randint(0, VOCAB, (n, k), generator=g)
        weights = torch.randn(n, k, generator=g)            # both signs
        data.append(nk_datum(torch.randint(0, VOCAB, (n,), generator=g).tolist(), targets.tolist(), weights.tolist()))
    logits = torch.randn(sum(lengths) + len(lengths), VOCAB, generator=g, dtype=torch.float64) * 3
    return data, logits


def test_nk_loss_and_gradient_are_the_signed_weighted_sum():
    data, logits = nk_batch()
    assert any(w < 0 for w in data[0].loss_fn_inputs["weights"].data)
    loss, grad = loss_and_grad(engine_loss, data, logits)
    want_loss, want_grad = loss_and_grad(reference_loss, data, logits)
    torch.testing.assert_close(loss, want_loss, rtol=1e-6, atol=0)
    torch.testing.assert_close(grad, want_grad, rtol=1e-5, atol=1e-7)


def test_a_zero_weight_column_adds_no_gradient():
    """A column the client zeroes must act as if it were never sent: a hook that scored it
    through a shared normaliser, or a loss that masked by position, would still move its logit."""
    data, logits = nk_batch()
    zeroed, dropped = [], []
    for d in data:
        targets, weights = wire_values(d, "target_tokens"), wire_values(d, "weights")
        weights[:, 1] = 0.0
        tokens = d.model_input.tokens
        zeroed.append(nk_datum(tokens, targets.tolist(), weights.tolist()))
        keep = [0, 2, 3]
        dropped.append(nk_datum(tokens, targets[:, keep].tolist(), weights[:, keep].tolist()))
    _, grad = loss_and_grad(engine_loss, zeroed, logits)
    _, grad_without_the_column = loss_and_grad(reference_loss, dropped, logits)
    torch.testing.assert_close(grad, grad_without_the_column, rtol=1e-5, atol=1e-7)


def test_n_path_is_the_signed_sum_and_k_equal_one_matches_it():
    g = torch.Generator().manual_seed(1)
    n = 6
    tokens = torch.randint(0, VOCAB, (n + 1,), generator=g).tolist()
    weights = torch.randn(n, generator=g).tolist()
    logits = torch.randn(n + 1, VOCAB, generator=g, dtype=torch.float64) * 3
    # the [N] path scores the shifted input ids, so the targets are those ids
    flat = [nk_datum(tokens[:-1], tokens[1:], weights)]
    one_column = [nk_datum(tokens[:-1], [[t] for t in tokens[1:]], [[w] for w in weights])]
    loss_n, grad_n = loss_and_grad(engine_loss, flat, logits)
    loss_k, grad_k = loss_and_grad(engine_loss, one_column, logits)
    torch.testing.assert_close(loss_k, loss_n, rtol=1e-6, atol=0)
    torch.testing.assert_close(grad_k, grad_n, rtol=1e-5, atol=1e-7)
    # and the [N] path itself is the signed sum (the weights above carry both signs)
    assert min(weights) < 0 < max(weights)
    want_loss, want_grad = loss_and_grad(reference_loss, flat, logits)
    torch.testing.assert_close(loss_n, want_loss, rtol=1e-6, atol=0)
    torch.testing.assert_close(grad_n, want_grad, rtol=1e-5, atol=1e-7)
