"""Cross-entropy datums in wire form, shared by the verl unit and GPU tests."""
import torch

from tinkercloud.training.models.requests import Datum


def nk_datum(tokens: list, targets, weights) -> Datum:
    """A cross-entropy datum; `targets` and `weights` are [N] or [N, K], as tensors or nested lists."""
    def wire(values):
        t = torch.as_tensor(values)
        return {"data": t.reshape(-1).tolist(), "shape": list(t.shape)}
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": wire(targets), "weights": wire(weights)}})


def wire_values(datum: Datum, name: str) -> torch.Tensor:
    """`datum`'s loss input `name` as a tensor in its wire shape."""
    sent = datum.loss_fn_inputs[name]
    return torch.tensor(sent.data).reshape(sent.shape)
