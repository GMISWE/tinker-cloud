"""
Tinker Datums -> padded tensors for the training ranks, and per-row results
back to per-request outputs (R3).

Layout: Tinker's own. model_input has length T and every loss input
(target_tokens, weights, logprobs, advantages) has length T; the logits at
position t score target_tokens[t]. The loss consumes this directly, so there is
no sequence reconstruction and no shift.

One pass may carry several requests (consecutive passes of one model with the
same loss, coalesced by the backend's queue). Validation is per request
(convert_request): a bad request fails alone, before it can share a pass.
coalesce() joins valid requests, pads the rows to a multiple of the rank count
(padding rows have length 0 and weight 0) and records each request's rows;
shard() cuts one equal chunk per rank, so every rank runs the same number of
micro-steps and FSDP2's collectives line up.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from ...models.requests import Datum

# loss_fn -> the loss_fn_inputs it reads; every one is required and no other
# key is accepted, so a key the loss would ignore fails instead of being dropped.
LOSS_INPUTS: Dict[str, Tuple[str, ...]] = {
    "cross_entropy": ("target_tokens", "weights"),
    "importance_sampling": ("target_tokens", "logprobs", "advantages"),
    "ppo": ("target_tokens", "logprobs", "advantages"),
}

_FLOAT_INPUTS = ("weights", "logprobs", "advantages")


@dataclass(frozen=True)
class RequestRows:
    """One request's datums as 1-D tensors, validated for `loss_fn`."""
    loss_fn: str
    tokens: List[torch.Tensor]                 # long [T_i] per datum
    inputs: Dict[str, List[torch.Tensor]]      # loss input -> [T_i] per datum


@dataclass(frozen=True)
class Batch:
    """Rows of one pass, every tensor [R, S] (R a multiple of the rank count)."""
    loss_fn: str
    input_ids: torch.Tensor                    # long, right-padded with 0
    lengths: torch.Tensor                      # long [R]; 0 for padding rows
    inputs: Dict[str, torch.Tensor]            # LOSS_INPUTS[loss_fn], 0 past each length
    request_rows: Tuple[Tuple[int, int], ...]  # (start, end) of each request's rows

    @property
    def num_rows(self) -> int:
        return self.input_ids.shape[0]


def convert_request(data: Sequence[Datum], loss_fn: str) -> RequestRows:
    """Validate one request's datums against `loss_fn` and read them as tensors.

    Raises:
        ValueError: unknown loss, empty request, image input, a missing or
            unexpected loss input, or an input whose length is not the token count.
    """
    if loss_fn not in LOSS_INPUTS:
        raise ValueError(f"loss_fn {loss_fn!r} is not served by the automodel backend; "
                         f"supported: {', '.join(sorted(LOSS_INPUTS))}")
    if not data:
        raise ValueError("the request carries no datums")
    wanted = LOSS_INPUTS[loss_fn]
    tokens: List[torch.Tensor] = []
    inputs: Dict[str, List[torch.Tensor]] = {k: [] for k in wanted}
    for i, datum in enumerate(data):
        ids = _tokens(datum, i)
        given = set(datum.loss_fn_inputs)
        if given != set(wanted):
            missing = sorted(set(wanted) - given)
            extra = sorted(given - set(wanted))
            raise ValueError(
                f"datum {i}: {loss_fn} reads exactly {', '.join(wanted)}"
                + (f"; missing {', '.join(missing)}" if missing else "")
                + (f"; not read: {', '.join(extra)}" if extra else "")
            )
        for key in wanted:
            dtype = torch.float32 if key in _FLOAT_INPUTS else torch.long
            values = torch.tensor(datum.loss_fn_inputs[key].data, dtype=dtype).reshape(-1)
            if values.numel() != ids.numel():
                raise ValueError(f"datum {i}: {key} has {values.numel()} entries for {ids.numel()} tokens")
            inputs[key].append(values)
        tokens.append(ids)
    return RequestRows(loss_fn=loss_fn, tokens=tokens, inputs=inputs)


def coalesce(requests: Sequence[RequestRows], num_ranks: int) -> Batch:
    """Join requests of one loss into one padded batch with R % num_ranks == 0."""
    loss_fn = requests[0].loss_fn
    assert all(r.loss_fn == loss_fn for r in requests), "the queue coalesces one loss per pass"
    tokens = [t for r in requests for t in r.tokens]
    real = len(tokens)
    rows = real + (-real) % num_ranks
    width = max(t.numel() for t in tokens)

    input_ids = torch.zeros(rows, width, dtype=torch.long)
    lengths = torch.zeros(rows, dtype=torch.long)
    for i, t in enumerate(tokens):
        input_ids[i, : t.numel()] = t
        lengths[i] = t.numel()
    inputs: Dict[str, torch.Tensor] = {}
    for key in LOSS_INPUTS[loss_fn]:
        dtype = torch.float32 if key in _FLOAT_INPUTS else torch.long
        dest = torch.zeros(rows, width, dtype=dtype)
        for i, values in enumerate(v for r in requests for v in r.inputs[key]):
            dest[i, : values.numel()] = values
        inputs[key] = dest

    bounds, start = [], 0
    for r in requests:
        bounds.append((start, start + len(r.tokens)))
        start += len(r.tokens)
    return Batch(loss_fn=loss_fn, input_ids=input_ids, lengths=lengths,
                 inputs=inputs, request_rows=tuple(bounds))


def shard(batch: Batch, num_ranks: int) -> List[Batch]:
    """One contiguous, equal chunk of rows per rank (each to be ray.put once)."""
    per_rank = batch.num_rows // num_ranks
    assert per_rank * num_ranks == batch.num_rows, "coalesce pads to a multiple of the rank count"
    chunks = []
    for r in range(num_ranks):
        rows = slice(r * per_rank, (r + 1) * per_rank)
        chunks.append(Batch(
            loss_fn=batch.loss_fn,
            input_ids=batch.input_ids[rows],
            lengths=batch.lengths[rows],
            inputs={k: v[rows] for k, v in batch.inputs.items()},
            request_rows=(),
        ))
    return chunks


def split_rows(batch: Batch, row_values: torch.Tensor) -> List[List[List[float]]]:
    """[R, S] per-position values (rank chunks concatenated in rank order) ->
    per request, per datum, the values over that datum's tokens. Padding rows
    fall outside every request's bounds and are dropped."""
    assert row_values.shape[0] == batch.num_rows
    out = []
    for start, end in batch.request_rows:
        out.append([row_values[i, : int(batch.lengths[i])].tolist() for i in range(start, end)])
    return out


def _tokens(datum: Datum, index: int) -> torch.Tensor:
    mi = datum.model_input
    if mi.chunks:
        if any(c.type == "image" for c in mi.chunks):
            raise ValueError(f"datum {index}: image input is not served by the automodel backend")
        tokens: List[int] = []
        for c in mi.chunks:
            if c.tokens is None:
                raise ValueError(f"datum {index}: a text chunk carries no tokens")
            tokens.extend(c.tokens)
        return torch.tensor(tokens, dtype=torch.long)
    if mi.tokens is not None:
        return torch.tensor(mi.tokens, dtype=torch.long)
    if mi.input_ids is not None:
        return torch.tensor(mi.input_ids, dtype=torch.long)
    raise ValueError(f"datum {index}: model_input carries no chunks, tokens or input_ids")


def row_sums(batch: Batch, row_values: torch.Tensor) -> List[float]:
    """Per request, the sum of a per-row scalar (e.g. each row's loss sum)."""
    assert row_values.shape == (batch.num_rows,)
    return [float(row_values[start:end].sum()) for start, end in batch.request_rows]
