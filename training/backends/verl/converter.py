"""
veRL data converter: Tinker Datum list <-> veRL padded/left-right TensorDict.

Layout: prompt = tokens[0:1] (prompt_len == 1 for every sample, satisfying
veRL's prompt_len > 0 requirement), responses = tokens[1:] right-padded.
Engine log_probs[t] = logp(tokens[t+1] | tokens[:t+1]) then align 1:1 with
client target_tokens / weights / advantages (all length N-1, Miles parity).

cross_entropy also takes `target_tokens` and `weights` of shape [N, K]: K
candidate ids per position, each with its own weight. Those travel as
[B, Rmax, K] tensors and the logprobs come back [N, K] per datum.

Datum schemas (from tinker-cookbook; arrives as Pydantic
ForwardBackwardDatum OR plain dict — handle both):
  SFT: model_input.tokens = tokens[:-1], target_tokens = tokens[1:],
       weights = weights[1:]  (full seq = tokens[:-1] + [target_tokens[-1]])
  RL:  model_input.tokens = full rollout, loss_fn_inputs = {advantages,
       logprobs (sampling), ...} aligned to tokens[1:].
"""
import math
from typing import Any, Dict, List, Optional

import torch

from ..base import DataConverter
from ...models.requests import Datum


def _tensor_data(tensor, dtype) -> Optional[torch.Tensor]:
    """A TensorData -> 1-D tensor; None stays None (the input was not sent)."""
    if tensor is None:
        return None
    return torch.tensor(tensor.data, dtype=dtype).flatten()


def _shaped(tensor, dtype, name: str, index: int) -> torch.Tensor:
    """A TensorData in its wire shape, rank 1 or 2 (no shape on the wire: rank 1).

    Raises:
        ValueError: the shape does not hold the data, or its rank is above 2.
    """
    shape = tensor.shape if tensor.shape is not None else [len(tensor.data)]
    if len(shape) > 2:
        raise ValueError(f"datum {index}: {name} has rank {len(shape)}; the verl backend reads rank 1 or 2")
    values = torch.tensor(tensor.data, dtype=dtype).reshape(-1)
    if values.numel() != math.prod(shape):
        raise ValueError(f"datum {index}: {name} carries {values.numel()} values for shape {list(shape)}")
    return values.reshape(shape)


def _cross_entropy_targets(data: List[Datum], seqs: List[torch.Tensor]):
    """Each datum's (target_tokens, weights) in wire shape, None where not sent.

    Raises:
        ValueError: a field whose first dimension is not the datum's target
            count, weights shaped unlike target_tokens, or datums that differ
            in rank or K.
    """
    targets: List[Optional[torch.Tensor]] = []
    weights: List[Optional[torch.Tensor]] = []
    for i, (datum, seq) in enumerate(zip(data, seqs)):
        t, w = _loss_input(datum, "target_tokens"), _loss_input(datum, "weights")
        t = None if t is None else _shaped(t, torch.long, "target_tokens", i)
        w = None if w is None else _shaped(w, torch.float32, "weights", i)
        for name, values in (("target_tokens", t), ("weights", w)):
            if values is not None and values.shape[0] != seq.numel() - 1:
                raise ValueError(f"datum {i}: {name} has {values.shape[0]} rows for {seq.numel() - 1} target positions")
        if t is not None and w is not None and w.shape != t.shape:
            raise ValueError(f"datum {i}: weights has shape {list(w.shape)}, target_tokens {list(t.shape)}")
        targets.append(t)
        weights.append(w)
    for i, (t, w) in enumerate(zip(targets, weights)):
        if t is None and w is not None and w.dim() > 1:
            raise ValueError(f"datum {i}: weights has shape {list(w.shape)} and no target_tokens to weigh")
    # a datum without target_tokens scores its own next tokens: one target per position
    trailing = {() if t is None else tuple(t.shape[1:]) for t in targets}
    if len(trailing) > 1:
        raise ValueError("target_tokens differs across the datums of one call: trailing shapes "
                         f"{sorted(trailing)} (() is one target per position)")
    return targets, weights


def refuse_nk_targets_outside_cross_entropy(data: List[Datum], loss_fn: str) -> None:
    """Raises ValueError naming target_tokens when a datum carries [N, K] targets for
    a loss that scores one target per position."""
    if loss_fn == "cross_entropy":
        return
    for i, datum in enumerate(data):
        target = _loss_input(datum, "target_tokens")
        if target is not None and target.shape is not None and len(target.shape) > 1:
            raise ValueError(f"datum {i}: target_tokens has rank {len(target.shape)}; "
                             f"the verl backend reads [N, K] targets for cross_entropy only, not {loss_fn}")


def _loss_input(datum: Datum, *names):
    """The first of `names` the client sent in loss_fn_inputs, else None."""
    for name in names:
        v = datum.loss_fn_inputs.get(name)
        if v is not None:
            return v
    return None


def _extract_tokens(datum: Datum) -> torch.Tensor:
    mi = datum.model_input
    if mi.chunks:
        toks = [t for c in mi.chunks for t in (c.tokens or [])]
    elif mi.tokens is not None:
        toks = list(mi.tokens)
    elif mi.input_ids is not None:
        toks = list(mi.input_ids)
    else:
        raise ValueError("model_input carries no chunks, tokens or input_ids")
    if not toks:
        raise ValueError("datum has no tokens")
    return torch.tensor(toks, dtype=torch.long)


def _full_tokens(datum: Datum) -> torch.Tensor:
    tokens = _extract_tokens(datum)
    target = _loss_input(datum, "target_tokens")
    if target is not None and target.data:
        # SFT shape: model tokens are tokens[:-1]; append the final target token.
        # With [N, K] targets that is the last value sent: the appended token
        # is an input whose own output is never read.
        return torch.cat([tokens, torch.tensor(target.data[-1:], dtype=torch.long)])
    return tokens


class VerlDataConverter(DataConverter):
    """Tinker Datum list -> padded left-right dict (backend nests it via
    verl's left_right_2_no_padding at dispatch time)."""

    def forward_backward_to_backend(
        self,
        data: List[Datum],
        loss_fn: str,
        args: Any,
    ) -> Dict[str, torch.Tensor]:
        seqs = [_full_tokens(d) for d in data]
        n_lens = [int(s.numel()) for s in seqs]
        resp_lens = [n - 1 for n in n_lens]
        b, rmax = len(seqs), max(resp_lens)
        smax = rmax + 1

        input_ids = torch.zeros(b, smax, dtype=torch.long)
        attention_mask = torch.zeros(b, smax, dtype=torch.long)
        position_ids = torch.zeros(b, smax, dtype=torch.long)
        responses = torch.zeros(b, rmax, dtype=torch.long)
        response_mask = torch.zeros(b, rmax, dtype=torch.long)
        sent_targets: List[Optional[torch.Tensor]]
        sent_weights: List[Optional[torch.Tensor]]
        if loss_fn == "cross_entropy":
            sent_targets, sent_weights = _cross_entropy_targets(data, seqs)
        else:
            refuse_nk_targets_outside_cross_entropy(data, loss_fn)
            sent_targets, sent_weights = [None] * b, [None] * b
        # () for [N] targets, (K,) for [N, K]; one shape per call (_cross_entropy_targets)
        trailing = next((tuple(t.shape[1:]) for t in sent_targets if t is not None), ())
        weights = torch.zeros(b, rmax, *trailing, dtype=torch.float32)
        target_tokens = torch.zeros(b, rmax, *trailing, dtype=torch.long)
        advantages = torch.zeros(b, rmax, dtype=torch.float32)
        old_log_probs = torch.zeros(b, rmax, dtype=torch.float32)

        for i, (seq, n) in enumerate(zip(seqs, n_lens)):
            input_ids[i, :n] = seq
            attention_mask[i, :n] = 1
            position_ids[i, :n] = torch.arange(n)
            responses[i, : n - 1] = seq[1:]
            response_mask[i, : n - 1] = 1

            w, t = sent_weights[i], sent_targets[i]
            if w is not None:
                weights[i, : n - 1] = w
            if t is not None:
                target_tokens[i, : n - 1] = t
            adv = _tensor_data(_loss_input(data[i], "advantages"), torch.float32)
            if adv is not None:
                advantages[i, : min(len(adv), n - 1)] = adv[: n - 1]
            lp = _tensor_data(_loss_input(data[i], "logprobs", "log_probs"), torch.float32)
            if lp is not None:
                old_log_probs[i, : min(len(lp), n - 1)] = lp[: n - 1]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": input_ids[:, :1],
            "responses": responses,
            "response_mask": response_mask,
        }
        if loss_fn == "cross_entropy":
            out["weights"] = weights
            if trailing:
                out["target_tokens"] = target_tokens
        else:
            out["advantages"] = advantages
            out["old_log_probs"] = old_log_probs
        return out

    def forward_to_backend(self, data: List[Datum], args: Any) -> Any:
        return self.forward_backward_to_backend(data, "cross_entropy", args)

    def backend_to_forward_result(self, result: Any, data: List[Any]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    def backend_to_forward_backward_result(self, result: Any, data: List[Any]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    @staticmethod
    def extract_logprobs(padded_logprobs: torch.Tensor, data: List[Any]) -> List[Dict[str, Any]]:
        """Each datum's logprobs out of a padded [B, Rmax] or [B, Rmax, K] tensor:
        shape [N] or [N, K], the data in row order."""
        outputs = []
        for i, datum in enumerate(data):
            n = int(_full_tokens(datum).numel())
            lp = padded_logprobs[i, : n - 1].detach().float().cpu()
            outputs.append({"logprobs": {"data": lp.reshape(-1).tolist(), "dtype": "float32", "shape": list(lp.shape)}})
        return outputs
