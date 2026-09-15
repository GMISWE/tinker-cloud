"""
veRL data converter: Tinker Datum list <-> veRL padded/left-right TensorDict.

Layout: prompt = tokens[0:1] (prompt_len == 1 for every sample, satisfying
veRL's prompt_len > 0 requirement), responses = tokens[1:] right-padded.
Engine log_probs[t] = logp(tokens[t+1] | tokens[:t+1]) then align 1:1 with
client target_tokens / weights / advantages (all length N-1, Miles parity).

Datum schemas (from tinker-cookbook; arrives as Pydantic
ForwardBackwardDatum OR plain dict — handle both):
  SFT: model_input.tokens = tokens[:-1], target_tokens = tokens[1:],
       weights = weights[1:]  (full seq = tokens[:-1] + [target_tokens[-1]])
  RL:  model_input.tokens = full rollout, loss_fn_inputs = {advantages,
       logprobs (sampling), ...} aligned to tokens[1:].
"""
from typing import Any, Dict, List, Optional

import torch

from ..base import DataConverter
from ...models.requests import Datum


def _tensor_data(tensor, dtype) -> Optional[torch.Tensor]:
    """A TensorData -> 1-D tensor; None stays None (the input was not sent)."""
    if tensor is None:
        return None
    return torch.tensor(tensor.data, dtype=dtype).flatten()


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
    target = _tensor_data(_loss_input(datum, "target_tokens"), torch.long)
    if target is not None and target.numel() > 0:
        # SFT shape: model tokens are tokens[:-1]; append final target token
        return torch.cat([tokens, target[-1:]])
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
        weights = torch.zeros(b, rmax, dtype=torch.float32)
        advantages = torch.zeros(b, rmax, dtype=torch.float32)
        old_log_probs = torch.zeros(b, rmax, dtype=torch.float32)

        for i, (seq, n) in enumerate(zip(seqs, n_lens)):
            input_ids[i, :n] = seq
            attention_mask[i, :n] = 1
            position_ids[i, :n] = torch.arange(n)
            responses[i, : n - 1] = seq[1:]
            response_mask[i, : n - 1] = 1

            w = _tensor_data(_loss_input(data[i], "weights"), torch.float32)
            if w is not None:
                weights[i, : min(len(w), n - 1)] = w[: n - 1]
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
        """Datum-aligned per-sample logprobs (length N_i - 1 each)."""
        outputs = []
        for i, datum in enumerate(data):
            n = int(_full_tokens(datum).numel())
            lp = padded_logprobs[i, : n - 1].detach().float().cpu()
            outputs.append({"logprobs": {"data": lp.tolist(), "dtype": "float32", "shape": [n - 1]}})
        return outputs
