"""
Pure-sum loss functions for the veRL backend (Thm 1 contract).

Signature matches verl.workers.utils.losses: (config, model_output, data,
dp_group) -> (loss, metrics). Per-micro-batch partial sums against a
client-owned normalizer (1), scaled by dp_size to cancel FSDP's grad-averaging
all-reduce — split-invariant by construction. We deliberately do NOT use
veRL's sft_loss (flattened-roll boundary leak, see specs/006 q2-plan.md) or
its batch-measured normalizers.

[N, K] cross-entropy (K candidate ids per position, `data["target_tokens"]`
of shape [B, Rmax, K]) needs K logprobs per position where the engine keeps
one. The engine's logits hook (data key `distillation_use_topk`) hands the
full logits to `target_logprob_columns` before the loss; each column returns
as its own per-position model output and `padded_logprobs` stacks them back.
"""
import torch
import torch.nn.functional as F

_METRIC_KEYS = ("weights", "advantages", "old_log_probs", "ref_log_prob")
_COLUMN_KEY = "target_logprobs_{}"


def target_logprob_columns(student_logits, data):
    """The engine's logits hook for [N, K] targets: logits (1, total_tokens, V)
    -> {column key: (1, total_tokens)}, column k holding log p(target_tokens[..., k])
    at every input position. Positions with no target (the prompt's all but last, each
    sequence's last) score token id 0; padded_logprobs never reads them."""
    input_ids, targets = data["input_ids"], data["target_tokens"]
    # response position r is predicted at input position prompt_len - 1 + r
    prompt_len = data["prompts"].shape[1]
    per_input = F.pad(targets, (0, 0, prompt_len - 1, 1))          # [B, prompt_len + Rmax, K]
    seq_lens = input_ids.offsets().diff()
    real = torch.arange(per_input.shape[1], device=per_input.device) < seq_lens.unsqueeze(1)
    logits = student_logits.squeeze(0)
    # under autocast the logits are bf16; the softmax runs in float32 at least
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, per_input[real])
    return {_COLUMN_KEY.format(k): log_probs[:, k].unsqueeze(0) for k in range(log_probs.shape[1])}


def padded_logprobs(model_output, data):
    """The engine's per-position output as padded response logprobs: [B, Rmax],
    or [B, Rmax, K] for a batch that carries [N, K] targets."""
    from verl.workers.utils.padding import no_padding_2_padding

    if "target_tokens" not in data.keys():
        return no_padding_2_padding(model_output["log_probs"], data)
    columns = [no_padding_2_padding(model_output[_COLUMN_KEY.format(k)], data)
               for k in range(data["target_tokens"].shape[-1])]
    return torch.stack(columns, dim=-1)


def _dp_size(data):
    from verl.utils import tensordict_utils as tu

    v = tu.get_non_tensor_data(data=data, key="dp_size", default=1)
    if hasattr(v, "data"):  # NonTensorData wrapper
        v = v.data
    return int(v) if v is not None else 1


def cross_entropy_loss(config, model_output, data, dp_group=None):
    """-sum(weights * logp(target)) * dp_size over every position (and column,
    for [N, K] targets), weights of any sign. Client owns normalization."""
    log_prob = padded_logprobs(model_output, data)  # [B, Rmax] or [B, Rmax, K]
    weights = data["weights"]
    if weights.is_nested:
        weights = weights.to_padded_tensor(0.0)
    loss = -(log_prob * weights).sum() * _dp_size(data)
    return loss, {"sum_weighted_logprob": -loss.detach().item() / max(_dp_size(data), 1)}


def importance_sampling_loss(config, model_output, data, dp_group=None):
    """RL: -sum(adv * exp(logp - logp_old)) * dp_size (Tinker importance_sampling).

    old_log_probs are the client-supplied sampling logprobs; per-token IS ratio,
    no clipping (clipping is client policy, not server policy).
    """
    log_prob = padded_logprobs(model_output, data)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    if old_log_prob.is_nested:
        old_log_prob = old_log_prob.to_padded_tensor(0.0)
    if advantages.is_nested:
        advantages = advantages.to_padded_tensor(0.0)
    mask = data["response_mask"]
    if mask.is_nested:
        mask = mask.to_padded_tensor(0)
    mask = mask.to(log_prob.dtype)

    ratio = torch.exp((log_prob - old_log_prob) * mask)
    loss = -(advantages * ratio * mask).sum() * _dp_size(data)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1)
        ppo_kl = ((old_log_prob - log_prob) * mask).sum() / denom
    return loss, {"ppo_kl": ppo_kl.detach().item()}


LOSS_FNS = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": importance_sampling_loss,
}


def get_loss_fn(name: str):
    if name not in LOSS_FNS:
        raise ValueError(f"verl backend: unsupported loss_fn {name!r} (have {sorted(LOSS_FNS)})")
    return LOSS_FNS[name]
