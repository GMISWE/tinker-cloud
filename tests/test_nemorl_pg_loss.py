"""Tinker's RL losses on the NeMo RL backend: pure sum, ratio against the client's
sampling logprobs, no clip and no KL for importance_sampling, the client's clip for ppo."""
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nemo_rl") is None:
    pytest.skip("torch / nemo_rl not installed", allow_module_level=True)

import torch

if not torch.cuda.is_available():
    pytest.skip("ClippedPGLossFn gathers on CUDA", allow_module_level=True)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

DEV = "cuda"

from tinkercloud.training.backends.nemo_rl.loss_config import NO_CLIP, NO_GRAD_CLIP, TINKER_PG_LOSS_DEFAULTS
from tinkercloud.training.backends.nemo_rl.losses import TinkerSumPGLoss

B, S, V = 2, 6, 16


def _batch(seed=0, mismatch=0.05):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, S, V, generator=g).to(DEV)
    input_ids = torch.randint(0, V, (B, S), generator=g).to(DEV)
    curr = torch.log_softmax(logits.float(), -1)[:, :-1].gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    prev = torch.zeros(B, S, device=DEV)
    prev[:, 1:] = curr + mismatch * torch.randn(B, S - 1, generator=g).to(DEV)  # sampler logprobs != trainer's
    adv = torch.randn(B, S, generator=g).to(DEV)
    token_mask = torch.ones(B, S, device=DEV)
    token_mask[0, :3] = 0  # prompt tokens carry no loss
    data = BatchedDataDict({
        "input_ids": input_ids, "advantages": adv, "prev_logprobs": prev,
        "generation_logprobs": prev.clone(), "reference_policy_logprobs": torch.zeros(B, S, device=DEV),
        "token_mask": token_mask, "sample_mask": torch.ones(B, device=DEV),
    })
    mask = token_mask[:, 1:]
    return logits, data, curr, prev[:, 1:], adv[:, 1:], mask


def _call(fn, logits, data, mask):
    loss, metrics = fn(logits, data, torch.tensor(float(B), device=DEV), mask.sum())
    return loss


@pytest.mark.parametrize("mismatch", [0.05, 3.0])
def test_importance_sampling_is_the_documented_sum(mismatch):
    logits, data, curr, prev, adv, mask = _batch(mismatch=mismatch)
    loss = _call(TinkerSumPGLoss(TINKER_PG_LOSS_DEFAULTS), logits, data, mask)
    expected = -(torch.exp(curr - prev) * adv * mask).sum()  # no clip even at ratio ~ e^3
    torch.testing.assert_close(loss, expected)


def test_reference_logprobs_do_not_enter_the_loss():
    logits, data, curr, prev, adv, mask = _batch()
    a = _call(TinkerSumPGLoss(TINKER_PG_LOSS_DEFAULTS), logits, data, mask)
    data["reference_policy_logprobs"] = torch.randn(B, S, device=DEV)
    b = _call(TinkerSumPGLoss(TINKER_PG_LOSS_DEFAULTS), logits, data, mask)
    torch.testing.assert_close(a, b)


def test_ppo_clips_with_the_client_bounds_only():
    logits, data, curr, prev, adv, mask = _batch(mismatch=0.5)
    fn = TinkerSumPGLoss({**TINKER_PG_LOSS_DEFAULTS, "ratio_clip_min": 0.2, "ratio_clip_max": 0.2})
    loss = _call(fn, logits, data, mask)
    ratio = torch.exp(curr - prev)
    expected = -(torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv) * mask).sum()
    torch.testing.assert_close(loss, expected)
    assert not torch.allclose(loss, -(ratio * adv * mask).sum())  # the clip did bind somewhere


def test_normalizer_is_one_regardless_of_batch_counts():
    logits, data, curr, prev, adv, mask = _batch()
    fn = TinkerSumPGLoss(TINKER_PG_LOSS_DEFAULTS)
    a, _ = fn(logits, data, torch.tensor(2.0, device=DEV), torch.tensor(7.0, device=DEV))
    b, _ = fn(logits, data, torch.tensor(200.0, device=DEV), torch.tensor(7000.0, device=DEV))
    torch.testing.assert_close(a, b)


def test_builder_defaults_follow_the_api():
    from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
    cfg, _ = NemoRLArgumentBuilder().build_args(base_model="Qwen/Qwen2.5-0.5B", num_gpus=1)
    lf = cfg["loss_fn"]
    assert lf["reference_policy_kl_penalty"] == 0.0
    assert (lf["ratio_clip_min"], lf["ratio_clip_max"], lf["ratio_clip_c"]) == (NO_CLIP["ratio_clip_min"], NO_CLIP["ratio_clip_max"], None)
    assert lf["use_importance_sampling_correction"] is False and lf["disable_ppo_ratio"] is False
    assert cfg["policy"]["max_grad_norm"] == NO_GRAD_CLIP  # never binds; None would zero the reported grad norm
