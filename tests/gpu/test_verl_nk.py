"""VerlBackend on real GPUs with the SmolLM3 student at full weights (LoRA rank
0): [N, K] cross-entropy against HF, the sampler following a full-weight step,
and the top-K logprobs of sampled tokens.

Needs 4 GPUs, verl with vLLM, and the model in the HF cache; starts its own
Ray instance. Skipped elsewhere. Each test prints what it measured (-s shows it;
a failure shows it too).

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_nk.py -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.base import SampleRequestError  # noqa: E402

from .verl_smollm3 import (  # noqa: E402
    K, STUDENT, adam, datum, hf_log_softmax, logprobs_of, text_tokens,
)

POSITIONS = 160
PROMPT_LEN = 32
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 64, "seed": 0}


@pytest.fixture(scope="module")
def tokens():
    return text_tokens(POSITIONS + 1)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def test_nk_forward_matches_hf(loop, student, tokens):
    backend, handle = student
    reference = hf_log_softmax(STUDENT, tokens[:-1])               # [N, V]
    ids = reference.topk(K, dim=-1).indices                        # HF's own top-16 at each position
    weights = torch.ones(POSITIONS, K)
    result = loop.run_until_complete(backend.forward(handle, [datum(tokens[:-1], ids, weights)], "cross_entropy"))
    got, want = logprobs_of(result), reference.gather(-1, ids)
    assert got.shape == (POSITIONS, K)
    diff = (got - want).abs()
    mean_abs = diff[1:].mean().item()
    corr = torch.corrcoef(torch.stack([got[1:].reshape(-1), want[1:].reshape(-1)]))[0, 1].item()
    print(f"\n[nk_forward] positions={POSITIONS} K={K} mean_abs(1:)={mean_abs:.5f} corr(1:)={corr:.6f} "
          f"max_abs(1:)={diff[1:].max().item():.4f} position0_mean_abs={diff[0].mean().item():.4f} "
          f"position0_max_abs={diff[0].max().item():.4f}")
    print("[nk_forward] mean_abs(1:) per column, best id first: "
          + " ".join(f"{v:.3f}" for v in diff[1:].mean(dim=0).tolist()))
    # printed, not judged: the same ids against HF in the engine's own precision
    bf16 = hf_log_softmax(STUDENT, tokens[:-1], dtype=torch.bfloat16, device="cuda:0").gather(-1, ids)
    print(f"[nk_forward] against HF bf16: backend mean_abs(1:)={(got - bf16).abs()[1:].mean().item():.5f}; "
          f"HF bf16 against HF fp32 mean_abs(1:)={(bf16 - want).abs()[1:].mean().item():.5f}")
    # limits for this stack: plain HF in bf16 reads 0.050 from fp32 over these 16 columns and the
    # backend 0.051 (measured 2026-11-10, both printed above); the best id alone reads 0.019
    assert diff[1:, 0].mean().item() <= 0.05
    assert mean_abs <= 0.08
    assert corr >= 0.995


def test_nk_forward_backward_signed_sum_and_k1(loop, student, tokens):
    backend, handle = student
    run = loop.run_until_complete
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 30000, (POSITIONS, K), generator=g)
    ids[:, 0] = torch.tensor(tokens[1:])
    weights = torch.randn(POSITIONS, K, generator=g)               # both signs
    result = run(backend.forward_backward(handle, [datum(tokens[:-1], ids, weights)], "cross_entropy"))
    logprobs = logprobs_of(result)
    want_loss = -(weights * logprobs).sum().item()
    loss = result["metrics"]["loss:sum"]
    signed_norm = run(backend.apply_optimizer_step(handle, adam(0.0)))["grad_norm"]
    print(f"\n[nk_forward_backward] loss:sum={loss:.4f} -sum(w*logprob)={want_loss:.4f} "
          f"rel={abs(loss - want_loss) / abs(want_loss):.2e} grad_norm={signed_norm:.2f}")
    assert logprobs.shape == (POSITIONS, K)
    assert loss == pytest.approx(want_loss, rel=1e-3)

    # K == 1 against the [N] call on the same ids; lr 0 leaves the weights where they were
    next_ids, ones = torch.tensor(tokens[1:]), torch.ones(POSITIONS)
    outs = {}
    for name, d in (("k1", datum(tokens[:-1], next_ids.unsqueeze(-1), ones.unsqueeze(-1))),
                    ("n", datum(tokens[:-1], next_ids, ones))):
        fb = run(backend.forward_backward(handle, [d], "cross_entropy"))
        outs[name] = (logprobs_of(fb).reshape(-1), run(backend.apply_optimizer_step(handle, adam(0.0)))["grad_norm"])
    (lp_k1, norm_k1), (lp_n, norm_n) = outs["k1"], outs["n"]
    print(f"[nk_forward_backward] K=1 vs [N]: logprobs rel={relative(lp_k1, lp_n):.2e} "
          f"max_abs={(lp_k1 - lp_n).abs().max().item():.2e} grad_norm {norm_k1:.3f} vs {norm_n:.3f} "
          f"rel={abs(norm_k1 - norm_n) / norm_n:.2e}")
    assert relative(lp_k1, lp_n) <= 1e-3
    assert norm_k1 == pytest.approx(norm_n, rel=1e-3)


def test_topk_sample_logprobs(loop, student, tokens):
    backend, handle = student
    run = loop.run_until_complete
    prompt = tokens[:PROMPT_LEN]
    assert handle.max_topk_logprobs >= K
    sample = run(backend.sample(handle, "topk", prompt, 1, dict(SAMPLING), topk_sample_logprobs=K))
    seq = sample["sequences"][0]
    head = seq["topk_sample_logprobs"]
    assert len(head) == len(seq["tokens"]) and all(len(row) == K for row in head)
    head_ids = torch.tensor([[t for t, _ in row] for row in head])
    head_lps = torch.tensor([[lp for _, lp in row] for row in head])
    assert (head_lps[:, :-1] >= head_lps[:, 1:]).all(), "each position's pairs come best first"
    in_head = 0
    for token, logprob, row in zip(seq["tokens"], seq["logprobs"], head):
        for t, lp in row:
            if t == token:
                in_head += 1
                assert lp == pytest.approx(logprob, abs=1e-5)
    assert in_head > 0, "no sampled token was in its own top-16: the equality above checked nothing"

    # the trainer's [N, K] forward on the same ids; prompt positions carry the next token, unread
    sampled = len(seq["tokens"])
    full = prompt + list(seq["tokens"])
    ids = torch.tensor(full[1:]).unsqueeze(-1).repeat(1, K)
    ids[-sampled:] = head_ids
    result = run(backend.forward(handle, [datum(full[:-1], ids, torch.ones(ids.shape))], "cross_entropy"))
    trainer = logprobs_of(result)[-sampled:]
    diff = (trainer - head_lps).abs()
    mean_abs, best_id = diff.mean().item(), diff[:, 0].mean().item()
    print(f"\n[topk] sampled={sampled} in_head={in_head} cap={handle.max_topk_logprobs} "
          f"trainer_vs_sampler mean_abs={mean_abs:.5f} best_id_mean_abs={best_id:.5f} max_abs={diff.max().item():.4f}")
    print("[topk] mean_abs per column, best id first: " + " ".join(f"{v:.3f}" for v in diff.mean(dim=0).tolist()))

    # the cap on the handle is one the engine serves
    at_cap = run(backend.sample(handle, "cap", prompt, 1, {**SAMPLING, "max_tokens": 4},
                                topk_sample_logprobs=handle.max_topk_logprobs))
    assert all(len(row) == handle.max_topk_logprobs for row in at_cap["sequences"][0]["topk_sample_logprobs"])

    with pytest.raises(SampleRequestError, match="temperature > 0"):
        run(backend.sample(handle, "greedy_topk", prompt, 1, {**SAMPLING, "temperature": 0.0}, topk_sample_logprobs=K))
    with pytest.raises(SampleRequestError, match="topk_prompt_logprobs"):
        run(backend.sample(handle, "prompt_topk", prompt, 1, dict(SAMPLING), topk_prompt_logprobs=1))
    assert best_id <= 0.05
    assert mean_abs <= 0.08


def test_full_finetune_sampler_follows(loop, student, tokens):
    """Runs last in the module: it moves the student's weights."""
    backend, handle = student
    run = loop.run_until_complete
    prompt = tokens[:PROMPT_LEN]

    def sample_and_score(tag):
        seq = run(backend.sample(handle, tag, prompt, 1, dict(SAMPLING)))["sequences"][0]
        full = prompt + list(seq["tokens"])
        d = datum(full[:-1], torch.tensor(full[1:]), torch.ones(len(full) - 1))
        trainer = logprobs_of(run(backend.forward(handle, [d], "cross_entropy")))[-len(seq["tokens"]):]
        return list(seq["tokens"]), (trainer - torch.tensor(seq["logprobs"])).abs().mean().item()

    before, diff_before = sample_and_score("before")
    step = datum(tokens[:-1], torch.tensor(tokens[1:]), torch.ones(POSITIONS))
    run(backend.forward_backward(handle, [step], "cross_entropy"))
    grad_norm = run(backend.apply_optimizer_step(handle, adam(1e-5)))["grad_norm"]
    after, diff_after = sample_and_score("after")
    print(f"\n[full_finetune] trainer_vs_sampler mean_abs before={diff_before:.5f} after={diff_after:.5f} "
          f"grad_norm={grad_norm:.1f} tokens_changed={before != after}")
    assert before != after
    assert diff_before <= 0.05 and diff_after <= 0.05
