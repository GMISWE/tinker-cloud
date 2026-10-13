"""AutomodelWorker on real GPUs (Qwen2.5-0.5B-Instruct, LoRA rank 8).

Needs >= 4 GPUs, the Automodel training env and the model in the HF cache;
starts its own Ray instance. Not part of the CPU suites.
"""
import os

import pytest
import ray
import torch

from tinkercloud.training.backends.automodel.converter import coalesce, convert_request, row_sums, shard
from tinkercloud.training.backends.automodel.lora import LoraSpec, lora_spec
from tinkercloud.training.backends.automodel.worker import AutomodelWorker, NonFiniteGradNorm
from tinkercloud.training.models.requests import AdamParams, Datum, LoraConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA: LoraSpec = lora_spec(LoraConfig(rank=8, alpha=16))
SEED = 1234


@pytest.fixture(scope="module", autouse=True)
def ray_local():
    ray.init(num_gpus=torch.cuda.device_count(), runtime_env={"env_vars": {
        "PYTHONPATH": os.environ["PYTHONPATH"], "HF_HUB_OFFLINE": "1"}})
    yield
    ray.shutdown()


class Group:
    def __init__(self, n, micro_batch_rows):
        cls = ray.remote(num_gpus=1, num_cpus=1)(AutomodelWorker)
        self.n = n
        self.ranks = [cls.remote(r, n, MODEL, LORA, SEED, micro_batch_rows) for r in range(n)]
        addr = ray.get(self.ranks[0].rendezvous_address.remote())
        self.trainable = ray.get([w.setup.remote(addr) for w in self.ranks])[0]

    def call(self, method, *per_rank_args, same=()):
        refs = [getattr(w, method).remote(*[a[i] for a in per_rank_args], *same) for i, w in enumerate(self.ranks)]
        return ray.get(refs)

    def fwd_bwd(self, batch, clip=None, method="forward_backward"):
        outs = self.call(method, shard(batch, self.n), same=(clip,))
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}

    def optim(self, adam):
        norms = [o["grad_norm"] for o in self.call("optim_step", same=(adam,))]
        assert len(set(norms)) == 1, f"ranks disagree on the grad norm: {norms}"
        return norms[0]

    def lora(self):
        return self.call("lora_state")[0]

    def kill(self):
        for w in self.ranks:
            ray.kill(w)


def ce_datum(g, length):
    tokens = torch.randint(100, 20000, (length,), generator=g).tolist()
    targets = tokens[1:] + [tokens[0]]
    weights = [0.0] * (length // 3) + [1.0] * (length - length // 3)
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": targets}, "weights": {"data": weights}}})


def requests():
    g = torch.Generator().manual_seed(0)
    x = convert_request([ce_datum(g, n) for n in (37, 64, 5)], "cross_entropy")
    y = convert_request([ce_datum(g, n) for n in (80, 21, 50, 9, 33)], "cross_entropy")
    return x, y


ADAM = AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
# beta1 = beta2 = 0 and eps >> |g| make AdamW's update lr * g / eps: linear in
# the (clipped) grad. lr / eps = 1e-7 keeps the state nearly still between steps.
LINEAR_KW = dict(beta1=0.0, beta2=0.0, eps=1e8, weight_decay=0.0, learning_rate=1e1)
LINEAR = AdamParams(**LINEAR_KW, grad_clip_norm=0.0)


def test_rank_count_micro_batching_and_coalescing_are_exact():
    """2 ranks, 2-row micro-batches, X and Y as two passes  ==  4 ranks,
    1-row micro-batches, X and Y coalesced: same init, losses, grad norms and
    LoRA updates over two optimizer steps. The steps use LINEAR (update
    proportional to the grad): real Adam's first update is lr * sign(g), and
    bf16 noise flips the sign of near-zero grads between the two runs."""
    x, y = requests()
    runs = {}
    for name, n, mbr, passes in (("a", 2, 2, [[x], [y]]), ("b", 4, 1, [[x, y]])):
        grp = Group(n, mbr)
        try:
            init = grp.lora()
            losses, norms = [], []
            for _ in range(2):
                step_losses = []
                for reqs in passes:
                    batch = coalesce(reqs, n)
                    step_losses += row_sums(batch, grp.fwd_bwd(batch)["row_loss"])
                losses.append(step_losses)
                norms.append(grp.optim(LINEAR))
            runs[name] = (grp.trainable, init, losses, norms, grp.lora())
        finally:
            grp.kill()
    (ta, ia, la, na, fa), (tb, ib, lb, nb, fb) = runs["a"], runs["b"]
    assert ta == tb > 0
    assert ia.keys() == ib.keys()
    for k in ia:
        torch.testing.assert_close(ia[k], ib[k], rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(la), torch.tensor(lb), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(torch.tensor(na), torch.tensor(nb), rtol=2e-2, atol=0)
    assert na[0] > 0
    ua = torch.cat([(fa[k] - ia[k]).flatten() for k in ia])
    ub = torch.cat([(fb[k] - ib[k]).flatten() for k in ia])
    assert ua.norm() > 0
    assert ((ua - ub).norm() / ua.norm()).item() < 2e-2


def test_loss_matches_unsharded_hf_reference_at_init():
    """At init B = 0, so the adapted model is the base model: the group's
    per-row CE sum equals an unsharded HF forward's."""
    from transformers import AutoModelForCausalLM

    x, _ = requests()
    batch = coalesce([x], 2)
    grp = Group(2, 2)
    try:
        got = grp.fwd_bwd(batch, method="forward")
    finally:
        grp.kill()
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        for i in range(len(x.tokens)):
            ids = x.tokens[i].cuda()[None]
            lp = torch.log_softmax(ref_model(input_ids=ids).logits.float(), -1)[0]
            lp = lp.gather(-1, x.inputs["target_tokens"][i].cuda()[:, None]).squeeze(-1)
            want = -(x.inputs["weights"][i].cuda() * lp).sum().item()
            assert got["row_loss"][i].item() == pytest.approx(want, rel=2e-2)
    del ref_model


def test_clip_and_nonfinite_refusal():
    x, _ = requests()
    batch = coalesce([x], 2)
    grp = Group(2, 2)
    try:
        # a non-finite grad: refused on every rank, grads zeroed, LoRA unchanged
        before = grp.lora()
        poisoned = with_weight(batch, 3e38)
        grp.fwd_bwd(poisoned)
        with pytest.raises(NonFiniteGradNorm):
            grp.optim(ADAM)
        after = grp.lora()
        for k in before:
            assert torch.equal(before[k], after[k])
        assert grp.optim(ADAM) == 0.0  # the grads were zeroed: nothing left to step on

        # clipping: under LINEAR, B's change is proportional to the clipped grad
        b_names = [k for k in before if ".lora_B." in k]
        s0 = grp.lora()
        grp.fwd_bwd(batch)
        norm = grp.optim(LINEAR)
        s1 = grp.lora()
        grp.fwd_bwd(batch)
        clipped_norm = grp.optim(AdamParams(**LINEAR_KW, grad_clip_norm=norm / 10))
        s2 = grp.lora()
        assert clipped_norm == pytest.approx(norm, rel=2e-2)  # the reported norm is before clipping
        d1 = torch.cat([(s1[k] - s0[k]).flatten() for k in b_names]).norm()
        d2 = torch.cat([(s2[k] - s1[k]).flatten() for k in b_names]).norm()
        assert (d2 / d1).item() == pytest.approx(0.1, rel=2e-2)
    finally:
        grp.kill()


def test_one_rank_group_runs():
    """Automodel leaves a 1-rank model without FSDP2 (no mixed-precision policy)."""
    x, _ = requests()
    batch = coalesce([x], 1)
    grp = Group(1, 2)
    try:
        out = grp.fwd_bwd(batch)
        assert torch.isfinite(out["row_loss"]).all()
        assert grp.optim(ADAM) > 0
    finally:
        grp.kill()


def with_weight(batch, w):
    """The same batch with every CE weight set to w where it was nonzero."""
    from dataclasses import replace
    weights = batch.inputs["weights"]
    return replace(batch, inputs={**batch.inputs, "weights": torch.where(weights != 0, torch.full_like(weights, w), weights)})
