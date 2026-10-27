"""AutomodelWorker on real GPUs (Qwen2.5-0.5B-Instruct, LoRA rank 8).

Needs >= 4 GPUs, the Automodel training env and the model in the HF cache;
starts its own Ray instance. Skipped elsewhere.

Run on a GPU pod (about 7 minutes for the 6 tests):
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_worker.py -v

When extending: a worker subclass defined at this module's top level fails to
import inside the Ray actor process; define it inside a function so it is
pickled by value. An fp32-compute variant needs attn_implementation="sdpa";
FlashAttention takes only fp16/bf16.
"""

import pytest

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import ray  # noqa: E402
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs >= 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.automodel.adapter import OPTIMIZER_DIRNAME, OPTIMIZER_FILE, peft_key  # noqa: E402
from tinkercloud.training.backends.automodel.converter import coalesce, convert_request, row_sums, shard  # noqa: E402
from tinkercloud.training.checkpoints import interchange as ci  # noqa: E402
from tinkercloud.training.backends.automodel.lora import LoraSpec, lora_spec  # noqa: E402
from tinkercloud.training.backends.automodel.worker import AutomodelWorker  # noqa: E402
from tinkercloud.training.backends.automodel.errors import NonFiniteGradNorm  # noqa: E402
from tinkercloud.training.models.requests import AdamParams, LoraConfig  # noqa: E402
from tests.datums import random_ce_datum  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA: LoraSpec = lora_spec(LoraConfig(rank=8, alpha=16))
SEED = 1234


class Group:
    """n ranks with the tenants {name: seed} or {name: (seed, LoraSpec)}
    added in order (the last one is live); one tenant "t" with LORA by
    default. Every op names its tenant."""
    def __init__(self, n, micro_batch_rows, tenants=None):
        cls = ray.remote(num_gpus=1, num_cpus=1)(AutomodelWorker)
        self.n = n
        self.ranks = [cls.remote(r, n, MODEL, LORA, micro_batch_rows) for r in range(n)]
        addr = ray.get(self.ranks[0].rendezvous_address.remote())
        self.trainable = ray.get([w.setup.remote(addr) for w in self.ranks])[0]
        for name, spec in (tenants or {"t": SEED}).items():
            seed, lora = spec if isinstance(spec, tuple) else (spec, LORA)
            self.call("add_tenant", before=(name, lora, seed))
        self.last_swap = 0.0    # seconds the last pass or step spent swapping (the slowest rank)

    def call(self, method, per_rank=(), *, before=(), after=()):
        """method(*before, <this rank's entry of each per_rank sequence>, *after) on every rank."""
        refs = [getattr(w, method).remote(*before, *[a[i] for a in per_rank], *after) for i, w in enumerate(self.ranks)]
        return ray.get(refs)

    def fwd_bwd(self, batch, clip=None, method="forward_backward", tenant="t"):
        outs = self.call(method, (shard(batch, self.n),), before=(tenant,), after=(clip,))
        self.last_swap = max(o["swap_seconds"] for o in outs)
        return {k: torch.cat([o[k] for o in outs]) for k in ("row_loss", "logprobs", "elementwise", "clamped")}

    def optim(self, adam, tenant="t"):
        outs = self.call("optim_step", before=(tenant, adam))
        norms = [o["grad_norm"] for o in outs]
        assert len(set(norms)) == 1, f"ranks disagree on the grad norm: {norms}"
        self.last_swap = max(o["swap_seconds"] for o in outs)
        return norms[0]

    def lora(self, tenant="t"):
        return self.call("lora_state", before=(tenant,))[0]

    def kill(self):
        for w in self.ranks:
            ray.kill(w)


def requests():
    g = torch.Generator().manual_seed(0)
    x = convert_request([random_ce_datum(g, n) for n in (37, 64, 5)], "cross_entropy")
    y = convert_request([random_ce_datum(g, n) for n in (80, 21, 50, 9, 33)], "cross_entropy")
    return x, y


ADAM = AdamParams(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
# beta1 = beta2 = 0 and eps >> |g| make AdamW's update lr * g / eps: linear in
# the (clipped) grad.
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


def test_nonfinite_grad_norm_is_refused():
    """Refused on every rank, grads zeroed, LoRA unchanged; the group then
    trains on."""
    x, _ = requests()
    batch = coalesce([x], 2)
    grp = Group(2, 2)
    try:
        before = grp.lora()
        grp.fwd_bwd(with_weight(batch, 3e38))
        with pytest.raises(NonFiniteGradNorm):
            grp.optim(ADAM)
        after = grp.lora()
        for k in before:
            assert torch.equal(before[k], after[k])
        assert grp.optim(ADAM) == 0.0  # the grads were zeroed: nothing left to step on
        grp.fwd_bwd(batch)
        assert grp.optim(ADAM) > 0  # the group still trains after a refusal
    finally:
        grp.kill()


def test_clip_scales_the_update():
    """Two fresh groups from the same init, one step each on the same batch:
    unclipped vs clipped at a tenth of the norm. A group's computation is
    deterministic, so both see the same grad, and under LINEAR the clipped
    B update is exactly a tenth. One group taking both steps would not do:
    the first step moves the state, and the second grad differs from it by
    up to ~1% under bf16 compute, at any step size."""
    x, _ = requests()
    batch = coalesce([x], 2)
    deltas, norms = [], []
    for clip_of in (None, 0.1):
        grp = Group(2, 2)
        try:
            s0 = grp.lora()
            grp.fwd_bwd(batch)
            clip = 0.0 if clip_of is None else norms[0] * clip_of
            norms.append(grp.optim(AdamParams(**LINEAR_KW, grad_clip_norm=clip)))
            s1 = grp.lora()
        finally:
            grp.kill()
        b_names = [k for k in s0 if ".lora_B." in k]
        deltas.append(torch.cat([(s1[k] - s0[k]).flatten() for k in b_names]))
    assert norms[0] > 0
    assert norms[1] == norms[0]  # the reported norm is before clipping
    torch.testing.assert_close(deltas[1], deltas[0] * 0.1, rtol=1e-4, atol=0)


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


def test_checkpoint_round_trip_restores_weights_and_optimizer(tmp_path):
    """save_state after one step; a fresh group that loads weights and
    optimizer reproduces the weights exactly and the next step to 1e-6; one
    that loads weights only takes a different next step (fresh Adam moments)."""
    from safetensors.torch import load_file

    x, _ = requests()
    batch = coalesce([x], 2)
    root = tmp_path / "ckpt"
    grp = Group(2, 2)
    try:
        grp.fwd_bwd(batch)
        grp.optim(ADAM)
        grp.call("save_state", before=("t", str(root)))
        s1 = grp.lora()
        grp.fwd_bwd(batch)
        grp.optim(ADAM)
        s2 = grp.lora()                     # the uninterrupted second step
    finally:
        grp.kill()
    adapter = ci.find_hf_adapter(str(root))
    assert adapter is not None
    saved = load_file(f"{adapter}/{ci.ADAPTER_WEIGHTS_FILE}")
    assert set(saved) == {peft_key(k) for k in s1}
    assert all(saved[peft_key(k)].dtype == torch.float32 and torch.equal(saved[peft_key(k)], s1[k]) for k in s1)
    assert (root / OPTIMIZER_DIRNAME / OPTIMIZER_FILE).is_file()

    grp = Group(2, 2)
    try:
        grp.call("load_state", before=("t", adapter, str(root / OPTIMIZER_DIRNAME)))
        r1 = grp.lora()
        assert all(torch.equal(r1[k], s1[k]) for k in s1)
        grp.fwd_bwd(batch)
        grp.optim(ADAM)
        r2 = grp.lora()
    finally:
        grp.kill()
    for k in s2:
        torch.testing.assert_close(r2[k], s2[k], atol=1e-6, rtol=0)

    grp = Group(2, 2)
    try:
        grp.call("load_state", before=("t", adapter, None))
        grp.fwd_bwd(batch)
        grp.optim(ADAM)
        w2 = grp.lora()
    finally:
        grp.kill()
    b_names = [k for k in s2 if ".lora_B." in k]
    assert any(not torch.allclose(w2[k], s2[k], atol=1e-6, rtol=0) for k in b_names)


# --- tenants of one group (the adapter store) --------------------------------

def equal_states(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[k], b[k]) for k in a)


def test_tenant_init_is_per_seed_and_equals_the_solo_init():
    """Two tenants with different seeds start from different A matrices, and
    each starts from exactly what a solo group with its seed builds."""
    solo = Group(2, 2, {"a": 1})
    try:
        ia = solo.lora("a")
    finally:
        solo.kill()
    solo = Group(2, 2, {"b": 2})
    try:
        ib = solo.lora("b")
    finally:
        solo.kill()
    shared = Group(2, 2, {"a": 1, "b": 2})   # b is live, a waits in its slot
    try:
        sa, sb = shared.lora("a"), shared.lora("b")
    finally:
        shared.kill()
    assert equal_states(sa, ia) and equal_states(sb, ib)
    a_names = [k for k in ia if ".lora_A." in k]
    assert all(not torch.equal(ia[k], ib[k]) for k in a_names)


def test_tenant_interleaved_passes_equal_the_solo_run_and_leave_the_other_untouched(tmp_path):
    """fwd_bwd(A), fwd_bwd(B), fwd_bwd(A), optim(A) is A's solo
    fwd_bwd, fwd_bwd, optim exactly; B's weights and optimizer state are as
    they were, its grads wait in its slot, and its first step is its solo one."""
    from safetensors.torch import load_file

    x, y = requests()
    bx, by = coalesce([x], 2), coalesce([y], 2)
    solo = Group(2, 2, {"a": 1})
    try:
        solo.fwd_bwd(bx, tenant="a")
        solo.fwd_bwd(by, tenant="a")
        norm_a = solo.optim(ADAM, "a")
        weights_a = solo.lora("a")
    finally:
        solo.kill()
    solo = Group(2, 2, {"b": 2})
    try:
        solo.fwd_bwd(bx, tenant="b")
        norm_b = solo.optim(ADAM, "b")
    finally:
        solo.kill()

    shared = Group(2, 2, {"a": 1, "b": 2})
    try:
        init_b = shared.lora("b")
        shared.fwd_bwd(bx, tenant="a")
        shared.fwd_bwd(bx, tenant="b")
        shared.fwd_bwd(by, tenant="a")
        assert shared.optim(ADAM, "a") == norm_a
        assert equal_states(shared.lora("a"), weights_a)
        shared.call("save_state", before=("b", str(tmp_path / "b")))
        saved = load_file(str(tmp_path / "b" / ci.HF_ADAPTER_DIRNAME / ci.ADAPTER_WEIGHTS_FILE))
        assert all(torch.equal(saved[peft_key(k)], init_b[k]) for k in init_b)
        assert load_file(str(tmp_path / "b" / OPTIMIZER_DIRNAME / OPTIMIZER_FILE)) == {}, "b has not stepped"
        assert shared.optim(ADAM, "b") == norm_b
    finally:
        shared.kill()


def test_tenant_dropout_is_the_tenants_not_the_groups():
    """b (dropout 0.5) shares a's group (dropout 0): after one step of b (B
    is zero at init, so the LoRA branch dropout drops is zero until then),
    two train passes of b on one batch score differently, two of a score
    the same, and b's eval forwards score the same (no dropout in eval)."""
    from dataclasses import replace

    x, _ = requests()
    bx = coalesce([x], 2)
    shared = Group(2, 2, {"a": 1, "b": (2, replace(LORA, dropout=0.5))})
    try:
        shared.fwd_bwd(bx, tenant="b")
        shared.optim(ADAM, "b")
        b1 = shared.fwd_bwd(bx, tenant="b")["row_loss"]
        b2 = shared.fwd_bwd(bx, tenant="b")["row_loss"]
        assert not torch.equal(b1, b2), "b trains with its dropout"
        a1 = shared.fwd_bwd(bx, tenant="a")["row_loss"]
        a2 = shared.fwd_bwd(bx, tenant="a")["row_loss"]
        assert torch.equal(a1, a2), "a trains without dropout, after b"
        e1 = shared.fwd_bwd(bx, tenant="b", method="forward")["row_loss"]
        e2 = shared.fwd_bwd(bx, tenant="b", method="forward")["row_loss"]
        assert torch.equal(e1, e2), "eval has no dropout"
    finally:
        shared.kill()


def test_tenant_eval_forward_in_between_leaves_the_step_and_the_scores_solo():
    """After a step of A: forward(B), forward(A) scores A's weights, not the
    copies B's forward left unsharded; fwd_bwd(A), forward(B), optim(A)
    reports A's solo grad norm. (A must have stepped: at init B = 0 and every
    tenant scores as the base model.)"""
    x, _ = requests()
    bx = coalesce([x], 2)
    solo = Group(2, 2, {"a": 1})
    try:
        solo.fwd_bwd(bx, tenant="a")
        norm_1 = solo.optim(ADAM, "a")
        scores_a = solo.fwd_bwd(bx, method="forward", tenant="a")["row_loss"]
        solo.fwd_bwd(bx, tenant="a")
        norm_2 = solo.optim(ADAM, "a")
    finally:
        solo.kill()
    shared = Group(2, 2, {"a": 1, "b": 2})
    try:
        shared.fwd_bwd(bx, tenant="a")
        assert shared.optim(ADAM, "a") == norm_1
        scores_b = shared.fwd_bwd(bx, method="forward", tenant="b")["row_loss"]
        assert not torch.equal(scores_b, scores_a), "b is at init, a has stepped"
        assert torch.equal(shared.fwd_bwd(bx, method="forward", tenant="a")["row_loss"], scores_a)
        shared.fwd_bwd(bx, tenant="a")
        shared.fwd_bwd(bx, method="forward", tenant="b")
        assert shared.optim(ADAM, "a") == norm_2
    finally:
        shared.kill()


def test_tenant_export_and_save_read_the_slot_without_swap(tmp_path):
    """With A live, export_adapter(B) and save_state(B) write B's weights and
    leave A live: A's next pass swaps nothing, B's next op does."""
    from safetensors.torch import load_file

    x, _ = requests()
    bx = coalesce([x], 2)
    shared = Group(2, 2, {"a": 1, "b": 2})   # b live
    try:
        shared.fwd_bwd(bx, tenant="a")
        assert shared.last_swap > 0, "a came from its slot"
        shared.call("export_adapter", before=("b", str(tmp_path / "export")))
        shared.call("save_state", before=("b", str(tmp_path / "save")))
        weights_b = shared.lora("b")
        for path in (tmp_path / "export", tmp_path / "save" / ci.HF_ADAPTER_DIRNAME):
            written = load_file(str(path / ci.ADAPTER_WEIGHTS_FILE))
            assert all(torch.equal(written[peft_key(k)], weights_b[k]) for k in weights_b)
        shared.fwd_bwd(bx, tenant="a")
        assert shared.last_swap == 0, "a stayed live through b's export and save"
        shared.optim(ADAM, "b")
        assert shared.last_swap > 0
    finally:
        shared.kill()
