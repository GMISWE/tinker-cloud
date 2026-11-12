"""VerlBackend on real GPUs: an optimizer step applies the client's own values (the gradient clip, weight
decay, beta1), and a sample with no stop ends at the model's EOS.

The update is not visible through the API, so the tests make it visible: with a very large `eps` and
`beta1 = 0` an Adam step moves the weights by `lr / eps` times the applied gradient (a "linear step"), and the
datum's loss then falls by about `(lr / eps) * g * a`, `g` the gradient norm and `a` the applied gradient's.
The clip test sums `STEPS` such steps: single steps with no clip read between 0.09 and 1.27 of that value
(2026-11-11; a low reading was followed by a high one, so an error in each loss reading that cancels over a
run is the likely cause, unverified). The summed drop reads about 0.6 of the first-order value (2026-11-11);
likely the loss's curvature, unverified.

Needs 4 GPUs, verl with vLLM, and the model in the HF cache; starts its own Ray instance. Skipped elsewhere.
Each test prints what it measured (-s shows it; a failure shows it too). The order matters: the weight-decay
test shrinks every weight by 1%, so the stop and clip tests run before it.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_optim.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.verl.builder import VerlArgumentBuilder  # noqa: E402
from tinkercloud.training.models.requests import AdamParams  # noqa: E402

from .verl_smollm3 import STUDENT, datum, logprobs_of, text_tokens  # noqa: E402

POSITIONS = 40
LINEAR_EPS = 1e6        # far above any gradient element, so the update is lr / eps times the gradient
LINEAR_LR = 0.0225      # lr / eps = 2.25e-8
STEPS = 8
# summed loss drop / its first-order value. An ignored clip reads several times the band's top in the clipped
# case; a clip fixed at 1.0 reads 0 in the unclipped one.
RATIO_BAND = (0.3, 1.3)
DECAY_LR = 1e-2         # with weight_decay 1.0: the weights shrink by 1%
DECAY_FLOOR = 1e-3      # mean abs logprob change that shrink must exceed
SAME = 1e-6


@pytest.fixture(scope="module")
def steps(loop, student):
    """The calls the tests are written in: loss and gradient of the one datum, a step, the logprobs."""
    backend, handle = student
    run = loop.run_until_complete
    tokens = text_tokens(POSITIONS + 1)

    class Steps:
        @staticmethod
        def backward(weight: float) -> float:
            """Accumulate the datum's gradient at `weight` per token; returns its `loss:sum`."""
            d = datum(tokens[:-1], torch.tensor(tokens[1:]), torch.full((POSITIONS,), weight))
            return run(backend.forward_backward(handle, [d], "cross_entropy"))["metrics"]["loss:sum"]

        @staticmethod
        def step(lr: float, *, beta1: float, weight_decay: float, clip: float, eps: float) -> float:
            """Apply the accumulated gradient; returns the step's `grad_norm`."""
            params = AdamParams(learning_rate=lr, beta1=beta1, beta2=0.999, eps=eps, weight_decay=weight_decay,
                                grad_clip_norm=clip)
            return run(backend.apply_optimizer_step(handle, params))["grad_norm"]

        @staticmethod
        def linear(lr: float, *, clip: float) -> float:
            """A linear step (module docstring) with the gradient clipped at `clip`; lr 0 only reads the norm."""
            return Steps.step(lr, beta1=0.0, weight_decay=0.0, clip=clip, eps=LINEAR_EPS)

        @staticmethod
        def logprobs() -> torch.Tensor:
            d = datum(tokens[:-1], torch.tensor(tokens[1:]), torch.ones(POSITIONS))
            return logprobs_of(run(backend.forward(handle, [d], "cross_entropy")))

    return Steps


def test_stop_is_the_models_eos_when_none_is_sent(loop, student):
    """Regression guarded: the sampler stops on another token, or not at all, when the client names none."""
    from transformers import AutoTokenizer, GenerationConfig

    backend, handle = student
    model_dir = VerlArgumentBuilder.resolve_hf_path(STUDENT)
    eos = GenerationConfig.from_pretrained(model_dir).eos_token_id
    prompt = list(AutoTokenizer.from_pretrained(model_dir).apply_chat_template(
        [{"role": "user", "content": "What is 17 + 26? Reply with the number only."}], add_generation_prompt=True,
        tokenize=True, return_dict=True)["input_ids"])
    for seed in range(8):
        sampling = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 768, "seed": seed}
        seq = loop.run_until_complete(backend.sample(handle, f"stop{seed}", prompt, 1, sampling))["sequences"][0]
        tokens = list(seq["tokens"])
        print(f"\n[stop] seed {seed}: {len(tokens)} tokens, stop_reason={seq['stop_reason']}, last {tokens[-3:]}, eos {eos}")
        assert seq["stop_reason"] == "stop"
        assert tokens[-1] == eos
        assert eos not in tokens[:-1]


def test_clip_follows_the_step(steps):
    """Regression guarded: every step is clipped at the engine's build-time norm, 1.0, whatever the step asks."""
    def summed_drop(lr: float, clip_over_g: float):
        """`STEPS` linear steps at `lr`, each clipped at `clip_over_g` times the last gradient norm read (0: no
        clip). Returns (first norm, loss drop over the run, its first-order value)."""
        steps.backward(1.0)
        g = steps.linear(0.0, clip=0.0)              # lr 0: reads the gradient norm, moves nothing
        first, predicted = g, 0.0
        start = steps.backward(1.0)
        for _ in range(STEPS):
            clip = clip_over_g * g
            g = steps.linear(lr, clip=clip)          # the norm of the gradient this step applied, before the clip
            predicted += lr / LINEAR_EPS * g * (min(g, clip) if clip else g)
            end = steps.backward(1.0)
        steps.linear(0.0, clip=0.0)                  # drops the gradient the last backward left
        return first, start - end, predicted

    cases = {   # name -> (lr, clip / g). The clipped case has 10 times the lr: the same first-order drop.
        "no clip": (LINEAR_LR, 0.0),
        "clip at g / 10": (10 * LINEAR_LR, 0.1),
        "clip at 10 g": (LINEAR_LR, 10.0),
    }
    for name, (lr, clip_over_g) in cases.items():
        g, drop, predicted = summed_drop(lr, clip_over_g)
        ratio = drop / predicted
        print(f"\n[clip] {name}: {STEPS} steps, first g={g:.2f} grad_clip_norm={clip_over_g} * g lr={lr} "
              f"eps={LINEAR_EPS} loss drop D={drop:.4f} first-order F={predicted:.4f} D / F={ratio:.3f}")
        assert RATIO_BAND[0] <= ratio <= RATIO_BAND[1]


def test_grad_norm_is_the_norm_before_the_clip(steps):
    """Regression guarded: a clipped step returns the clip, the norm after clipping, as its `grad_norm`."""
    steps.backward(1.0)
    g = steps.linear(0.0, clip=0.0)
    steps.backward(1.0)
    clipped = steps.linear(0.0, clip=g / 10)         # lr 0: the same weights, so the same gradient
    print(f"\n[clip] grad_norm {g:.2f} with no clip, {clipped:.2f} with grad_clip_norm {g / 10:.2f}")
    assert clipped == pytest.approx(g, rel=1e-6)


def test_weight_decay_is_the_steps(steps):
    """Regression guarded: the step's `weight_decay` no longer reaches the param groups."""
    before = steps.logprobs()
    steps.backward(0.0)
    # no gradient, no decay: nothing may move
    steps.step(DECAY_LR, beta1=0.0, weight_decay=0.0, clip=0.0, eps=1e-8)
    unmoved = (steps.logprobs() - before).abs().max().item()
    steps.backward(0.0)
    # no gradient: the weights shrink by lr * weight_decay
    steps.step(DECAY_LR, beta1=0.0, weight_decay=1.0, clip=0.0, eps=1e-8)
    moved = (steps.logprobs() - before).abs().mean().item()
    print(f"\n[weight_decay] zero-gradient step: max abs logprob change {unmoved:.2e} without decay, "
          f"mean abs {moved:.2e} with lr * weight_decay = {DECAY_LR}")
    assert unmoved <= SAME
    assert moved > DECAY_FLOOR


def test_beta1_is_the_steps(steps):
    """Regression guarded: the step's `betas` no longer reach the param groups."""
    def after_a_zero_gradient_step(beta1: float) -> torch.Tensor:
        steps.backward(1.0)
        # leaves a first moment behind
        steps.step(LINEAR_LR, beta1=0.9, weight_decay=0.0, clip=0.0, eps=LINEAR_EPS)
        before = steps.logprobs()
        steps.backward(0.0)
        # no gradient: only a kept moment can move the weights
        steps.step(LINEAR_LR, beta1=beta1, weight_decay=0.0, clip=0.0, eps=LINEAR_EPS)
        return (steps.logprobs() - before).abs()

    kept = after_a_zero_gradient_step(0.9).mean().item()
    dropped = after_a_zero_gradient_step(0.0).max().item()
    print(f"\n[beta1] zero-gradient step after a real one: mean abs logprob change {kept:.2e} at beta1 0.9, "
          f"max abs {dropped:.2e} at beta1 0")
    assert kept > SAME
    assert dropped <= SAME
