"""Three forward-only teachers beside the student on one VerlBackend: four
full-weight SmolLM3 models resident on the same 4 GPUs, one sampler (the
student's, built at its first sample), teachers untouched by the student's step.

Needs 4 GPUs, verl with vLLM, and the four models in the HF cache; starts its
own Ray instance. Skipped elsewhere.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_teachers.py -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import itertools  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import ray  # noqa: E402
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.base import SampleRequestError  # noqa: E402
from tinkercloud.training.backends.verl.backend import VerlBackend  # noqa: E402

from .verl_smollm3 import (  # noqa: E402
    K, STUDENT, TEACHERS, adam, create_full_weights, datum, gpu_mib, live_placement_groups, logprobs_of,
    text_tokens, verl_actors, vllm_actors,
)

POSITIONS = 128
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "seed": 0}


def test_three_teachers_share_the_students_gpus(monkeypatch, loop):
    monkeypatch.setenv("TINKERCLOUD_VERL_MODELS_PER_GPU", "4")
    # the sampler takes its share of GPUs that already hold four models
    backend = VerlBackend(overrides={"rollout_gpu_memory_utilization": 0.3})
    run = loop.run_until_complete
    tokens = text_tokens(POSITIONS + 1)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 30000, (POSITIONS, K), generator=g)
    ids[:, 0] = torch.tensor(tokens[1:])
    nk = datum(tokens[:-1], ids, torch.ones(POSITIONS, K))
    handles = {}
    try:
        for name in (STUDENT, *TEACHERS):
            handles[name] = create_full_weights(loop, backend, "m_" + name.rsplit("-", 1)[1].lower(), name)
        student, teachers = handles[STUDENT], [handles[t] for t in TEACHERS]
        reserved = ray.cluster_resources()["GPU"] - ray.available_resources().get("GPU", 0.0)
        print(f"\n[teachers] after 4 creates: gpu MiB {gpu_mib()} ray reserved GPUs {reserved}")
        assert vllm_actors() == [], "a sampler is built at a model's first sample"
        assert reserved <= 4
        assert len(live_placement_groups()) == 1, "four models, one shared placement group"

        def forward(handle):
            return logprobs_of(run(backend.forward(handle, [nk], "cross_entropy")))

        scores = {name: forward(h) for name, h in handles.items()}
        for name, lp in scores.items():
            assert lp.shape == (POSITIONS, K) and torch.isfinite(lp).all(), name
        for a, b in itertools.combinations(scores, 2):
            gap = (scores[a] - scores[b]).abs().mean().item()
            print(f"[teachers] mean |logprob gap| {a.rsplit('-', 1)[1]} vs {b.rsplit('-', 1)[1]}: {gap:.4f}")
            assert gap > 1e-3, f"{a} and {b} answer alike"

        prompt = tokens[:32]
        first = run(backend.sample(student, "s1", prompt, 1, dict(SAMPLING)))["sequences"][0]["tokens"]
        actors_with_the_sampler = verl_actors()
        with pytest.raises(SampleRequestError, match="one sampling model per server"):
            run(backend.sample(teachers[0], "t", prompt, 1, dict(SAMPLING)))
        assert verl_actors() == actors_with_the_sampler, "the refused sample built nothing"

        step = datum(tokens[:-1], torch.tensor(tokens[1:]), torch.ones(POSITIONS))
        run(backend.forward_backward(student, [step], "cross_entropy"))
        grad_norm = run(backend.apply_optimizer_step(student, adam(1e-5)))["grad_norm"]
        print(f"[teachers] after the student's step: gpu MiB {gpu_mib()} grad_norm {grad_norm:.1f}")
        second = run(backend.sample(student, "s2", prompt, 1, dict(SAMPLING)))["sequences"][0]["tokens"]
        assert len(first) > 0 and len(second) > 0
        for name in TEACHERS:
            drift = (forward(handles[name]) - scores[name]).abs().max().item()
            print(f"[teachers] {name.rsplit('-', 1)[1]} max |logprob drift| across the student's step: {drift:.2e}")
            assert drift <= 1e-6
    finally:
        for handle in handles.values():
            run(backend.delete_model(handle))
    assert verl_actors() == []
    assert live_placement_groups() == []
