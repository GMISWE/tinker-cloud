"""VerlBackend on real GPUs at the size of a long-sequence update: 32 sequences of about 13K tokens with
[N, 16] targets, first with the student alone, then with three forward-only models resident beside it.

The input is a recorded batch, `RECORDED_BATCH`: a torch file `{"tensors": {...}}` of 128 padded rows,
with `prompts` [B, P], `input_ids` and `attention_mask` [B, P + R] (the prompt left-padded, the response
right-padded) and `student_top_k_ids` [B, R, 16]. The models load from `MODEL_ROOT`, a directory of
`<org>/<name>` model directories whose configs carry the RoPE base the batch was recorded at. Needs both,
the four models' `config.json` in the HF cache, 4 GPUs of 80 GB and verl; starts its own Ray instance.
Each test prints seconds and nvidia-smi memory per pass (-s shows it).

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_fit.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tinkercloud.training.backends.verl.backend import VerlBackend  # noqa: E402
from tinkercloud.training.backends.verl.builder import VerlArgumentBuilder  # noqa: E402

from .verl_smollm3 import K, STUDENT, TEACHERS, adam, create_full_weights, datum, gpu_mib, verl_actors  # noqa: E402

RECORDED_BATCH = "/data/runs/mopd/ref/step1.pt"
MODEL_ROOT = "/data/runs/mopd/model_root_theta2e6"
SHARDS, SHARD_ROWS, UPDATE_ROWS, UPDATE = 4, 32, 8, 1   # rows [8, 16) of each rank's 32: the batch's second update
MAX_SEQ_LEN = 18432                                      # 2,048 prompt tokens and 16,384 response tokens
# What a forward, or an update once the optimizer has state, may leave on a GPU above its memory before
# the pass. Read 2026-11-12: 142 MiB at most with the cache released; without, about 24,000 after a forward
# and 33,000 or more after an update.
RESIDUE_LIMIT_MIB = 1024


class Peak:
    """The highest nvidia-smi reading per GPU while the block runs, polled twice a second."""

    def __enter__(self):
        self.mib, self._stop = gpu_mib(), threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(0.5):
            self.mib = [max(a, b) for a, b in zip(self.mib, gpu_mib(), strict=True)]

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()


@pytest.fixture(scope="module")
def update():
    """The recorded update as [N, 16] cross-entropy datums: the student's 16 ids at every response
    position, a small weight on each, zeros on the prompt's positions."""
    recorded = torch.load(RECORDED_BATCH, weights_only=True)["tensors"]
    width = recorded["prompts"].shape[1]
    attention = recorded["attention_mask"].bool()
    data = []
    for row in (SHARD_ROWS * shard + UPDATE_ROWS * UPDATE + i for shard in range(SHARDS) for i in range(UPDATE_ROWS)):
        tokens = recorded["input_ids"][row][attention[row]]
        prompt_len = int(attention[row, :width].sum())
        positions = len(tokens) - 1
        targets = torch.zeros(positions, K, dtype=torch.long)
        targets[prompt_len - 1:] = recorded["student_top_k_ids"][row, :positions - prompt_len + 1]
        weights = torch.zeros(positions, K)
        weights[prompt_len - 1:] = 1e-3
        data.append(datum(tokens[:-1].tolist(), targets, weights))
    return data


@pytest.fixture(scope="module")
def backend():
    with pytest.MonkeyPatch.context() as env:
        env.setenv("TINKERCLOUD_VERL_MODELS_PER_GPU", "4")
        env.setenv("TINKERCLOUD_VERL_MAX_SEQ_LEN", str(MAX_SEQ_LEN))
        env.setenv("TINKERCLOUD_MODEL_ROOT", MODEL_ROOT)
        # a name the root does not hold falls through to another directory or the hub id, silently
        for name in (STUDENT, *TEACHERS):
            resolved = VerlArgumentBuilder.resolve_hf_path(name)
            assert resolved.startswith(MODEL_ROOT + "/"), resolved
        yield VerlBackend()
    assert verl_actors() == []


@pytest.fixture(scope="module")
def fit_student(loop, backend):
    handle = create_full_weights(loop, backend, "fit_student", STUDENT)
    yield handle
    loop.run_until_complete(backend.delete_model(handle))


@pytest.fixture(scope="module")
def fit_teachers(loop, backend, fit_student):
    handles = []
    try:
        for i, teacher in enumerate(TEACHERS):
            handles.append(create_full_weights(loop, backend, f"fit_teacher{i}", teacher))
        yield handles
    finally:
        for handle in handles:
            loop.run_until_complete(backend.delete_model(handle))


def _timed(loop, name, call):
    """Run one pass; print and return (result, memory before, memory after)."""
    before, t0 = gpu_mib(), time.time()
    with Peak() as peak:
        result = loop.run_until_complete(call)
    after = gpu_mib()
    print(f"[fit] {name}: {time.time() - t0:.1f} s; MiB before {before} peak {peak.mib} after {after}", flush=True)
    return result, before, after


def _assert_no_residue(name, before, after):
    left = [a - b for a, b in zip(after, before, strict=True)]
    assert max(left) <= RESIDUE_LIMIT_MIB, f"{name} left {left} MiB on the GPUs"


def _forward_leaving_no_residue(loop, backend, handle, update, name):
    result, before, after = _timed(loop, name, backend.forward(handle, update, "cross_entropy"))
    _assert_no_residue(name, before, after)
    return result


def test_an_update_of_13k_token_sequences_fits_with_one_model(loop, backend, fit_student, update):
    """With 4 sequences per GPU in a forward's micro-batch this forward asked for 30 GiB at once and ran
    out of memory; the engine holds a micro-batch's full logits."""
    print(f"\n[fit] {len(update)} sequences, {sum(len(d.model_input.tokens) for d in update)} input tokens")
    result = _forward_leaving_no_residue(loop, backend, fit_student, update, "student forward")
    shapes = [out["logprobs"]["shape"] for out in result["loss_fn_outputs"]]
    assert shapes == [[len(d.model_input.tokens), K] for d in update]
    _timed(loop, "student forward_backward", backend.forward_backward(fit_student, update, "cross_entropy"))
    step, _, _ = _timed(loop, "student optim_step", backend.apply_optimizer_step(fit_student, adam(1.5e-6)))
    assert step["grad_norm"] > 0


def test_an_update_fits_with_three_forward_only_models_resident(loop, backend, fit_student, fit_teachers, update):
    """Every model is a process of its own on the shared GPUs, and a process keeps the memory a pass
    reserved: read 2026-11-12, a forward-only model held 28 GB after its forward against 4 GB before,
    and the student's backward then ran out of memory; the student held 70 GB after an update, and a
    model built beside it then ran out of memory. Two steps: the second update finds the optimizer's
    state in place, so what it leaves is the pass's alone."""
    for step in (1, 2):
        _forward_leaving_no_residue(loop, backend, fit_student, update, f"step {step} student forward")
        for i, teacher in enumerate(fit_teachers):
            _forward_leaving_no_residue(loop, backend, teacher, update, f"step {step} forward-only model {i}")
        _, before, _ = _timed(loop, f"step {step} student forward_backward",
                              backend.forward_backward(fit_student, update, "cross_entropy"))
        result, _, after = _timed(loop, f"step {step} student optim_step",
                                  backend.apply_optimizer_step(fit_student, adam(1.5e-6)))
        assert result["grad_norm"] > 0
    _assert_no_residue("the second update", before, after)
