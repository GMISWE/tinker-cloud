"""Four full-weight models on one live verl server, through the tinker SDK: the student holds the
server's sampler, and every model's pass at the size of a long-sequence update runs with the
sampler asleep. Before this the first teacher's forward ran out of GPU memory beside the awake
sampler.

The input is the recorded batch of `test_verl_fit.py`, its second update: 32 sequences with
`[N, 16]` targets. Needs the batch, its model root, 4 GPUs of 80 GB and verl; starts its own Ray
instance and its own server. Prints the seconds and the nvidia-smi memory of every stage, and the
server's request line of every pass (-s shows them).

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_sampler_sleep.py -s -v
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tinkercloud.training.core.request_log import RequestLine  # noqa: E402

from .test_verl_fit import MAX_SEQ_LEN, MODEL_ROOT, RECORDED_BATCH, SHARD_ROWS, SHARDS, UPDATE, UPDATE_ROWS  # noqa: E402
from .verl_smollm3 import K, NUM_GPUS, STUDENT, TEACHERS, gpu_mib, recorded_update  # noqa: E402

SAMPLE_TOKENS = 88




def test_every_models_pass_runs_with_the_students_sampler_asleep(live_sdk):
    import tinker
    from tinker import types

    data, prompt, prompt_lens = recorded_update(RECORDED_BATCH, SHARDS, SHARD_ROWS, UPDATE_ROWS, UPDATE)
    service = live_sdk("verl", NUM_GPUS=str(NUM_GPUS), TINKERCLOUD_MODEL_ROOT=MODEL_ROOT,
                       TINKERCLOUD_VERL_MAX_SEQ_LEN=str(MAX_SEQ_LEN), TINKERCLOUD_VERL_MODELS_PER_GPU="4")
    log_path = live_sdk.servers[-1].log_path
    student = service.create_lora_training_client(base_model=STUDENT, rank=0).result()
    teachers = [service.create_lora_training_client(base_model=name, rank=0).result() for name in TEACHERS]
    print(f"\n[sleep] four models resident: MiB {gpu_mib()}", flush=True)

    def stage(name: str, call):
        """Run `call`, then print its seconds and the memory after it."""
        t0 = time.time()
        result = call()
        print(f"[sleep] {name}: {time.time() - t0:.1f} s; MiB {gpu_mib()}", flush=True)
        return result

    def sample():
        params = types.SamplingParams(max_tokens=SAMPLE_TOKENS, temperature=1.0)
        return sampler.sample(types.ModelInput.from_ints(prompt), num_samples=1, sampling_params=params).result()

    sampler = student.save_weights_and_get_sampling_client(name="sleep")
    first = stage("the student's first sample (builds the sampler)", sample)
    assert 0 < len(first.sequences[0].tokens) <= SAMPLE_TOKENS

    for i, teacher in enumerate(teachers):
        scored = stage(f"teacher {i} forward", lambda: teacher.forward(data, "cross_entropy").result())
        assert len(scored.loss_fn_outputs) == len(data)
        for datum, output in zip(data, scored.loss_fn_outputs, strict=True):
            assert output["logprobs"].shape == [datum.model_input.length, K]

    def loss_fn(_data, logprobs_list):
        """The mean logprob of the 16 ids over each sequence's response positions."""
        total = torch.zeros(())
        for prompt_len, logprobs in zip(prompt_lens, logprobs_list, strict=True):
            total = total - logprobs[prompt_len - 1:].float().mean()
        return total / len(logprobs_list), {}

    stage("the student's custom pass", lambda: student.forward_backward_custom(data, loss_fn).result())
    adam = tinker.AdamParams(learning_rate=1.5e-6, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
                             grad_clip_norm=1.0)
    step = stage("the student's optim_step", lambda: student.optim_step(adam).result())
    assert step.metrics["grad_norm"] > 0

    second = stage("the student's second sample (wakes the sampler)", sample)
    assert len(second.sequences[0].tokens) > 0

    # read after the second sample: a line is written once its request's result is stored
    with open(log_path) as log:
        lines = [line for line in map(RequestLine.parse, log.read().splitlines()) if line is not None]
    for line in lines:
        print(f"[sleep] {line.format()}", flush=True)
    # three teacher forwards, the custom pass's forward and forward_backward, the optim_step
    assert [line.operation for line in lines] == ["forward"] * 4 + ["forward_backward", "optim_step"]
    assert all(line.outcome == "completed" and line.size.body_bytes > 0 for line in lines)
