"""A long-sequence update through the tinker SDK and this tree's own server: how many passes the
server runs for it, and what the client waits. The SDK cuts a request into chunks by estimated bytes,
at the limit the server's client config hands it, and every chunk is a pass of its own.

The input is the recorded batch of `test_verl_fit.py`, with the loss the batch was recorded under: a
clipped surrogate on the 16 recorded ids of each response position, each sequence's token mean, the
mean over the update's 32 sequences. Needs the batch, its model root, 4 GPUs of 80 GB and verl;
starts its own Ray instance and its own server.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_chunk.py -s -v
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("verl", reason="needs the verl training env")
import torch  # noqa: E402

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from .test_verl_fit import MAX_SEQ_LEN, MODEL_ROOT, RECORDED_BATCH, SHARD_ROWS, SHARDS, UPDATE_ROWS  # noqa: E402
from .verl_smollm3 import K, NUM_GPUS, STUDENT  # noqa: E402

CLIP_LOW, CLIP_HIGH, CLIP_C = 0.2, 0.28, 3.0
# The second update's gradient norm before the clip, after the first update's step. Read 2026-11-12
# on this batch through the SDK: 0.7731, 0.7740 and 0.7738 in three runs.
GRAD_NORM, GRAD_NORM_REL = 0.7731, 0.01
# Seconds the client waits for the second update's custom pass. Read 2026-11-12: 18.9 s, 19.0 s and
# 19.0 s in three runs, and 37 s on the same pass while the request and its result were still written
# out as JSON on the way. The limit sits between the two, about 30% above the readings.
PASS_LIMIT_S = 25.0


def _passes(log_path, start: int) -> dict:
    """How many forward and forward_backward passes the server started since byte `start` of its log."""
    with open(log_path) as log:
        log.seek(start)
        lines = log.read().splitlines()
    return {op: sum(f"Starting {op} for " in line for line in lines) for op in ("forward", "forward_backward")}


def test_a_long_update_is_one_pass_each_way_through_the_sdk(live_sdk):
    """The SDK's own chunk limit of 5 MB cuts these 32 sequences into chunks of one or two, each a
    pass of its own and each padded to a row per GPU. Under the server's limit the update is one
    forward and one forward_backward, and still the recorded update: the same gradient norm."""
    import tinker
    from tinker import types

    recorded = torch.load(RECORDED_BATCH, weights_only=True)["tensors"]
    width = recorded["prompts"].shape[1]
    attention = recorded["attention_mask"].bool()
    prompt_len = attention[:, :width].sum(dim=1).tolist()
    response_len = attention[:, width:].sum(dim=1).tolist()

    def rows_of(update: int) -> list:
        return [SHARD_ROWS * shard + UPDATE_ROWS * update + i for shard in range(SHARDS) for i in range(UPDATE_ROWS)]

    def datum(row: int) -> types.Datum:
        tokens = recorded["input_ids"][row][attention[row]]
        targets = torch.zeros(len(tokens) - 1, K, dtype=torch.long)
        targets[prompt_len[row] - 1:] = recorded["student_top_k_ids"][row, :response_len[row]]
        return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1].tolist()),
                           loss_fn_inputs={"target_tokens": types.TensorData.from_torch(targets)})

    def loss_of(rows: list):
        def loss_fn(_data, logprobs_list):
            total = torch.zeros(())
            for row, logprobs in zip(rows, logprobs_list, strict=True):
                r = response_len[row]
                logp = logprobs[prompt_len[row] - 1:].float()
                assert logp.shape == (r, K)
                advantage = recorded["advantages"][row, :r] * recorded["domain_loss_weight"][row]
                ratio = torch.exp(torch.clamp(logp - recorded["student_top_k_log_probs"][row, :r], -20.0, 20.0))
                clipped = torch.maximum(-advantage * ratio, -advantage * torch.clamp(ratio, 1 - CLIP_LOW, 1 + CLIP_HIGH))
                total = total + torch.where(advantage < 0, torch.minimum(-advantage * CLIP_C, clipped), clipped).sum() / r
            return total / len(rows), {}
        return loss_fn

    service = live_sdk("verl", NUM_GPUS=str(NUM_GPUS), TINKERCLOUD_MODEL_ROOT=MODEL_ROOT,
                       TINKERCLOUD_VERL_MAX_SEQ_LEN=str(MAX_SEQ_LEN))
    log_path = live_sdk.servers[-1].log_path
    client = service.create_lora_training_client(base_model=STUDENT, rank=0).result()
    adam = tinker.AdamParams(learning_rate=1.5e-6, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01,
                             grad_clip_norm=1.0)
    first = rows_of(0)
    client.forward_backward_custom([datum(row) for row in first], loss_of(first)).result()
    client.optim_step(adam).result()

    second = rows_of(1)
    data = [datum(row) for row in second]
    start = os.path.getsize(log_path)
    t0 = time.time()
    client.forward_backward_custom(data, loss_of(second)).result()
    seconds = time.time() - t0
    passes = _passes(log_path, start)
    step = client.optim_step(adam).result()
    print(f"\n[chunk] {len(data)} sequences, {sum(response_len[row] for row in second)} response tokens: "
          f"passes {passes}, custom pass {seconds:.1f}s, grad_norm {step.metrics['grad_norm']:.4f}")
    assert passes == {"forward": 1, "forward_backward": 1}
    assert step.metrics["grad_norm"] == pytest.approx(GRAD_NORM, rel=GRAD_NORM_REL)
    assert seconds <= PASS_LIMIT_S
