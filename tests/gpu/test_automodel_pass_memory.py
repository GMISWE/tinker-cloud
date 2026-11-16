"""What a pass reports of its ranks' GPU memory, through the tinker SDK and this tree's own server
(Qwen2.5-0.5B-Instruct, LoRA rank 8, 2 ranks): the three `gpu_*_bytes:max` metrics of a
`forward_backward` and of a `forward`.

Needs >= 3 GPUs (2 ranks + vLLM), the Automodel training env with /opt/vllm-env, and the model in
the HF cache; starts its own Ray instance and its own server.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_automodel_pass_memory.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("nemo_automodel", reason="needs the Automodel training env")
import torch

if torch.cuda.device_count() < 3:
    pytest.skip("needs >= 3 GPUs", allow_module_level=True)

from .pass_memory import ALLOCATED, check_readings_nest, rows

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER = dict(AUTOMODEL_TRAIN_GPUS="2", AUTOMODEL_MAX_LORAS="2", AUTOMODEL_MAX_LORA_RANK="8", AUTOMODEL_MAX_CPU_LORAS="4",
              AUTOMODEL_MICRO_BATCH_ROWS="2", AUTOMODEL_ACTIVATION_CHECKPOINTING="0")
ROWS = 4   # one micro-batch of 2 rows on each of the 2 ranks


def test_a_pass_reports_its_ranks_gpu_memory_through_the_sdk(live_sdk):
    """Four passes on one model, in this order: short rows, long rows, short rows again, then a
    forward of the long rows. The allocated peak follows the pass in hand: the long pass reads more
    than the short one before it, and the short pass after it reads less again, which a peak that
    was never reset would not. A forward saves no activations, so it peaks below the
    forward_backward of the same rows."""
    import tinker

    service = live_sdk("automodel", NUM_GPUS=str(torch.cuda.device_count()), **SERVER)
    client = service.create_lora_training_client(base_model=MODEL, rank=8).result()
    generator = torch.Generator().manual_seed(0)
    short, long = rows(ROWS, 256, generator), rows(ROWS, 2048, generator)
    short_before = dict(client.forward_backward(short, "cross_entropy").result().metrics)
    long_pass = dict(client.forward_backward(long, "cross_entropy").result().metrics)
    short_after = dict(client.forward_backward(short, "cross_entropy").result().metrics)
    long_forward = dict(client.forward(long, "cross_entropy").result().metrics)
    client.optim_step(tinker.AdamParams(learning_rate=0.0)).result()
    check_readings_nest({"short": short_before, "long": long_pass, "short again": short_after, "forward long": long_forward})
    assert long_pass[ALLOCATED] > short_before[ALLOCATED]
    assert short_after[ALLOCATED] < long_pass[ALLOCATED]
    assert long_forward[ALLOCATED] < long_pass[ALLOCATED]
