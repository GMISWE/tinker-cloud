"""What a verl pass reports of its ranks' GPU memory, through the tinker SDK and this tree's own
server (the SmolLM3 student, full weights, one row a GPU): the three `gpu_*_bytes:max` metrics of a
`forward_backward` and of a `forward`.

Needs 4 GPUs, verl, and the student in the HF cache; starts its own Ray instance and its own server.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_pass_memory.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("verl", reason="needs the verl training env")
import torch

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from .pass_memory import ALLOCATED, check_readings_nest, rows
from .verl_smollm3 import NUM_GPUS, STUDENT

GIB = 2**30


def test_a_pass_reports_its_ranks_gpu_memory_through_the_sdk(live_sdk):
    """Four passes on one model, in this order: short rows, long rows, short rows again, then a
    forward of the long rows; then a step. The allocated peak follows the pass in hand: the long
    pass reads at least 1 GiB more than the short one before it (read: 5.6 GiB more), and the short
    pass after it reads less again,
    which a peak that never restarts would not (upstream's own does not). A forward saves
    no activations, so it peaks below the forward_backward of the same rows. A step's metrics are
    not reduced and carry no such key."""
    import tinker

    service = live_sdk("verl", NUM_GPUS=str(NUM_GPUS), TINKERCLOUD_VERL_MAX_SEQ_LEN="18432")
    client = service.create_lora_training_client(base_model=STUDENT, rank=0).result()
    generator = torch.Generator().manual_seed(0)
    short, long = rows(NUM_GPUS, 256, generator), rows(NUM_GPUS, 16384, generator)   # one row a GPU
    short_first = dict(client.forward_backward(short, "cross_entropy").result().metrics)
    long_pass = dict(client.forward_backward(long, "cross_entropy").result().metrics)
    short_after = dict(client.forward_backward(short, "cross_entropy").result().metrics)
    long_forward = dict(client.forward(long, "cross_entropy").result().metrics)
    step = dict(client.optim_step(tinker.AdamParams(learning_rate=0.0)).result().metrics)
    check_readings_nest({"short": short_first, "long": long_pass, "short again": short_after, "forward long": long_forward})
    assert long_pass[ALLOCATED] > short_after[ALLOCATED]
    assert long_pass[ALLOCATED] - short_first[ALLOCATED] >= 1 * GIB
    assert long_forward[ALLOCATED] < long_pass[ALLOCATED]
    assert not [k for k in step if k.startswith("gpu_")]
