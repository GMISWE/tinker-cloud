"""NemoRLBackend on the megatron worker: the optimizer step's per-datum logprobs follow
the datums (tests/gpu/nemo_rl_order.py). Needs >= 2 GPUs, the NeMo RL image and
Qwen2.5-0.5B-Instruct in the HF cache; the pod's server and Ray head stopped.
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_nemo_rl_logprob_order_megatron.py -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("nemo_rl", reason="needs the NeMo RL image")
import torch  # noqa: E402

if torch.cuda.device_count() < 2:
    pytest.skip("needs >= 2 GPUs", allow_module_level=True)

from tests.gpu.nemo_rl_order import check_step_logprobs_follow_the_datums  # noqa: E402


def test_step_logprobs_follow_the_datums(tmp_path):
    check_step_logprobs_follow_the_datums(megatron=True, tmp_path=tmp_path)
