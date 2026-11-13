"""The verl backend through the tinker SDK and this tree's own server: what a client's call does end to
end, which the suites that drive `VerlBackend` skip.

Needs 4 GPUs, verl, and the student in the HF cache; starts its own Ray instance and its own server.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_live.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("verl", reason="needs the verl training env")
import torch

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from .verl_smollm3 import K, NUM_GPUS, STUDENT, candidate_ids, text_tokens

POSITIONS, PROMPT = 160, 120


def test_a_custom_loss_on_nk_targets_runs_through_the_sdk(live_sdk):
    """`forward_backward_custom` sends `weights = -gradient` as its second request. The loss here reads
    the best id of the last 40 positions only, as a loss on a response does, so the SDK encodes those
    weights as sparse CSR. The server's own loss over the weights it decoded, `surrogate_loss:sum`,
    must equal the client's loss: for this linear loss the two are the same sum, and weights decoded
    to other cells give another one. A server that refuses the encoding fails the call."""
    import tinker
    from tinker import types

    service = live_sdk("verl", NUM_GPUS=str(NUM_GPUS))
    client = service.create_lora_training_client(base_model=STUDENT, rank=0).result()
    tokens = text_tokens(POSITIONS + 1)
    ids = candidate_ids(tokens, torch.Generator().manual_seed(0))
    datum = types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1]),
                        loss_fn_inputs={"target_tokens": types.TensorData.from_torch(ids)})
    client_loss = []

    def loss_fn(_data, logprobs_list):
        (logprobs,) = logprobs_list
        assert logprobs.shape == (POSITIONS, K)
        loss = -logprobs[PROMPT:, 0].sum()
        client_loss.append(loss.item())
        return loss, {}

    result = client.forward_backward_custom([datum], loss_fn).result()
    weights = torch.zeros(POSITIONS, K)
    weights[PROMPT:, 0] = 1.0   # -gradient of the loss above
    assert types.TensorData.from_torch_sparse(weights).sparse_crow_indices is not None, \
        "these weights no longer go out sparse: the test does not reach the decode"
    step = client.optim_step(tinker.AdamParams(learning_rate=1e-6)).result()
    print(f"\n[live] client loss {client_loss[0]:.4f} metrics {dict(result.metrics)} step {dict(step.metrics)}")
    # read 2026-11-12: 95.45502 on both sides
    assert result.metrics["surrogate_loss:sum"] == pytest.approx(client_loss[0], rel=1e-3)
    assert step.metrics["grad_norm"] > 0
