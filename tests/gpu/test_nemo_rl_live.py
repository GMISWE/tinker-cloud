"""The nemo_rl backend through the tinker SDK and this tree's own server: pipelined steps reach the model's
scheduler as a client sends them, which the suites that drive `NemoRLBackend` skip.

Needs 2 GPUs, nemo_rl, and the model in the HF cache; starts its own Ray instance and its own server.

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_nemo_rl_live.py -s -v
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("nemo_rl", reason="needs the nemo_rl training env")
import torch

if torch.cuda.device_count() < 2:
    pytest.skip("needs 2 GPUs", allow_module_level=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT, RESPONSE = 8, 8


def random_sdk_rl_datum(types, g):
    """One importance_sampling datum: random tokens, the loss on the response only."""
    tokens = torch.randint(100, 5000, (PROMPT + RESPONSE + 1,), generator=g).tolist()
    n = PROMPT + RESPONSE
    mask = [0.0] * PROMPT + [1.0] * RESPONSE
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=tokens[1:], dtype="int64", shape=[n]),
            "logprobs": types.TensorData(data=[-m for m in mask], dtype="float32", shape=[n]),
            "advantages": types.TensorData(data=mask, dtype="float32", shape=[n]),
        })


def test_pipelined_steps_train_their_own_data_and_a_saved_sampler_samples(live_sdk):
    """Regression guarded: a forward_backward sent before the previous step returned landing in that step.
    All six calls of three steps (2, 3 and 4 datums) are sent before any result is read; each forward_backward
    is the deferred result and each optim_step returns one output per datum of its own step. Second
    regression: a sampler save sent behind those steps never serving, because the save waits on the
    model's scheduler or reaches vLLM before the last step's refit; its sampler must return tokens."""
    import tinker
    from tinker import types

    service = live_sdk("nemo_rl", NUM_GPUS="2")
    client = service.create_lora_training_client(base_model=MODEL, rank=8).result()
    g = torch.Generator().manual_seed(0)
    try:
        sent = []
        for n in (2, 3, 4):
            fb = client.forward_backward([random_sdk_rl_datum(types, g) for _ in range(n)], "importance_sampling")
            sent.append((n, fb, client.optim_step(tinker.AdamParams(learning_rate=1e-5))))
        for n, fb, optim in sent:
            assert fb.result().loss_fn_outputs == []
            assert len(optim.result().loss_fn_outputs) == n
        sampler = client.save_weights_and_get_sampling_client()
        out = sampler.sample(types.ModelInput.from_ints([100, 200, 300]), 1,
                             types.SamplingParams(max_tokens=8)).result()
        print(f"\n[live] sampled {out.sequences[0].tokens}")
        assert out.sequences[0].tokens
    finally:
        server = live_sdk.servers[-1]
        rid = server.post("/api/v1/unload_model", {"model_id": client.model_id}).json()["request_id"]
        assert server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=300).status_code == 200
