"""Passes on a live verl server whose models share the GPUs, through the tinker SDK: the server runs
one model's pass at a time, and a pass that fails in the workers ends its model alone.

Before this, three teachers' forwards submitted together ran out of GPU memory. And after one rank
of a forward ran out of memory while the other ranks went on, the model's next forward and its
delete never returned, and the model's memory stayed on the other GPUs.

The first test's input is the recorded batch of `test_verl_fit.py`, each teacher scoring the rows
of its own domain. Needs the batch, its model root, 4 GPUs of 80 GB and verl; starts its own Ray
instance and its own server. Prints the seconds and the nvidia-smi memory of every stage (-s shows them).

Run on a GPU pod:
  KX_TIMEOUT=1700 scripts/pod_pytest.sh <pod> tests/gpu/test_verl_pass_failure.py -s -v
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("verl", reason="needs the verl training env")
import requests
import torch

if torch.cuda.device_count() < 4:
    pytest.skip("needs 4 GPUs", allow_module_level=True)

from tests.conformance.conftest import API_KEY

from .test_verl_fit import MAX_SEQ_LEN, MODEL_ROOT, RECORDED_BATCH, Peak
from .verl_smollm3 import K, NUM_GPUS, STUDENT, TEACHER_OF_DOMAIN, TEACHERS, gpu_mib, recorded_by_domain

# One rank's failure: the test leaves LEFT_MIB free on the first GPU, and a forward of LONG tokens
# needs about 12,000 MiB of each GPU (1,440 MiB and 1.31 MiB a token, read 2026-11-14 on the student;
# assumed for a forward-only model).
LONG, LEFT_MIB = 8192, 3000
# An answer against no answer: before this the model's next forward and its delete gave none in 180 s.
ANSWER_S, FREED_S = 30, 60
REQUEST_S = 600   # a request that must answer; the longest read is 17 s
# A freed model against a held one: a model at rest holds 4,500 to 5,300 MiB of each GPU.
FREED_SLACK_MIB = 1024


@pytest.fixture(scope="module")
def server(live_sdk):
    """(the SDK's service client, a function that deletes a model over HTTP within a time limit)."""
    service = live_sdk("verl", NUM_GPUS=str(NUM_GPUS), TINKERCLOUD_MODEL_ROOT=MODEL_ROOT,
                       TINKERCLOUD_VERL_MAX_SEQ_LEN=str(MAX_SEQ_LEN), TINKERCLOUD_VERL_MODELS_PER_GPU="4")
    base_url = live_sdk.servers[-1].base_url

    def delete(client, limit_s: float) -> int:
        return requests.post(f"{base_url}/api/v1/delete_model", json={"model_id": client.model_id},
                             headers={"X-API-Key": API_KEY}, timeout=limit_s).status_code

    return service, delete


def _create(service, base_model: str):
    return service.create_lora_training_client(base_model=base_model, rank=0).result()


def _long_datum(generator: torch.Generator):
    from tinker import types

    tokens = torch.randint(1000, 30000, (LONG,), generator=generator)   # ordinary ids of the family's vocabulary
    targets = torch.randint(1000, 30000, (LONG, K), generator=generator)
    return types.Datum(model_input=types.ModelInput.from_ints(tokens.tolist()),
                       loss_fn_inputs={"target_tokens": types.TensorData.from_torch(targets)})


def test_three_teachers_forwards_submitted_together_all_complete(server):
    """A client that scores one step's rollouts with three teachers at once. Their passes overlapped on
    the shared GPUs: one asked for 7.63 GiB with 7.60 GiB free."""
    service, delete = server
    data = recorded_by_domain(RECORDED_BATCH)
    assert sorted(data) == sorted(TEACHER_OF_DOMAIN)
    clients = []
    try:
        clients.append(_create(service, STUDENT))
        teachers = {domain: _create(service, name) for domain, name in TEACHER_OF_DOMAIN.items()}
        clients.extend(teachers.values())
        print(f"\n[together] four models resident: MiB {gpu_mib()}", flush=True)
        started = time.time()
        with Peak() as peak:
            futures = {domain: teacher.forward(data[domain], "cross_entropy") for domain, teacher in teachers.items()}
            for domain, future in futures.items():
                outputs = future.result().loss_fn_outputs
                print(f"[together] {domain}: {len(outputs)} sequences, {time.time() - started:.1f} s after the "
                      f"three were submitted", flush=True)
                assert [o["logprobs"].shape for o in outputs] == [[d.model_input.length, K] for d in data[domain]]
        print(f"[together] highest MiB {peak.mib}; now {gpu_mib()}", flush=True)
    finally:
        for client in clients:
            assert delete(client, REQUEST_S) == 200


def test_one_ranks_failure_ends_its_model_and_no_other(server):
    """After one rank of a forward ran out of memory while the others went on, the model's next forward
    and its delete gave no answer in 180 s, and three GPUs kept 11.6 GB of the model's memory."""
    service, delete = server
    generator = torch.Generator().manual_seed(1)
    data = [_long_datum(generator) for _ in range(NUM_GPUS)]
    clients = {}

    def stage(name: str, call):
        t0 = time.time()
        result = call()
        print(f"[failure] {name}: {time.time() - t0:.1f} s; MiB {gpu_mib()}", flush=True)
        return result

    def forward(client, limit_s: float = REQUEST_S):
        return client.forward(data, "cross_entropy").result(limit_s)

    try:
        torch.zeros(1, device="cuda:0")   # this process's CUDA context, before the first reading that is compared
        clients["b"] = b = _create(service, TEACHERS[1])
        stage("model b's forward, alone on the server", lambda: forward(b))
        b_alone = gpu_mib()
        clients["a"] = a = _create(service, TEACHERS[0])
        stage("model a's forward, memory free", lambda: forward(a))

        free, _ = torch.cuda.mem_get_info(0)
        held = torch.empty(free - LEFT_MIB * 2**20, dtype=torch.uint8, device="cuda:0")
        print(f"[failure] the test holds the first GPU until {LEFT_MIB} MiB are left: MiB {gpu_mib()}", flush=True)
        try:
            with pytest.raises(ValueError, match="(?i)out of memory"):
                forward(a)
            print(f"[failure] model a's forward, one rank out of memory: failed; MiB {gpu_mib()}", flush=True)
        finally:
            del held
            torch.cuda.empty_cache()
        print(f"[failure] the test's memory is freed: MiB {gpu_mib()}", flush=True)

        assert len(stage("model b's forward", lambda: forward(b)).loss_fn_outputs) == NUM_GPUS
        refused_at = time.time()
        with pytest.raises(ValueError, match=a.model_id):
            forward(a, ANSWER_S)
        print(f"[failure] model a's next forward was refused in {time.time() - refused_at:.1f} s", flush=True)

        limit = [mib + FREED_SLACK_MIB for mib in b_alone]
        while any(now > most for now, most in zip(gpu_mib(), limit, strict=True)) and time.time() < refused_at + FREED_S:
            time.sleep(1)
        now = gpu_mib()
        print(f"[failure] {time.time() - refused_at:.1f} s after the refusal: MiB {now}; with model b alone "
              f"{b_alone}", flush=True)
        assert all(mib <= most for mib, most in zip(now, limit, strict=True)), "the stopped model holds GPU memory"

        assert stage("delete model a", lambda: delete(clients.pop("a"), FREED_S)) == 200
        clients["a again"] = again = _create(service, TEACHERS[0])
        assert len(stage("the model created again: forward", lambda: forward(again)).loss_fn_outputs) == NUM_GPUS
    finally:
        for client in clients.values():
            assert delete(client, REQUEST_S) == 200
