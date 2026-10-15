"""AutomodelBackend over a stand-in for the ranks: per-request results, the
two fault paths (refused step vs lost group), request validation, delete."""
import asyncio

import pytest
import torch

from tinkercloud.training.backends.automodel.backend import AutomodelBackend, AutomodelHandle, _Group
from tinkercloud.training.backends.automodel.converter import LossFn
from tinkercloud.training.backends.automodel.errors import NonFiniteGradNorm
from tinkercloud.training.backends.automodel.lora import LoraSpec
from tinkercloud.training.backends.automodel.scheduler import GroupScheduler
from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.models.requests import Datum

CONFIG = dict(train_gpus=2, max_loras=1, max_lora_rank=8, max_cpu_loras=1, micro_batch_rows=2)
ADAM = dict(beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
LORA = LoraSpec(rank=8, alpha=16, target_modules=("q_proj",), dropout=0.0)
run = asyncio.run


def ce(tokens):
    return Datum.model_validate({"model_input": {"tokens": tokens}, "loss_fn_inputs": {
        "target_tokens": {"data": tokens}, "weights": {"data": [1.0] * len(tokens)}}})


class Ranks:
    """TrainingGroup stand-in: every real position scores logprob -1 and term
    +1, so a row's loss is its length; optim_step reports the lr as the norm,
    or raises `fault` once."""
    num_ranks = 2

    def __init__(self):
        self.passes = []        # (kind, rows per rank, clip)
        self.fault = None
        self.stopped = False

    async def forward_backward(self, shards, clip):
        return self._run("forward_backward", shards, clip)

    async def forward(self, shards, clip):
        return self._run("forward", shards, clip)

    def _run(self, kind, shards, clip):
        self.passes.append((kind, [s.num_rows for s in shards], clip))
        outs = []
        for s in shards:
            valid = torch.arange(s.input_ids.shape[1])[None, :] < s.lengths[:, None]
            outs.append({
                "row_loss": s.lengths.float(),
                "logprobs": -valid.float(),
                "elementwise": valid.float(),
                "clamped": torch.zeros(s.num_rows),
            })
        return outs

    async def optim_step(self, adam):
        if self.fault is not None:
            fault, self.fault = self.fault, None
            raise fault
        return [{"grad_norm": adam.learning_rate}] * self.num_ranks

    async def stop(self):
        self.stopped = True


def wired():
    """A backend with one group "g" serving model "m" on a Ranks stand-in.
    Call inside a running loop: the scheduler starts its worker task."""
    backend = AutomodelBackend(overrides=CONFIG)
    ranks = Ranks()
    scheduler = GroupScheduler("g", lambda pk, payloads: backend._run_pass("g", pk, payloads))
    backend._groups["g"] = _Group("g", ranks, scheduler, {"m"})
    handle = AutomodelHandle(model_id="m", base_model="b", lora=LORA, group_key="g")
    return backend, ranks, handle


async def until_stopped(ranks, tries=100):
    for _ in range(tries):
        if ranks.stopped:
            return
        await asyncio.sleep(0)
    raise AssertionError("the ranks were not stopped")


def test_pass_results_are_split_per_request():
    async def go():
        backend, ranks, handle = wired()
        a, b = await asyncio.gather(
            backend.forward_backward(handle, [ce([1, 2, 3]), ce([1, 2, 3, 4, 5])], "cross_entropy"),
            backend.forward_backward(handle, [ce([7, 8])], "cross_entropy"),
        )
        assert a["loss"] == 8.0 and b["loss"] == 2.0
        assert a["loss_fn_output_type"] == LossFn.CROSS_ENTROPY.value
        assert a["metrics"] == {"loss:sum": 8.0, "clamped_positions": 0}
        assert [o["logprobs"]["shape"] for o in a["loss_fn_outputs"]] == [[3], [5]]
        assert a["loss_fn_outputs"][1]["logprobs"]["data"] == [-1.0] * 5
        assert a["loss_fn_outputs"][0]["elementwise_loss"]["data"] == [1.0] * 3
        assert b["loss_fn_outputs"][0]["logprobs"] == {"data": [-1.0, -1.0], "shape": [2], "dtype": "float32"}
        for kind, rows, clip in ranks.passes:
            assert kind == "forward_backward" and clip is None
            assert len(set(rows)) == 1, "every rank runs the same number of rows"
        await backend.close()
    run(go())


def test_supported_losses_are_the_wire_strings():
    assert AutomodelBackend.SUPPORTED_LOSS_FNS == {"cross_entropy", "importance_sampling", "ppo"}


def test_nonfinite_step_fails_the_request_and_keeps_the_group():
    async def go():
        backend, ranks, handle = wired()
        ranks.fault = NonFiniteGradNorm("grad norm is inf")
        with pytest.raises(BackendError, match="grad norm is inf"):
            await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert not ranks.stopped and handle.weight_version == 0
        result = await backend.apply_optimizer_step(handle, 0.5, ADAM)
        assert result["grad_norm"] == 0.5 and result["weight_version"] == 1
        await backend.close()
    run(go())


def test_rank_fault_releases_the_group():
    async def go():
        backend, ranks, handle = wired()
        ranks.fault = RuntimeError("CUDA error")
        with pytest.raises(BackendError, match="released"):
            await backend.apply_optimizer_step(handle, 0.5, ADAM)
        await until_stopped(ranks)
        assert "g" not in backend._groups
        with pytest.raises(BackendError, match="CUDA error"):
            await backend.forward_backward(handle, [ce([1, 2])], "cross_entropy")
        await backend.delete_model(handle)   # nothing left to free; not an error
    run(go())


def test_bad_request_fails_before_the_ranks():
    async def go():
        backend, ranks, handle = wired()
        no_weights = Datum.model_validate({"model_input": {"tokens": [1, 2]},
                                           "loss_fn_inputs": {"target_tokens": {"data": [1, 2]}}})
        with pytest.raises(BackendError, match="missing weights"):
            await backend.forward_backward(handle, [no_weights], "cross_entropy")
        with pytest.raises(BackendError, match="not served"):
            await backend.forward_backward(handle, [ce([1, 2])], "dro")
        with pytest.raises(BackendError, match="adam_params"):
            await backend.apply_optimizer_step(handle)
        assert ranks.passes == []
        await backend.close()
    run(go())


def test_delete_closes_the_scheduler_and_stops_the_ranks():
    async def go():
        backend, ranks, handle = wired()
        await backend.delete_model(handle)
        assert ranks.stopped and "g" not in backend._groups
        with pytest.raises(BackendError, match="deleted"):
            await backend.forward(handle, [ce([1, 2])], "cross_entropy")
    run(go())
