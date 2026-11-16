"""A pass's GPU memory in its results, over stand-ins for the ranks: every request of a pass that
several requests share reports the pass's readings, each the largest over the ranks; an optimizer
step reports none."""
import asyncio

from tinkercloud.training.models.requests import AdamParams
from tests.datums import ce_datum
from tests.unit.backends.automodel.test_backend import ADAM, run, wired

# no rank holds every largest reading, so a result that copied one rank's readings fails
BY_RANK = [{"gpu_allocated_peak_bytes": 30, "gpu_reserved_peak_bytes": 40, "gpu_device_used_bytes": 70},
           {"gpu_allocated_peak_bytes": 20, "gpu_reserved_peak_bytes": 50, "gpu_device_used_bytes": 60}]
LARGEST = {"gpu_allocated_peak_bytes:max": 30, "gpu_reserved_peak_bytes:max": 50, "gpu_device_used_bytes:max": 70}


def with_rank_readings(ranks):
    """Give each stand-in rank its own readings."""
    inner = ranks._run

    def per_rank(kind, tenant, shards, clip):
        outs = inner(kind, tenant, shards, clip)
        for out, memory in zip(outs, BY_RANK, strict=True):
            out["memory"] = memory
        return outs

    ranks._run = per_rank


def memory_of(result):
    return {k: v for k, v in result["metrics"].items() if k.startswith("gpu_")}


def test_every_request_of_a_shared_pass_reports_the_largest_reading_over_the_ranks(tmp_path):
    """A result that carried one rank's readings would under-report the fuller rank, and a request
    that joined a pass without the readings would look as if it used no memory."""
    async def go():
        backend, ranks, handle = wired(tmp_path)
        with_rank_readings(ranks)
        # two requests queue behind a held step, so they run as one pass
        ranks.hold_step = asyncio.Event()
        step = asyncio.create_task(backend.apply_optimizer_step(handle, AdamParams(learning_rate=0.5, **ADAM)))
        both = [asyncio.create_task(backend.forward_backward(handle, [ce_datum(tokens)], "cross_entropy"))
                for tokens in ([1, 2, 3], [7, 8])]
        while backend._groups["g"].scheduler.queued() < 2:
            await asyncio.sleep(0)
        ranks.hold_step.set()
        a, b = await asyncio.gather(*both)
        await step
        assert [kind for kind, _, _ in ranks.passes] == ["forward_backward"], "the two requests did not share a pass"
        assert memory_of(a) == LARGEST
        assert memory_of(b) == LARGEST
        forward = await backend.forward(handle, [ce_datum([1, 2])], "cross_entropy")
        assert memory_of(forward) == LARGEST
        await backend.close()
    run(go())


def test_an_optimizer_step_reports_no_memory(tmp_path):
    """Step metrics are not reduced, so a `name:max` key there would be a key no client can read as
    its name says."""
    async def go():
        backend, ranks, handle = wired(tmp_path)
        with_rank_readings(ranks)
        await backend.forward_backward(handle, [ce_datum([1, 2, 3])], "cross_entropy")
        result = await backend.apply_optimizer_step(handle, AdamParams(learning_rate=0.5, **ADAM))
        assert not [k for k in (*result, *result["metrics"]) if k.startswith("gpu_")]
        await backend.close()
    run(go())
