"""Drain before sleep (specs/019): a sample is admitted and counted under the
same lock a training op uses to take the engine; the op waits for admitted
samples to return before sleeping; a refit resets the prefix cache or fails."""
import asyncio

import pytest

from tinkercloud.training.backends.nemo_rl import backend as nb
from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle


class FakeGeneration:
    def __init__(self, log):
        self.log = log
        self.invalidate_ok = True

    def finish_generation(self):
        self.log.append("sleep")

    def prepare_for_generation(self, *a, **k):
        self.log.append("wake")

    def update_weights_via_ipc_zmq(self):
        return []

    def invalidate_kv_cache(self):
        self.log.append("invalidate")
        return self.invalidate_ok


class FakePolicy:
    def __init__(self, log):
        self.log = log

    def offload_after_refit(self):
        self.log.append("offload")

    def offload_before_refit(self):
        pass

    def get_free_memory_bytes(self):
        return 1 << 30

    def stream_weights_via_ipc_zmq(self, buffer_size_bytes):
        return []


def make_handle(log):
    return NemoRLHandle(model_id="m", backend_type="nemo_rl",
                        policy=FakePolicy(log), policy_generation=FakeGeneration(log))


def test_slot_counts_while_the_sample_runs():
    async def run():
        h = make_handle([])
        assert h.in_flight_samples == 0 and h._samples_drained.is_set()
        async with nb._sampling_slot(h):
            assert h.in_flight_samples == 1 and not h._samples_drained.is_set()
        assert h.in_flight_samples == 0 and h._samples_drained.is_set()
    asyncio.run(run())


def test_training_waits_for_admitted_samples_before_sleeping():
    async def run():
        log = []
        h = make_handle(log)
        release = asyncio.Event()

        async def sample():
            async with nb._sampling_slot(h):
                log.append("sample-in")
                await release.wait()
                log.append("sample-out")

        s = asyncio.create_task(sample())
        await asyncio.sleep(0.01)

        async def train():
            async with h._training_lock:
                await nb._quiesce_generation(h)
                log.append("train")
                async with h._generation_state_lock:
                    h.generation_state = "generation_ready"

        t = asyncio.create_task(train())
        await asyncio.sleep(0.05)
        assert h.generation_state == "training_ready"
        assert "sleep" not in log                      # still waiting on the in-flight sample
        release.set()
        await asyncio.gather(s, t)
        assert log == ["sample-in", "sample-out", "sleep", "train"]
    asyncio.run(run())


def test_sample_arriving_after_the_flip_waits_for_training():
    async def run():
        log = []
        h = make_handle(log)
        done = asyncio.Event()

        async def train():
            async with h._training_lock:
                await nb._quiesce_generation(h)
                log.append("train")
                await done.wait()
                async with h._generation_state_lock:
                    h.generation_state = "generation_ready"

        t = asyncio.create_task(train())
        await asyncio.sleep(0.01)

        async def sample():
            async with nb._sampling_slot(h):
                log.append("sample")

        s = asyncio.create_task(sample())
        await asyncio.sleep(0.05)
        assert "sample" not in log                     # blocked on the training lock
        done.set()
        await asyncio.gather(t, s)
        assert log == ["sleep", "train", "sample"]
        assert h.in_flight_samples == 0
    asyncio.run(run())


def test_cancelled_sample_releases_its_slot():
    async def run():
        h = make_handle([])
        started = asyncio.Event()

        async def sample():
            async with nb._sampling_slot(h):
                started.set()
                await asyncio.sleep(3600)

        s = asyncio.create_task(sample())
        await started.wait()
        s.cancel()
        with pytest.raises(asyncio.CancelledError):
            await s
        assert h.in_flight_samples == 0 and h._samples_drained.is_set()
    asyncio.run(run())


def test_refit_resets_the_prefix_cache_or_fails(monkeypatch):
    import ray
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    log = []
    policy, gen = FakePolicy(log), FakeGeneration(log)
    nb._refit_policy_generation(policy, gen, colocated_inference=True)
    assert log[-1] == "invalidate" and log.count("invalidate") == 1
    gen.invalidate_ok = False
    with pytest.raises(RuntimeError, match="drain violated"):
        nb._refit_policy_generation(policy, gen, colocated_inference=True)
