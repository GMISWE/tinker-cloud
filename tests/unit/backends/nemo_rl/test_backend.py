"""NemoRLBackend without a cluster: R9 buffering (forward_backward defers, optim_step trains the buffer) and the
drain-before-sleep generation lifecycle (specs/019): a sample is admitted and counted under the same lock a training
op uses to take the engine; the op waits for admitted samples before sleeping; a refit resets the prefix cache or fails."""
import asyncio
import importlib.util

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.nemo_rl import backend as nb
from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle
from tests.datums import make_synthetic_batch

HAS_NEMO_RL = importlib.util.find_spec("nemo_rl") is not None


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

    def update_weights_from_collective(self):
        self.log.append("nccl-recv")
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

    def broadcast_weights_for_collective(self):
        self.log.append("nccl-send")
        return []


def make_handle(log, **kw):
    return NemoRLHandle(model_id="m", backend_type="nemo_rl",
                        policy=FakePolicy(log), policy_generation=FakeGeneration(log), **kw)


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


def test_split_training_leaves_the_engine_with_its_samples():
    async def run():
        log = []
        h = make_handle(log, colocated_inference=False)
        release = asyncio.Event()

        async def sample():
            async with nb._sampling_slot(h):
                log.append("sample-in")
                await release.wait()
                log.append("sample-out")

        s = asyncio.create_task(sample())
        await asyncio.sleep(0.01)
        async with h._training_lock:
            await nb._take_engine_for_training(h)      # returns at once
            log.append("train")
        assert h.generation_state == "generation_ready" and h.in_flight_samples == 1
        release.set()
        await s
        assert log == ["sample-in", "train", "sample-out"]
        # delete still drains (no sample may be mid-flight on a torn-down engine)
        async with h._training_lock:
            await nb._drain_samples(h)
        assert h.generation_state == "training_ready" and "sleep" not in log
    asyncio.run(run())


def test_version_bumps_after_the_swap_even_when_the_swap_fails(monkeypatch):
    seen = []

    async def fake_refit(handle):
        seen.append(("refit", h.weight_version, h.generation_synced_version))
        if fail:
            raise RuntimeError("nccl down")

    monkeypatch.setattr(nb, "_refit", fake_refit)
    h = make_handle([], colocated_inference=False, weight_version=3, generation_synced_version=3)
    fail = False
    asyncio.run(nb._advance_weights(h))
    assert seen == [("refit", 3, 3)]                   # swap runs before the bump
    assert (h.weight_version, h.generation_synced_version) == (4, 4)
    fail = True
    with pytest.raises(RuntimeError, match="nccl down"):
        asyncio.run(nb._advance_weights(h))
    assert (h.weight_version, h.generation_synced_version) == (5, 4)   # trainer moved, engine did not


def test_deferred_refit_under_staleness_k_bumps_only_the_trainer(monkeypatch):
    async def no_refit(handle):
        pytest.fail("refit must be deferred")

    monkeypatch.setattr(nb, "_refit", no_refit)
    log = []
    h = make_handle(log, colocated_inference=False, staleness_k=1, weight_version=2, generation_synced_version=2)
    asyncio.run(nb._advance_weights(h))
    assert (h.weight_version, h.generation_synced_version) == (3, 2) and "wake" not in log


def test_deferred_refit_colocated_offloads_the_policy_for_the_next_step(monkeypatch):
    """Colocated, the stale wake offloads the trainer like a refit would, so the
    next optim_step must run prepare_for_training again."""
    async def no_refit(handle):
        pytest.fail("refit must be deferred")

    monkeypatch.setattr(nb, "_refit", no_refit)
    log = []
    h = make_handle(log, colocated_inference=True, staleness_k=1, weight_version=2,
                    generation_synced_version=2, training_resident=True)
    asyncio.run(nb._advance_weights(h))
    assert (h.weight_version, h.generation_synced_version) == (3, 2)
    assert h.training_resident is False


def test_create_time_split_sync_tolerates_a_busy_prefix_cache(monkeypatch):
    """The Ray-driven NCCL sync is used once, at create_model, before any HTTP
    traffic exists; later refits go over the workers' app (test_nemorl_http_sampling)."""
    import ray
    monkeypatch.setattr(ray, "get", lambda refs: refs)
    log = []
    policy, gen = FakePolicy(log), FakeGeneration(log)
    gen.invalidate_ok = False
    nb._refit_policy_generation(policy, gen, colocated_inference=False)   # no raise: the salt covers it
    assert log == ["nccl-send", "nccl-recv", "invalidate"]
    with pytest.raises(RuntimeError, match="drain violated"):
        nb._refit_policy_generation(policy, gen, colocated_inference=True)


@pytest.mark.skipif(not HAS_NEMO_RL, reason="NeMo RL not installed")
class TestNemoRLBackendBuffering:
    """Test the R9 buffering strategy without GPU."""

    @pytest.fixture
    def backend(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLBackend
        return NemoRLBackend()

    @pytest.fixture
    def handle(self):
        from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle
        return NemoRLHandle(model_id="test", backend_type="nemo_rl", config={})

    def test_forward_backward_returns_deferred(self, backend, handle):
        """forward_backward() should return deferred=True."""
        data = make_synthetic_batch(batch_size=2)
        result = asyncio.run(
            backend.forward_backward(handle, data, "importance_sampling")
        )
        assert result["deferred"] is True
        assert result["metrics"] == {}
        assert len(handle.data_buffer) == 1

    def test_forward_backward_accumulates_buffer(self, backend, handle):
        """Multiple forward_backward() calls accumulate in buffer."""
        for _ in range(3):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )
        assert len(handle.data_buffer) == 3

    def test_forward_backward_empty_data_is_noop(self, backend, handle):
        """Empty data (GRPO all-zero advantages) is a deferred no-op, not an error."""
        result = asyncio.run(backend.forward_backward(handle, [], "importance_sampling"))
        assert result == {
            "loss_fn_output_type": "importance_sampling",
            "metrics": {}, "deferred": True, "loss_fn_outputs": [],
        }
        assert len(handle.data_buffer) == 0

    def test_apply_optimizer_step_empty_buffer_is_noop(self, backend, handle):
        """An empty buffer steps nothing and reports success with grad_norm 0."""
        result = asyncio.run(backend.apply_optimizer_step(handle))
        assert result["success"] is True
        assert result["grad_norm"] == 0.0
        assert result["loss_fn_outputs"] == []

    def test_buffer_overflow_raises(self, backend, handle):
        """forward_backward() should raise when buffer exceeds max_buffer_size (CHK006)."""
        handle.max_buffer_size = 3
        for _ in range(3):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )
        assert len(handle.data_buffer) == 3
        with pytest.raises(BackendError, match="Buffer full"):
            asyncio.run(
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
            )

    def test_delete_model_clears_buffer(self, backend, handle):
        """delete_model() should warn and clear pending buffer (CHK007)."""
        asyncio.run(
            backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
        )
        assert len(handle.data_buffer) == 1
        # delete_model will fail on policy shutdown (no real policy), but
        # the buffer should be cleared before that error
        try:
            asyncio.run(backend.delete_model(handle))
        except BackendError:
            pass  # Expected — no real policy
        assert len(handle.data_buffer) == 0

    def test_deferred_result_format(self, backend, handle):
        """forward_backward() deferred result must have exact contract fields (CHK010)."""
        data = make_synthetic_batch(batch_size=2)
        result = asyncio.run(
            backend.forward_backward(handle, data, "importance_sampling")
        )
        assert result == {
            "loss_fn_output_type": "importance_sampling",
            "metrics": {}, "deferred": True, "loss_fn_outputs": [],
        }

    def test_concurrent_forward_backward_thread_safe(self, backend, handle):
        """Concurrent forward_backward() calls should be serialized by lock (CHK018)."""
        async def run_concurrent():
            tasks = [
                backend.forward_backward(handle, make_synthetic_batch(2), "importance_sampling")
                for _ in range(5)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_concurrent())
        assert len(handle.data_buffer) == 5
        assert all(r["deferred"] is True for r in results)


def test_learning_rate_rpc_failure_is_a_backend_error():
    from tinkercloud.training.backends.nemo_rl.backend import _set_learning_rate

    class _WorkerGroup:
        def run_all_workers_single_data(self, *a, **k):
            raise RuntimeError("actor died")

    class _Policy:
        worker_group = _WorkerGroup()

    with pytest.raises(BackendError, match="could not set learning rate 0.5"):
        _set_learning_rate(_Policy(), 0.5)
