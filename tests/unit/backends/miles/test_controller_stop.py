"""Controller teardown distinguishes failed Ray calls from local bugs, on both delete paths."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ray.exceptions import RayActorError, RayTaskError

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.miles import backend as miles_backend
from tinkercloud.training.backends.miles.backend import MilesBackend, MilesHandle, MilesPool


@pytest.fixture
def deletion(monkeypatch):
    calls = []
    stop = AsyncMock()
    controller = SimpleNamespace(stop=SimpleNamespace(remote=stop))
    actors = [object(), object()]
    rollout = object()
    placement = object()
    group = SimpleNamespace(_actor_handles=actors)
    pgs = {"train": (placement, [], []), "rollout": (placement, [], [])}
    backend = MilesBackend()
    backend._pool_stop = AsyncMock()
    handle = MilesHandle(model_id="m", backend_type="miles", adapter_slot=0,
                         adapter_name="m", controller=controller, train_group=group,
                         rollout_manager=rollout, placement_group=pgs)
    pool = MilesPool(train_group=group, rollout_manager=rollout, placement_group=pgs,
                     controller=controller, args=None, hf_path="fake", base_model="fake",
                     tenants={"m": 0})
    monkeypatch.setattr(miles_backend.ray, "kill", lambda actor, **kwargs: calls.append(("kill", actor, kwargs)))
    monkeypatch.setattr(miles_backend.ray.util, "remove_placement_group",
                        lambda pg: calls.append(("placement", pg)))
    return backend, handle, pool, stop, calls


def _delete(deletion, last_tenant):
    backend, handle, pool, _, _ = deletion
    if last_tenant:
        backend._pool = pool
    asyncio.run(backend.delete_model(handle))


@pytest.mark.parametrize("last_tenant", [True, False])
@pytest.mark.parametrize("remote_error", [None, RayActorError(),
    RayTaskError("stop", "remote traceback", RuntimeError("remote stop failed")).as_instanceof_cause()])
def test_a_successful_or_failed_remote_stop_still_kills_the_controller(deletion, last_tenant, remote_error, caplog):
    backend, handle, pool, stop, calls = deletion
    stop.side_effect = remote_error
    _delete(deletion, last_tenant)

    stop.assert_awaited_once_with()
    assert calls == [
        ("kill", pool.controller, {"no_restart": True}),
        *(("kill", actor, {"no_restart": True}) for actor in pool.train_group._actor_handles),
        ("kill", pool.rollout_manager, {"no_restart": True}),
        ("placement", next(iter(pool.placement_group.values()))[0]),
    ]
    assert ("controller stop failed" in caplog.text) is (remote_error is not None)
    assert not handle.lock.locked() and not backend._pool_admin.locked()
    assert backend._pool is None
    if last_tenant:
        backend._pool_stop.assert_awaited_once_with(pool)
    else:
        backend._pool_stop.assert_not_awaited()


@pytest.mark.parametrize("last_tenant", [True, False])
@pytest.mark.parametrize("error_type", [ValueError, TypeError, AssertionError])
def test_an_unexpected_local_stop_error_is_not_reported_as_a_success(deletion, last_tenant, error_type):
    backend, handle, _, stop, calls = deletion
    error = error_type("local stop bug")
    stop.side_effect = error
    with pytest.raises(BackendError, match="local stop bug") as raised:
        _delete(deletion, last_tenant)

    assert raised.value.original_error is error
    assert raised.value.__cause__ is error
    assert raised.value.operation == "delete_model" and raised.value.backend == "miles"
    stop.assert_awaited_once_with()
    assert calls == []
    assert not handle.lock.locked() and not backend._pool_admin.locked()
    assert backend._pool is None


@pytest.mark.parametrize("last_tenant", [True, False])
def test_a_failed_kill_propagates_and_releases_the_delete_locks(deletion, monkeypatch, last_tenant):
    backend, handle, _, stop, _ = deletion
    error = RayActorError(error_msg="controller kill failed")

    def fail_kill(actor, **kwargs):
        raise error

    monkeypatch.setattr(miles_backend.ray, "kill", fail_kill)
    with pytest.raises(BackendError, match="controller kill failed") as raised:
        _delete(deletion, last_tenant)
    assert raised.value.original_error is error
    stop.assert_awaited_once_with()
    assert not handle.lock.locked() and not backend._pool_admin.locked()
    assert backend._pool is None


@pytest.mark.parametrize("last_tenant", [True, False])
def test_cancelling_a_controller_stop_propagates_and_releases_the_delete_locks(deletion, last_tenant):
    backend, handle, _, stop, calls = deletion
    stop.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _delete(deletion, last_tenant)
    assert calls == []
    assert not handle.lock.locked() and not backend._pool_admin.locked()
    assert backend._pool is None
