"""Register, heartbeat, unregister on the engines face; the SDK face answers 503 while no engine is registered;
a registration is shared through the file with another worker's registry."""
import asyncio
import time

import httpx
import orjson

from training.gateway.app import engines_face
from training.gateway.registry import Registry, RegistryConflict

from .support import Harness, run, serve_uds


def _engines(registry: Registry) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=engines_face(registry)), base_url="http://gateway")


ROW = {"name": "fake", "backend": "fake", "gpus": [0, 1], "socket": "/tmp/fake.sock", "pid": 7, "base_model_sampling": False}


def test_register_list_heartbeat_unregister():
    async def main():
        registry = Registry()
        async with _engines(registry) as c:
            assert (await c.get("/engines")).json() == {"engines": []}
            r = await c.post("/engines/register", json=ROW)
            assert r.status_code == 200 and r.json() == {"state": "UP"}
            rows = (await c.get("/engines")).json()["engines"]
            assert len(rows) == 1 and rows[0]["name"] == "fake" and rows[0]["state"] == "UP"
            assert rows[0]["pid"] == 7 and rows[0]["gpus"] == [0, 1] and rows[0]["socket"] == "/tmp/fake.sock"
            assert rows[0]["heartbeat_age_s"] < 1.0
            registry._conn.execute("UPDATE engines SET last_heartbeat = last_heartbeat - 10")   # an engine that went quiet
            registry.refresh()
            assert (await c.get("/engines")).json()["engines"][0]["heartbeat_age_s"] >= 10
            assert (await c.post("/engines/fake/heartbeat")).status_code == 200
            assert (await c.get("/engines")).json()["engines"][0]["heartbeat_age_s"] < 1.0
            assert (await c.post("/engines/ghost/heartbeat")).status_code == 404
            assert (await c.delete("/engines/fake")).status_code == 204
            assert (await c.get("/engines")).json() == {"engines": []}
            assert (await c.delete("/engines/fake")).status_code == 404
            assert (await c.get("/engines/fake")).status_code == 404
    run(main())


def test_second_engine_is_refused_and_a_new_pid_replaces_the_old_row():
    dropped = []
    registry = Registry()
    registry.on_drop(dropped.append)
    registry.register(**ROW)
    try:
        registry.register(**{**ROW, "name": "other"})
    except RegistryConflict as exc:
        assert "fake" in str(exc)
    else:
        raise AssertionError("a second engine registered")
    row = registry.register(**{**ROW, "pid": 8})
    assert row.pid == 8 and [r.pid for r in dropped] == [7]
    assert registry.only() == row
    registry.unregister("fake")
    assert [r.pid for r in dropped] == [7, 8] and registry.only() is None


def test_a_registration_reaches_another_worker_on_its_refresh(tmp_path):
    """Two registries over one file stand in for two gateway workers."""
    a, b = Registry(tmp_path / "engines.db"), Registry(tmp_path / "engines.db")
    dropped = []
    b.on_drop(dropped.append)
    a.register(**ROW)
    assert b.only() is None
    b.refresh()
    assert b.only() is not None and b.only().pid == 7
    a.heartbeat("fake")
    a.register(**{**ROW, "pid": 9})
    b.refresh()
    assert [r.pid for r in dropped] == [7] and b.only().pid == 9
    a.unregister("fake")
    b.refresh()
    assert [r.pid for r in dropped] == [7, 9] and b.only() is None


def test_bad_registration_is_400():
    async def main():
        async with _engines(Registry()) as c:
            assert (await c.post("/engines/register", content=b"nope")).status_code == 400
            assert (await c.post("/engines/register", json={"name": "fake"})).status_code == 400
            assert (await c.post("/engines/register", json={**ROW, "gpus": "0,1"})).status_code == 400
            assert (await c.post("/engines/register", json={**ROW, "gpus": ["0"]})).status_code == 400
            r = await c.post("/engines/register", json={**ROW, "gpus": None})   # not stated
            assert r.status_code == 200 and (await c.get("/engines")).json()["engines"][0]["gpus"] is None
            assert (await c.delete("/engines/fake")).status_code == 204
            r = await c.post("/engines/register", json={**ROW, "name": "other"})
            assert r.status_code == 200
            assert (await c.post("/engines/register", json=ROW)).status_code == 409
    run(main())


def test_sdk_face_is_503_until_an_engine_registers_and_again_after_it_leaves(tmp_path):
    async def main():
        h = Harness(str(tmp_path))
        try:
            r = await h.post("/api/v1/create_model", orjson.dumps({"session_id": "s"}))
            assert r.status_code == 503 and r.json()["error"] == "no engine registered", r.text
            assert r.json()["path"] == "/api/v1/create_model"
            assert (await h.sdk.get("/health")).status_code == 503
            async with serve_uds(h.engine, h.engine_socket):
                assert (await h.register()).status_code == 200
                t0 = time.monotonic()
                r = await h.post("/api/v1/create_model", orjson.dumps({"session_id": "s"}))
                assert r.status_code == 200 and time.monotonic() - t0 < 1.0
                assert (await h.sdk.get("/health")).status_code == 200
                assert (await h.engines.delete("/engines/fake")).status_code == 204
                t0 = time.monotonic()
                r = await h.post("/api/v1/create_model", orjson.dumps({"session_id": "s"}))
                assert r.status_code == 503 and time.monotonic() - t0 < 1.0
                await asyncio.sleep(0)   # the dropped engine's client closes in its own task
        finally:
            await h.close()
    run(main())


def test_unregister_on_a_worker_that_never_cached_the_row_returns_it(tmp_path):
    """The DELETE takes effect in the file whichever worker serves it; the row comes from the file, not the cache."""
    a, b = Registry(tmp_path / "engines.db"), Registry(tmp_path / "engines.db")
    a.register(**ROW)
    row = b.unregister("fake")
    assert row is not None and row.pid == ROW["pid"] and row.gpus == ROW["gpus"]
    assert a.unregister("fake") is None
    a.refresh()
    assert a.rows() == []


def test_a_file_without_the_base_model_sampling_column_gains_it(tmp_path):
    import sqlite3
    path = tmp_path / "engines.db"
    old = sqlite3.connect(str(path))
    old.execute("""CREATE TABLE engines (
        name TEXT PRIMARY KEY, backend TEXT NOT NULL, gpus TEXT, socket TEXT NOT NULL, pid INTEGER NOT NULL,
        registered_at REAL NOT NULL, last_heartbeat REAL NOT NULL)""")
    old.execute("INSERT INTO engines VALUES ('fake', 'fake', NULL, '/tmp/fake.sock', 7, 0.0, 0.0)")
    old.commit()
    old.close()
    registry = Registry(path)
    assert registry.get("fake").base_model_sampling is False   # the old row reads as not declared
    row = registry.register(**{**ROW, "base_model_sampling": True})
    assert row.base_model_sampling is True and Registry(path).get("fake").base_model_sampling is True
