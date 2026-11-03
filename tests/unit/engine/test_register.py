"""The engine's registration client against the gateway's engines face: register, heartbeat on the interval, re-register, unregister."""
import asyncio
import os

import httpx

from training.engine.register import RegisterClient
from training.gateway.app import engines_face
from training.gateway.registry import Registry

from ..gateway.support import run, serve_uds

INTERVAL = 0.05


def test_heartbeats_keep_the_row_fresh_and_close_unregisters(tmp_path):
    async def main():
        registry = Registry()
        gw_sock = str(tmp_path / "gw.sock")
        async with serve_uds(engines_face(registry), gw_sock):
            client = RegisterClient(gw_sock, "fake", "fake", [0], "/tmp/fake.sock", INTERVAL)
            client.start()
            await asyncio.wait_for(client.registered.wait(), 5)
            row = registry.get("fake")
            assert row is not None and row.pid == os.getpid() and row.gpus == [0]
            for _ in range(6):
                await asyncio.sleep(INTERVAL)
                async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=gw_sock), base_url="http://g") as c:
                    rows = (await c.get("/engines")).json()["engines"]
                assert len(rows) == 1 and rows[0]["heartbeat_age_s"] < 2 * INTERVAL, rows
            await client.close()
            assert registry.rows() == []
            await client.close()   # idempotent
    run(main())


def test_a_gateway_that_lost_the_row_gets_it_back_on_the_next_heartbeat(tmp_path):
    async def main():
        registry = Registry()
        gw_sock = str(tmp_path / "gw.sock")
        async with serve_uds(engines_face(registry), gw_sock):
            client = RegisterClient(gw_sock, "fake", "fake", [], "/tmp/fake.sock", INTERVAL)
            client.start()
            await asyncio.wait_for(client.registered.wait(), 5)
            registry.unregister("fake")
            deadline = asyncio.get_running_loop().time() + 2
            while registry.get("fake") is None and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(INTERVAL)
            assert registry.get("fake") is not None
            await client.close()
    run(main())


def test_a_refused_registration_is_not_retried(tmp_path):
    async def main():
        registry = Registry()
        registry.register("other", "fake", None, "/tmp/other.sock", 1)   # the gateway is taken
        gw_sock = str(tmp_path / "gw.sock")
        async with serve_uds(engines_face(registry), gw_sock):
            client = RegisterClient(gw_sock, "fake", "fake", None, "/tmp/fake.sock", INTERVAL)
            client.start()
            await asyncio.wait_for(client.refused.wait(), 5)
            await asyncio.sleep(0.3)
            assert not client.registered.is_set() and registry.get("fake") is None
            assert registry.get("other") is not None
            await client.close()
            assert registry.get("other") is not None   # close() of a refused client deletes nothing
    run(main())


def test_registration_waits_for_a_gateway_that_is_not_up_yet(tmp_path):
    async def main():
        registry = Registry()
        gw_sock = str(tmp_path / "gw.sock")
        client = RegisterClient(gw_sock, "fake", "fake", [], "/tmp/fake.sock", INTERVAL)
        client.start()
        await asyncio.sleep(0.3)
        assert not client.registered.is_set()
        async with serve_uds(engines_face(registry), gw_sock):
            await asyncio.wait_for(client.registered.wait(), 10)
            assert registry.get("fake") is not None
            await client.close()
    run(main())
