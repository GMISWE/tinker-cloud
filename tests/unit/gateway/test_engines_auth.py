"""The engines listener takes the API key and the unix sockets are owner-only: a local user
who can reach the socket still cannot register, unregister or record anything."""
import os
import stat

import httpx

from training.gateway.app import build_app
from training.gateway.registry import Registry
from training.gateway.sessions import SessionStore
from training.gateway.sockets import bind_unix_socket

from .support import API_KEY, config, run

ROW = {"name": "fake", "backend": "fake", "gpus": [], "socket": "/tmp/fake.sock", "pid": 1}


def test_the_engines_listener_refuses_a_request_without_the_api_key():
    async def main():
        app = build_app(config(), Registry(), SessionStore(None))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.state.engines), base_url="http://g") as c:
            for headers in ({}, {"X-API-Key": "wrong"}):
                assert (await c.post("/engines/register", json=ROW, headers=headers)).status_code == 401
                assert (await c.get("/engines", headers=headers)).status_code == 401
                assert (await c.post("/engines/fake/heartbeat", headers=headers)).status_code == 401
                assert (await c.delete("/engines/fake", headers=headers)).status_code == 401
                assert (await c.delete("/engines/fake/models/m", headers=headers)).status_code == 401
            ok = {"X-API-Key": API_KEY}
            r = await c.post("/engines/register", json=ROW, headers=ok)
            assert r.status_code == 200, r.text
            assert [e["name"] for e in (await c.get("/engines", headers=ok)).json()["engines"]] == ["fake"]
    run(main())


def test_unix_sockets_are_owner_only(tmp_path):
    path = str(tmp_path / "s.sock")
    sock = bind_unix_socket(path)
    try:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        sock.close()
