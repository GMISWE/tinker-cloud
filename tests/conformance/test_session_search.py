"""Session display names and search (SDK >= 0.38), with one implicit owner and project."""
import uuid

import pytest
import requests
from tinker import types

from .conftest import API_KEY

OWNER = "urn:tinkercloud:owner:local"


@pytest.fixture(scope="module")
def run_tag():
    return uuid.uuid4().hex


def _session(server, run_tag, display_name=None, **metadata):
    import tinker
    sc = tinker.ServiceClient(base_url=server.base_url, api_key=API_KEY, display_name=display_name,
                              user_metadata={"run": run_tag, **metadata})
    return sc, sc.holder.get_session_id()


def _search(service_client, **fields):
    return service_client.create_rest_client().search_sessions(types.SessionSearchRequest(**fields)).result()


def _get(server, path):
    return requests.get(server.base_url + path, headers={"X-API-Key": API_KEY}, timeout=30)


def test_display_name_is_stored_and_echoed(server, run_tag):
    _, named = _session(server, run_tag, display_name="Run Alpha")
    _, plain = _session(server, run_tag)
    assert _get(server, f"/api/v1/sessions/{named}").json()["display_name"] == "Run Alpha"
    assert _get(server, f"/api/v1/sessions/{plain}").json()["display_name"] is None


def test_search_filters_as_documented(service_client, server, run_tag):
    tag = {"run": run_tag}
    _, alpha = _session(server, run_tag, display_name="Nightly Run Alpha", kind="a")
    _, beta = _session(server, run_tag, display_name="nightly beta", kind="b")
    sc_done, done = _session(server, run_tag, display_name="finished one", kind="a")
    sc_done.close("success", detail="t").result()

    names = lambda page: {s.session_id for s in page.sessions}   # noqa: E731
    assert names(_search(service_client, user_metadata=tag, display_name="alpha NIGHTLY")) == {alpha}
    assert names(_search(service_client, user_metadata=tag, display_name="nightly")) == {alpha, beta}
    assert names(_search(service_client, user_metadata=tag, session_id=beta, display_name="nightly alpha")) == {alpha, beta}
    assert names(_search(service_client, user_metadata={**tag, "kind": "a"})) == {alpha, done}
    active = _search(service_client, user_metadata=tag, statuses=[types.ActiveSessionStatusQuery()])
    assert done not in names(active) and {alpha, beta} <= names(active)
    finished = _search(service_client, user_metadata=tag, statuses=[types.ClientFinishedSessionStatusQuery(reasons=["success"])])
    assert names(finished) == {done}
    s = finished.sessions[0]
    assert s.owner == OWNER and s.project_id == "default" and s.display_name == "finished one"
    assert s.status.type == "client_finished" and s.status.reason.type == "success"
    assert names(_search(service_client, user_metadata=tag, owner=types.UserSearchOwner(owner_urns=["urn:x"]))) == set()
    assert {alpha, beta, done} <= names(_search(service_client, user_metadata=tag, owner=types.AccessibleSearchOwner()))


def test_search_pages_newest_first_with_a_cursor(service_client, server, run_tag):
    tag = {"run": run_tag, "page": "yes"}
    made = [_session(server, run_tag, page="yes")[1] for _ in range(7)]
    seen, cursor = [], None
    while True:
        page = _search(service_client, user_metadata=tag, limit=3, cursor=cursor)
        seen += [s.session_id for s in page.sessions]
        created = [s.created_at for s in page.sessions]
        assert created == sorted(created, reverse=True)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert sorted(seen) == sorted(made) and len(seen) == 7
    assert seen[0] == made[-1]   # newest first


def test_search_refuses_an_unknown_project(server):
    r = server.post("/api/v1/sessions/search", {"project_ids": ["p1"]})
    assert r.status_code == 404, r.text
    assert "p1" in r.text
    assert server.post("/api/v1/sessions/search", {"project_ids": ["default"]}).status_code == 200
    assert server.post("/api/v1/sessions/search", {"limit": 0}).status_code == 422
    assert server.post("/api/v1/sessions/search", {"limit": 201}).status_code == 422


def test_search_counts_the_sessions_checkpoints(service_client, server, run_tag):
    sc, sid = _session(server, run_tag, display_name="with checkpoints")
    tc = sc.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.save_state("one").result()
    tc.save_state("two").result()
    page = _search(service_client, session_id=sid)
    assert [s.checkpoints_count for s in page.sessions] == [2]
