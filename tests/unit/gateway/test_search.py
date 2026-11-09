"""gateway.search: the documented filters and the cursor, without a server."""
import pytest

from tinkercloud.training.gateway.search import search
from tinkercloud.training.gateway.sessions import SessionStore
from tinkercloud.training.models.requests import SessionSearchRequest, UnknownProject


def _store_with(n):
    store = SessionStore(None)
    ids = [store.create_session([], {"i": str(i)}, "t", f"name {i}") for i in range(n)]
    return store, ids


def test_ties_on_created_at_still_page_exactly_once():
    store, ids = _store_with(5)
    store._conn.execute("UPDATE sessions SET created_at = '2026-11-06T00:00:00+00:00'")   # one timestamp for all
    seen, cursor = [], None
    while True:
        page, cursor_doc = search(store.sessions_newest_first(), SessionSearchRequest(limit=2, cursor=cursor))
        seen += [s.session_id for s in page]
        if cursor_doc is None:
            break
        cursor = cursor_doc
    assert sorted(seen) == sorted(ids) and len(seen) == 5


def test_an_unknown_project_is_refused_by_name():
    store, _ = _store_with(1)
    with pytest.raises(UnknownProject, match="p1"):
        search(store.sessions_newest_first(), SessionSearchRequest(project_ids=["default", "p1"]))


def test_user_owner_needs_urns():
    with pytest.raises(ValueError):
        SessionSearchRequest.model_validate({"owner": {"type": "user"}})


def test_metadata_matches_in_the_encoding_the_sdk_reads():
    store = SessionStore(None)
    sid = store.create_session([], {"flag": True, "n": 3}, "t", None)
    page, _ = search(store.sessions_newest_first(), SessionSearchRequest(user_metadata={"flag": "true", "n": "3"}))
    assert [s.session_id for s in page] == [sid]


def test_a_naive_cursor_time_is_refused():
    with pytest.raises(ValueError):
        SessionSearchRequest.model_validate(
            {"cursor": {"last_created_at": "2026-11-06T00:00:00", "last_session_id": "x"}})
