"""Checkpoint router: SDK search filters and stable cursor paging over real stored records."""
from uuid import UUID
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tinkercloud.training.checkpoints import CheckpointKind, CheckpointSaveOptions, CheckpointStore
from tinkercloud.training.config import AuthConfig
from tinkercloud.training.models.requests import LOCAL_OWNER
from tinkercloud.training.routers import checkpoints
from tinkercloud.training.services.checkpoint_service import CheckpointService
from tinkercloud.training.storage.metadata import MetadataStorage
from tinkercloud.training.utils.auth import APIKeyAuth

PATH = "/api/v1/checkpoints/search"
KEY = {"X-API-Key": "checkpoint-search-test-key"}
W, S, E = CheckpointKind.WEIGHTS, CheckpointKind.SAMPLER_WEIGHTS, CheckpointKind.EXTERNAL_WEIGHTS


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(tmp_path / "checkpoints", MetadataStorage(tmp_path / "metadata"))


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(checkpoints.router)
    app.state.checkpoint_store = store
    app.state.metadata_storage = store.metadata
    app.state.checkpoint_service = CheckpointService(backend=None, store=store)
    app.state.auth = APIKeyAuth(AuthConfig(api_key=KEY["X-API-Key"]))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _save(store, model_id, kind, name, time, metadata=None, base_model="org/model-a", persist=True, status="completed"):
    if base_model is not None:
        store.metadata.save_training_run(model_id, {"base_model": base_model, "lora_config": {"rank": 2}})
    ticket = store.begin_save(model_id, kind, name,
        options=CheckpointSaveOptions(ttl_seconds=None, overwrite=False, user_metadata=metadata),
        persist=persist, optimizer_config={"type": "adamw"})
    if persist:
        (ticket.root / "weights.bin").write_bytes(name.encode())
    if status == "completed":
        store.complete(ticket.ref)
    elif status == "failed":
        store.fail(ticket.ref, "test failure")
    record = store.get(ticket.ref)
    record["created_at"] = time
    store._write(ticket.ref, record)
    return ticket.ref.uri


@pytest.fixture
def seeded(store):
    return [
        _save(store, "run_a", W, "a", "2026-01-01T00:00:01", {"experiment": "nightly", "candidate": "best"}),
        _save(store, "run_a", S, "s", "2026-01-01T00:00:02", {"experiment": "nightly"}),
        _save(store, "run_b", W, "b", "2026-01-01T00:00:03", {"experiment": "other", "candidate": "best"}, "org/model-b"),
        _save(store, "run_b", E, "e", "2026-01-01T00:00:04", {"experiment": "nightly", "candidate": "best"}, "org/model-b"),
    ]


def _search(client, body):
    response = client.post(PATH, json=body, headers=KEY)
    assert response.status_code == 200, response.text
    return response.json()


def test_filters_compose_before_the_page_is_selected(client, seeded):
    response = _search(client, {"user_metadata": {"experiment": "nightly", "candidate": "best"},
        "checkpoint_type": "training", "base_models": ["org/model-a"], "limit": 1})
    assert [c["tinker_path"] for c in response["checkpoints"]] == [seeded[0]]
    assert response["checkpoints"][0]["base_model"] == "org/model-a"
    assert response["next_cursor"] is None


@pytest.mark.parametrize("metadata, indices", [
    ({"experiment": "nightly", "candidate": "best"}, [3, 0]),
    ({"experiment": "Nightly"}, []),
    ({"experiment": "night"}, []),
    ({"missing": ""}, []),
])
def test_every_metadata_pair_must_match_exactly(client, seeded, metadata, indices):
    response = _search(client, {"user_metadata": metadata})
    assert [c["tinker_path"] for c in response["checkpoints"]] == [seeded[i] for i in indices]


@pytest.mark.parametrize("kind, indices", [("training", [2, 0]), ("sampler", [1]), ("external", [3])])
def test_the_sdk_checkpoint_types_are_searchable(client, seeded, kind, indices):
    response = _search(client, {"checkpoint_type": kind})
    assert [c["tinker_path"] for c in response["checkpoints"]] == [seeded[i] for i in indices]


@pytest.mark.parametrize("models, indices", [(None, [3, 2, 1, 0]),
    (["org/model-a", "org/model-b"], [3, 2, 1, 0]), (["org/model-a"], [1, 0]), ([], []), (["missing"], [])])
def test_base_model_matching_is_any_of_and_an_empty_list_matches_none(client, seeded, models, indices):
    response = _search(client, {"base_models": models})
    assert [c["tinker_path"] for c in response["checkpoints"]] == [seeded[i] for i in indices]


@pytest.mark.parametrize("owner", [{"type": "own"}, {"type": "accessible"},
    {"type": "user", "owner_urns": [LOCAL_OWNER]}, {"type": "user", "owner_urns": ["other", LOCAL_OWNER]}])
def test_the_local_owner_is_searchable(client, seeded, owner):
    response = _search(client, {"owner": owner})
    assert [c["tinker_path"] for c in response["checkpoints"]] == list(reversed(seeded))


def test_another_owner_has_no_results(client, seeded):
    assert _search(client, {"owner": {"type": "user", "owner_urns": ["another-owner"]}}) == {
        "checkpoints": [], "next_cursor": None}


@pytest.mark.parametrize("projects", [None, [], ["default"]])
def test_projects_follow_the_existing_session_search_policy(client, seeded, projects):
    response = _search(client, {"project_ids": projects})
    assert [c["tinker_path"] for c in response["checkpoints"]] == list(reversed(seeded))


@pytest.mark.parametrize("projects", [["missing-project"], ["default", "missing-project"]])
def test_an_unknown_project_is_a_named_404(client, seeded, projects):
    response = client.post(PATH, json={"project_ids": projects}, headers=KEY)
    assert response.status_code == 404 and "missing-project" in response.text


@pytest.mark.parametrize("limit", [1, 2, 3, 50, 200])
def test_tied_timestamps_across_runs_and_kinds_page_exactly_once(client, store, limit):
    expected = [_save(store, f"run_{i % 2}", kind, f"c{i}", "2026-01-01T00:00:00")
                for i, kind in enumerate([W, S, E, W, S, E])]
    seen, cursor = [], None
    for _ in range(len(expected) + 1):
        response = _search(client, {"limit": limit, "cursor": cursor})
        seen.extend(c["tinker_path"] for c in response["checkpoints"])
        cursor = response["next_cursor"]
        if cursor is None:
            break
        assert UUID(cursor["last_checkpoint_uuid"]).version == 5
        assert datetime.fromisoformat(cursor["last_created_at"].replace("Z", "+00:00")).utcoffset() == timedelta(0)
    else:
        pytest.fail("cursor did not reach the end")
    assert len(seen) == len(set(seen)) == len(expected)
    assert set(seen) == set(expected)


def test_a_newer_insert_and_an_older_delete_do_not_repeat_the_first_page(client, store):
    paths = [_save(store, "run", W, f"c{i}", f"2026-01-01T00:00:0{i}") for i in range(1, 5)]
    first = _search(client, {"limit": 2})
    assert [c["tinker_path"] for c in first["checkpoints"]] == paths[:1:-1]
    _save(store, "run", W, "new", "2026-01-01T00:00:05")
    from tinkercloud.training.checkpoints import CheckpointRef
    store.delete(CheckpointRef.parse(paths[0]))
    second = _search(client, {"limit": 2, "cursor": first["next_cursor"]})
    assert [c["tinker_path"] for c in second["checkpoints"]] == [paths[1]]
    assert second["next_cursor"] is None


def test_file_sizes_are_walked_only_for_the_filtered_page(client, seeded, store, monkeypatch):
    newest = _save(store, "run_a", W, "newest", "2026-01-01T00:00:05", {"experiment": "nightly"})
    walked = []
    original = store.size_bytes
    monkeypatch.setattr(store, "size_bytes", lambda ref: walked.append(ref.uri) or original(ref))
    response = _search(client, {"checkpoint_type": "training", "user_metadata": {"experiment": "nightly"}, "limit": 1})
    assert [c["tinker_path"] for c in response["checkpoints"]] == [newest]
    assert walked == [newest] and response["checkpoints"][0]["size_bytes"] == len("newest")
    assert response["next_cursor"] is not None


def test_search_and_existing_lists_share_one_checkpoint_shape(client, store):
    _save(store, "run", W, "c", "2026-01-01T00:00:00")
    searched = _search(client, {})["checkpoints"][0]
    all_runs = client.get("/api/v1/checkpoints", headers=KEY).json()["checkpoints"][0]
    one_run = client.get("/api/v1/training_runs/run/checkpoints", headers=KEY).json()["checkpoints"][0]
    assert searched == all_runs == one_run
    schema = client.app.openapi()["components"]["schemas"]
    assert schema["CheckpointSearchResponse"]["properties"]["checkpoints"]["items"]["$ref"].endswith("/Checkpoint")
    assert schema["CheckpointSearchRequest"]["properties"]["limit"] == {
        "type": "integer", "maximum": 200, "minimum": 1, "title": "Limit", "default": 50}


@pytest.mark.parametrize("body", [
    {"limit": 0}, {"limit": 201}, {"limit": None}, {"checkpoint_type": "weights"},
    {"user_metadata": {"number": 1}}, {"unexpected": "ignored"},
    {"owner": {"type": "user"}}, {"owner": {"type": "own", "owner_urns": [LOCAL_OWNER]}},
    {"owner": {"type": "user", "owner_urns": ["user"] * 11}},
    {"cursor": {"last_created_at": "2026-01-01T00:00:00", "last_checkpoint_uuid": "x"}},
    {"cursor": {"last_created_at": "2026-01-01T00:00:00Z"}},
])
def test_invalid_search_input_is_rejected(client, seeded, body):
    response = client.post(PATH, json=body, headers=KEY)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_auth_is_checked_before_project_details_are_exposed(client, seeded, headers):
    response = client.post(PATH, json={"project_ids": ["private-project"]}, headers=headers)
    assert response.status_code == 401 and "private-project" not in response.text


def test_search_uses_the_same_persistent_record_set_as_the_existing_list(client, store):
    paths = [_save(store, "run", W, status, f"2026-01-01T00:00:0{i}", status=status)
             for i, status in enumerate(["pending", "completed", "failed"])]
    _save(store, "run", S, "ephemeral", "2026-01-01T00:00:09", persist=False)
    response = _search(client, {})
    assert [c["tinker_path"] for c in response["checkpoints"]] == list(reversed(paths))
    assert [c["status"] for c in response["checkpoints"]] == ["failed", "completed", "pending"]


def test_legacy_records_without_metadata_or_a_training_run_are_searchable(client, store):
    path = _save(store, "orphan", W, "legacy", "2026-01-01T00:00:00", base_model=None)
    response = _search(client, {})
    assert response["checkpoints"][0]["tinker_path"] == path
    assert response["checkpoints"][0]["base_model"] is None
    assert _search(client, {"user_metadata": {"missing": ""}})["checkpoints"] == []
    assert _search(client, {"base_models": ["org/model-a"]})["checkpoints"] == []


def test_empty_search_has_no_next_cursor(client):
    assert _search(client, {}) == {"checkpoints": [], "next_cursor": None}


def test_dates_are_compared_as_instants_and_reported_in_utc(client, store):
    older = _save(store, "run", W, "older", "2026-01-01T12:00:00+05:00")
    newer = _save(store, "run", W, "newer", "2026-01-01T08:00:00+00:00")
    first = _search(client, {"limit": 1})
    assert first["checkpoints"][0]["tinker_path"] == newer
    second = _search(client, {"limit": 1, "cursor": first["next_cursor"]})
    assert second["checkpoints"][0]["tinker_path"] == older
    assert second["checkpoints"][0]["time"] == "2026-01-01T07:00:00+00:00"


def test_a_cursor_stays_valid_when_the_store_is_reopened(client, store, seeded):
    first = _search(client, {"limit": 2})
    client.app.state.checkpoint_service.store = CheckpointStore(store.base, MetadataStorage(store.metadata.metadata_dir))
    second = _search(client, {"limit": 2, "cursor": first["next_cursor"]})
    assert [c["tinker_path"] for c in second["checkpoints"]] == list(reversed(seeded[:2]))
    assert _search(client, {"limit": 2})["next_cursor"] == first["next_cursor"]


def test_corrupt_training_run_metadata_is_not_treated_as_a_nonmatching_model(client, store, seeded):
    (store.metadata.training_runs_dir / "run_a.json").write_text("{")
    response = client.post(PATH, json={"base_models": ["org/model-a"]}, headers=KEY)
    assert response.status_code == 500
