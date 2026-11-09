"""Training runs through the SDK's RestClient and the `tinker` CLI: list_training_runs pages
every run the server recorded, and a run reports its newest loadable checkpoints."""
import json
import os
import subprocess
import sys

import pytest
import requests
import tinker

from .conftest import API_KEY


@pytest.fixture
def rest(server, service_client):
    return tinker.RestClient(base_url=server.base_url, api_key=API_KEY)


def _created(service_client, rank):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=rank)
    tc.get_info()   # the create's future has resolved: the run is recorded
    return tc


def _all_runs(rest, page_size):
    runs, offset = [], 0
    while True:
        resp = rest.list_training_runs(limit=page_size, offset=offset).result()
        assert (resp.cursor.offset, resp.cursor.limit) == (offset, page_size)
        runs.extend(resp.training_runs)
        offset += page_size
        if offset >= resp.cursor.total_count:
            return runs, resp.cursor.total_count


def test_list_training_runs_pages_newest_first(service_client, rest):
    a = _created(service_client, rank=2)
    b = _created(service_client, rank=4)

    first = rest.list_training_runs(limit=2).result()
    assert [r.training_run_id for r in first.training_runs] == [b.model_id, a.model_id]
    assert first.cursor.total_count >= 2
    [rb, ra] = first.training_runs
    assert (ra.base_model, ra.is_lora, ra.lora_rank, rb.lora_rank) == ("fake/tiny", True, 2, 4)
    assert ra.last_checkpoint is None and ra.last_sampler_checkpoint is None

    runs, total = _all_runs(rest, page_size=1)
    ids = [r.training_run_id for r in runs]
    assert len(ids) == total == len(set(ids))   # offset pages neither skip nor repeat a run
    assert ids == [r.training_run_id for r in _all_runs(rest, page_size=1000)[0]]


def test_an_unloaded_run_is_still_listed_with_its_checkpoint(service_client, server, rest):
    tc = _created(service_client, rank=2)
    p = tc.save_state("keep").result().path
    rid = server.post("/api/v1/unload_model", {"model_id": tc.model_id}).json()["request_id"]
    assert server.post("/api/v1/retrieve_future", {"request_id": rid}, timeout=60).status_code == 200
    [run] = [r for r in _all_runs(rest, page_size=100)[0] if r.training_run_id == tc.model_id]
    assert run.last_checkpoint.tinker_path == p   # what a resume after the unload loads


def test_run_reports_its_newest_loadable_checkpoints(service_client, rest):
    tc = _created(service_client, rank=2)
    tc.save_state("w1").result()
    w2 = tc.save_state("w2").result().path
    s1 = tc.save_weights_for_sampler("s1").result().path
    tc.save_weights_external("hf", ttl_seconds=3600).result()   # not a kind a TrainingRun names

    run = rest.get_training_run(tc.model_id).result()
    assert run.last_checkpoint.tinker_path == w2 and run.last_checkpoint.checkpoint_type == "training"
    assert run.last_sampler_checkpoint.tinker_path == s1
    [listed] = [r for r in _all_runs(rest, page_size=100)[0] if r.training_run_id == tc.model_id]
    assert listed.last_checkpoint.tinker_path == w2

    rest.delete_checkpoint_from_tinker_path(w2).result()
    run = rest.get_training_run(tc.model_id).result()
    assert run.last_checkpoint.checkpoint_id == "weights/w1"   # never a path that no longer resolves


def test_bad_paging_is_refused(server):
    url = f"{server.base_url}/api/v1/training_runs"
    for q in ({"limit": 0}, {"limit": 1001}, {"offset": -1}, {"access_scope": "everyone"}):
        r = requests.get(url, params=q, headers={"X-API-Key": API_KEY}, timeout=30)
        assert r.status_code == 422, (q, r.text)
    assert requests.get(url, timeout=30).status_code == 401


def test_tinker_cli_run_list(service_client, server):
    tc = _created(service_client, rank=2)
    env = {**os.environ, "TINKER_BASE_URL": server.base_url, "TINKER_API_KEY": API_KEY}
    out = subprocess.run([sys.executable, "-m", "tinker.cli", "--format", "json", "run", "list", "--limit", "0"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert tc.model_id in {r["training_run_id"] for r in json.loads(out.stdout)["runs"]}
