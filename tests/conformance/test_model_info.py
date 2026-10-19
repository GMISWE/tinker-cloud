"""LoRA info comes from the stored lora_config; a failed create leaves no session link."""
import sqlite3

import json
import requests

from .conftest import API_KEY


def test_get_info_and_weights_info_report_lora(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=16)
    info = server.post("/api/v1/get_info", {"model_id": tc.model_id}).json()
    assert info["is_lora"] is True and info["lora_rank"] == 16

    path = tc.save_state("info").result().path
    wi = server.post("/api/v1/weights_info", {"tinker_path": path}).json()
    assert wi == {"base_model": "fake/tiny", "is_lora": True, "lora_rank": 16,
                  "optimizer_config": {"type": "adamw"}}

    spath = tc.save_weights_for_sampler("sinfo").result().path
    assert spath == f"tinker://{tc.model_id}/sampler_weights/sinfo"
    wi = server.post("/api/v1/weights_info", {"tinker_path": spath}).json()
    assert wi["is_lora"] is True and wi["lora_rank"] == 16
    saved = [t for t in server.trace() if t["op"] == "save_checkpoint" and t["model_id"] == tc.model_id
             and t["root"].endswith(f"/{tc.model_id}/sampler_weights/sinfo")]
    assert saved and saved[0]["persist"] is True and saved[0]["step"] == 2


def test_full_param_model_reports_no_lora(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=0)
    info = server.post("/api/v1/get_info", {"model_id": tc.model_id}).json()
    assert info["is_lora"] is False and info["lora_rank"] is None


def test_failed_create_leaves_no_session_link(server, tmp_path):
    sess = server.post("/api/v1/create_session", {"tags": [], "user_metadata": {}, "sdk_version": "t"}).json()
    session_id = sess["session_id"]
    # a model whose config.json declares a classification head: the fake backend
    # is language-modeling only, so this create fails inside the task
    model_dir = tmp_path / "fake-cls"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"architectures": ["FakeForSequenceClassification"], "num_labels": 3, "max_position_embeddings": 2048}))
    r = server.post("/api/v1/create_model", {
        "session_id": session_id, "model_seq_id": 1, "base_model": str(model_dir),
        "lora_config": {"rank": 8},
    })
    assert r.status_code == 200, r.text
    fut = server.post("/api/v1/retrieve_future", {"request_id": r.json()["request_id"]}, timeout=60)
    assert fut.status_code == 400, fut.text  # failed future
    summary = requests.get(f"{server.base_url}/api/v1/sessions/{session_id}",
                           headers={"X-API-Key": API_KEY}, timeout=10)
    assert summary.status_code == 200, summary.text
    assert summary.json()["training_run_ids"] == []
    with sqlite3.connect(server.metadata_dir / "sessions.db") as db:
        rows = db.execute("SELECT model_id FROM session_models WHERE session_id = ?", (session_id,)).fetchall()
    assert rows == []
