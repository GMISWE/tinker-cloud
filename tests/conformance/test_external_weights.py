"""save_weights_external + external_weights_urls over the real SDK: the
export lands as an `external` checkpoint, every file gets a signed URL the
server itself serves, and the URL is the only credential."""
import hashlib
import hmac
import time

import requests
from tinker import types

from .conftest import API_KEY, SIGNING_KEY, make_datum


def _retrieve(server, request_id):
    return server.post("/api/v1/retrieve_future", {"request_id": request_id}, timeout=60)


def test_export_list_urls_download_delete(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
    tc.optim_step(types.AdamParams(learning_rate=0.5)).result()
    path = tc.save_weights_external("hf1").result().path
    assert path == f"tinker://{tc.model_id}/external_weights/hf1"

    rest = service_client.create_rest_client()
    listed = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert [(c.checkpoint_id, c.checkpoint_type, c.tinker_path) for c in listed] == [
        ("external_weights/hf1", "external", path)]
    assert listed[0].size_bytes > 0 and listed[0].expires_at is None

    urls = rest.get_external_weights_urls(path).result()
    assert set(urls.urls) == {"adapter_config.json", "adapter_model.safetensors"}
    assert urls.expires.timestamp() > time.time() + 3000
    root = server.checkpoint_base / tc.model_id / "external_weights" / "hf1"
    for relpath, url in urls.urls.items():
        assert url.startswith(server.base_url + "/api/v1/external_weights/")
        got = requests.get(url, timeout=30)                    # no API key: the URL is the credential
        assert got.status_code == 200 and got.content == (root / relpath).read_bytes()
        tampered = url[:-1] + ("0" if url[-1] != "0" else "1")
        assert requests.get(tampered, timeout=30).status_code == 404
    assert b"fake-adapter w=0.5 v=1" in (root / "adapter_model.safetensors").read_bytes()

    # an expired token, correctly signed, is refused
    exp = int(time.time()) - 1
    msg = "\n".join(["external_weights", tc.model_id, "hf1", "adapter_config.json", str(exp)]).encode()
    sig = hmac.new(SIGNING_KEY.encode(), msg, hashlib.sha256).hexdigest()
    expired = f"{server.base_url}/api/v1/external_weights/{tc.model_id}/hf1/adapter_config.json?exp={exp}&sig={sig}"
    assert requests.get(expired, timeout=30).status_code == 404

    # the SDK's own delete (checkpoint_id external_weights/hf1) removes record, bytes and URLs
    rest.delete_checkpoint_from_tinker_path(path).result()
    assert rest.list_checkpoints(tc.model_id).result().checkpoints == []
    assert not root.exists()
    assert requests.get(url, timeout=30).status_code == 404
    r = requests.get(f"{server.base_url}/api/v1/training_runs/{tc.model_id}/checkpoints/external_weights/hf1/external_weights_urls",
                     headers={"X-API-Key": API_KEY}, timeout=30)
    assert r.status_code == 404


def test_ttl_is_recorded_and_bounded(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    path = tc.save_weights_external("ttl", ttl_seconds=3600).result().path
    rest = service_client.create_rest_client()
    [c] = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert c.tinker_path == path and c.expires_at is not None
    assert 3500 < (c.expires_at.timestamp() - time.time()) <= 3600
    for bad in (10, 3599, 10 * 365 * 24 * 3600 + 1):
        r = server.post("/api/v1/save_weights_external", {"model_id": tc.model_id, "path": "x", "ttl_seconds": bad})
        assert r.status_code == 422, (bad, r.text)


def test_encryption_recipients_are_refused(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    r = server.post("/api/v1/save_weights_external",
                    {"model_id": tc.model_id, "path": "enc", "age_encryption_recipients": ["age1qqq"]})
    assert r.status_code == 400 and "encryption not supported" in r.json()["error"], r.text
    assert server.post("/api/v1/save_weights_external",
                       {"model_id": tc.model_id, "path": "plain", "age_encryption_recipients": []}).status_code == 200


def test_full_finetune_fails_clearly_not_silently(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=0)
    r = server.post("/api/v1/save_weights_external", {"model_id": tc.model_id, "path": "full"})
    assert r.status_code == 200, r.text
    fut = _retrieve(server, r.json()["request_id"])
    assert fut.status_code == 400 and "no HF export" in fut.json()["error"], fut.text
    assert not any((server.checkpoint_base / tc.model_id / "external_weights" / "full").rglob("*"))
    r = requests.get(f"{server.base_url}/api/v1/training_runs/{tc.model_id}/checkpoints/external_weights/full/external_weights_urls",
                     headers={"X-API-Key": API_KEY}, timeout=30)
    assert r.status_code == 500 and "failed" in r.json()["error"]   # the store's verdict on a failed save


def test_unknown_model_and_wrong_kind(service_client, server):
    assert server.post("/api/v1/save_weights_external", {"model_id": "model_missing", "path": "x"}).status_code == 404
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    p = tc.save_state("w1").result().path
    rest = service_client.create_rest_client()
    try:
        rest.get_external_weights_urls(p).result()
    except ValueError as e:
        assert "external weights" in str(e)   # the SDK refuses a weights path before any request
    else:
        raise AssertionError("SDK accepted a weights path for external_weights_urls")
