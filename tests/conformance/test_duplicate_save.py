"""A `save_weights` over a name that is taken, without `overwrite`: the status the client gets."""


def test_a_duplicate_save_is_a_400_that_names_the_cause(service_client, server):
    """A status the SDK does not send again: it retries 408, 409, 429 and 5xx
    (`tinker/lib/retry_handler.py`), and the name stays taken."""
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.save_state("taken").result()
    sent = server.post("/api/v1/save_weights", {"model_id": tc.model_id, "path": "taken"})
    assert sent.status_code == 200, sent.text
    done = server.post("/api/v1/retrieve_future", {"request_id": sent.json()["request_id"]}, timeout=60)
    assert done.status_code == 400 and "already exists" in done.text, done.text
