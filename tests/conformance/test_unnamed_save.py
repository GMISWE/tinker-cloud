"""A save that names no checkpoint: the server chooses the name, and two such saves are two checkpoints."""
import pytest


def _saved_path(server, route: str, model_id: str) -> str:
    sent = server.post(route, {"model_id": model_id})
    assert sent.status_code == 200, sent.text
    done = server.post("/api/v1/retrieve_future", {"request_id": sent.json()["request_id"]}, timeout=60)
    assert done.status_code == 200, done.text
    return done.json()["path"]


@pytest.mark.parametrize("route, prefix", [
    ("/api/v1/save_weights", "checkpoint_"),
    ("/api/v1/save_weights_for_sampler", "sampler_"),
])
def test_two_unnamed_saves_in_one_second_are_two_checkpoints(service_client, server, route, prefix):
    """The name was `<prefix><unix seconds>` alone. The second `save_weights` of a second failed with
    "already exists" under a name the client never chose; the second sampler save replaced the first."""
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    paths = [_saved_path(server, route, tc.model_id) for _ in range(3)]
    names = [path.rsplit("/", 1)[1] for path in paths]
    seconds = {name.rsplit("_", 1)[0] for name in names}
    assert len(seconds) < 3, f"no two of the three saves fell in one second ({names}): the test did not reach the case"
    assert len(set(names)) == 3 and all(name.startswith(prefix) for name in names)
    listed = service_client.create_rest_client().list_checkpoints(tc.model_id).result().checkpoints
    assert {c.tinker_path for c in listed} == set(paths)
