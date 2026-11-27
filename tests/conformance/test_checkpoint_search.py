"""SDK 0.39 checkpoint search, from a real SDK save through cursor paging on the CPU backend."""
import pytest
import tinker
from tinker import types


def test_search_checkpoints_finds_saved_metadata_with_sdk(service_client):
    client = service_client.create_lora_training_client(base_model="fake/tiny", rank=2).result()
    tags = {"experiment": client.model_id, "candidate": "best"}
    paths = [client.save_state(f"best-{i}", user_metadata=tags).result().path for i in range(3)]
    client.save_state("other", user_metadata={"experiment": client.model_id, "candidate": "other"}).result()
    client.save_weights_for_sampler("sampler", user_metadata=tags).result()
    rest = service_client.create_rest_client()
    query = types.CheckpointSearchRequest(user_metadata=tags, checkpoint_type="training",
                                         base_models=["fake/tiny"], limit=1)
    seen = []
    for _ in range(4):
        response = rest.search_checkpoints(query).result()
        assert all(c.base_model == "fake/tiny" and c.user_metadata == tags for c in response.checkpoints)
        seen.extend(c.tinker_path for c in response.checkpoints)
        if response.next_cursor is None:
            break
        query = query.model_copy(update={"cursor": response.next_cursor})
    else:
        pytest.fail("checkpoint search did not exhaust its cursor")
    assert len(seen) == len(set(seen)) == len(paths)
    assert set(seen) == set(paths)
    empty = rest.search_checkpoints(types.CheckpointSearchRequest(base_models=[])).result()
    assert empty.checkpoints == [] and empty.next_cursor is None
    with pytest.raises(tinker.NotFoundError, match="missing-project"):
        rest.search_checkpoints(types.CheckpointSearchRequest(project_ids=["missing-project"])).result()
