"""Corrupt metadata is a server failure, not an absent record."""

from .support import SOURCE, engine as _engine


def test_a_corrupt_training_run_is_not_reported_missing(tmp_path):
    with _engine(tmp_path) as (client, _):
        (tmp_path / "meta" / "training_runs" / f"{SOURCE}.json").write_text("{")
        response = client.get(f"/api/v1/training_runs/{SOURCE}/checkpoints")
        assert response.status_code == 500, response.text
