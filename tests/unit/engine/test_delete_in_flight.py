"""One delete per model at a time on the engine: a second delete of a model whose backend delete is
still running is answered 409 and never reaches the backend, whose teardown would run twice."""
import queue
import threading

import httpx

from .support import SOURCE, client, engine, traced


def test_a_second_delete_while_the_first_runs_is_409_and_the_backend_deletes_once(tmp_path):
    with engine(tmp_path) as (c, _):
        (tmp_path / "gate" / f"{SOURCE}.delete.hold").touch()
        answers: "queue.Queue[httpx.Response]" = queue.Queue()

        def delete() -> None:
            with client(tmp_path) as own:
                answers.put(own.post("/api/v1/delete_model", json={"model_id": SOURCE}))

        threads = [threading.Thread(target=delete) for _ in range(2)]
        for t in threads:
            t.start()
        # one delete is held inside the backend; the other must not wait for it
        refused = answers.get(timeout=10)
        assert refused.status_code == 409 and "delete is in progress" in refused.json()["error"], refused.text
        (tmp_path / "gate" / f"{SOURCE}.delete.hold").unlink()
        finished = answers.get(timeout=30)
        assert finished.status_code == 200, finished.text
        for t in threads:
            t.join(timeout=30)
        after = c.post("/api/v1/delete_model", json={"model_id": SOURCE})
        assert after.status_code == 404, after.text
    assert traced(tmp_path, "delete_model") == [SOURCE]
