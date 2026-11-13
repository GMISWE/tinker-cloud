"""The server logs one line for every finished forward, forward_backward and optim_step request: what
the request carried and where its seconds went. The lines are read through `RequestLine.parse`, the
format's one definition."""
import time
from collections import Counter

import pytest
from tinker import types

from tinkercloud.training.core.request_log import RequestLine

from .conftest import make_datum

SETTLE_S = 5.0   # a line is written after the request's result is stored, so after the client has it


def _lines_of(server, model_id: str, count: int) -> list:
    """The request lines of `model_id` in the server's logs, once `count` of them are there."""
    deadline = time.monotonic() + SETTLE_S
    while True:
        text = server.log_path.read_text()
        if server.engine_log_path.exists():   # the split server's engine writes the lines
            text += server.engine_log_path.read_text()
        parsed = [RequestLine.parse(line) for line in text.splitlines()]
        lines = [line for line in parsed if line is not None and line.model_id == model_id]
        if len(lines) >= count or time.monotonic() > deadline:
            return lines
        time.sleep(0.05)


def _waited(submit) -> float:
    """Seconds from sending a request to having its result."""
    start = time.monotonic()
    submit().result()
    return time.monotonic() - start


def test_every_finished_pass_request_has_one_line(service_client, server):
    """The line's fields, for each of the three operations and for a request that fails."""
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    model_id = tc.model_id
    data = [make_datum([1, 2, 3, 4]), make_datum([5, 6, 7])]
    waited = {
        ("forward", "completed"): _waited(lambda: tc.forward(data, "cross_entropy")),
        ("forward_backward", "completed"): _waited(lambda: tc.forward_backward(data, "cross_entropy")),
        ("optim_step", "completed"): _waited(lambda: tc.optim_step(types.AdamParams(learning_rate=0.1))),
    }
    fail = server.gate / f"{model_id}.forward.fail"
    fail.touch()
    try:
        with pytest.raises(Exception, match="failed by"):
            tc.forward(data, "cross_entropy").result()
    finally:
        fail.unlink()

    lines = _lines_of(server, model_id, 4)
    assert Counter((line.operation, line.outcome) for line in lines) == Counter(
        [*waited, ("forward", "failed")]), [line.format() for line in lines]
    assert len({line.request_id for line in lines}) == 4
    for line in lines:
        assert line.size.datums == (None if line.operation == "optim_step" else 2), line.format()
        assert line.size.body_bytes > 0, line.format()
        assert min(line.queued_s, line.backend_s, line.store_s) >= 0, line.format()
        if line.outcome == "completed":
            assert line.queued_s + line.backend_s + line.store_s <= waited[(line.operation, "completed")], line.format()


def test_a_request_with_more_datums_reports_more_body_bytes(service_client, server):
    """The body bytes are the request's own, not a constant: they grow with what the request carries."""
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    tc.forward([make_datum([1, 2, 3, 4])], "cross_entropy").result()
    tc.forward([make_datum(list(range(i, i + 40))) for i in range(8)], "cross_entropy").result()
    by_datums = {line.size.datums: line for line in _lines_of(server, tc.model_id, 2)}
    assert sorted(by_datums) == [1, 8]
    assert by_datums[8].size.body_bytes > by_datums[1].size.body_bytes
