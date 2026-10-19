"""Automodel converter: per-request validation, coalescing, rank shards, row splits."""
import pytest
import torch

from tinkercloud.training.backends.automodel.converter import (
    coalesce,
    convert_request,
    row_sums,
    shard,
    split_rows,
)
from tinkercloud.training.models.requests import Datum
from tests.datums import rl_datum


def ce_chunked(tokens, targets, weights):
    return Datum.model_validate({
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {"target_tokens": {"data": targets}, "weights": {"data": weights}},
    })


def test_layout_is_position_aligned_no_shift():
    r = convert_request([ce_chunked([1, 2, 3], [2, 3, 4], [0.0, 1.0, 1.0])], "cross_entropy")
    b = coalesce([r], num_ranks=1)
    assert b.input_ids.tolist() == [[1, 2, 3]]
    assert b.inputs["target_tokens"].tolist() == [[2, 3, 4]]
    assert b.inputs["weights"].tolist() == [[0.0, 1.0, 1.0]]
    assert set(b.inputs) == {"target_tokens", "weights"}


def test_coalesce_pads_rows_to_rank_multiple_and_records_requests():
    a = convert_request([ce_chunked([1, 2], [2, 3], [1, 1]), ce_chunked([5], [6], [1])], "cross_entropy")
    b = convert_request([ce_chunked([7, 8, 9], [8, 9, 10], [1, 1, 1])], "cross_entropy")
    batch = coalesce([a, b], num_ranks=2)
    assert batch.num_rows == 4                       # 3 real + 1 padding row
    assert batch.request_rows == ((0, 2), (2, 3))
    assert batch.lengths.tolist() == [2, 1, 3, 0]
    # padding (columns past a length, and the padding row) carries zero weight
    assert batch.inputs["weights"].tolist() == [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]


def test_shard_gives_each_rank_equal_contiguous_rows():
    r = convert_request([ce_chunked([i], [i], [1]) for i in range(5)], "cross_entropy")
    chunks = shard(coalesce([r], num_ranks=3), num_ranks=3)
    assert [c.num_rows for c in chunks] == [2, 2, 2]
    assert [c.input_ids[:, 0].tolist() for c in chunks] == [[0, 1], [2, 3], [4, 0]]
    assert chunks[2].lengths.tolist() == [1, 0]


def test_split_rows_trims_to_length_and_drops_padding_rows():
    a = convert_request([ce_chunked([1, 2], [2, 3], [1, 1])], "cross_entropy")
    b = convert_request([ce_chunked([4, 5, 6], [5, 6, 7], [1, 1, 1])], "cross_entropy")
    batch = coalesce([a, b], num_ranks=4)
    values = torch.arange(batch.num_rows * 3, dtype=torch.float32).reshape(batch.num_rows, 3)
    assert split_rows(batch, values) == [[[0.0, 1.0]], [[3.0, 4.0, 5.0]]]
    assert row_sums(batch, torch.tensor([1.0, 2.0, 99.0, 99.0])) == [1.0, 2.0]


def test_rl_inputs_are_read():
    r = convert_request([rl_datum([1, 2], [2, 3], [-0.5, -0.25], [1.0, 0.0])], "ppo")
    b = coalesce([r], num_ranks=1)
    assert b.inputs["logprobs"].tolist() == [[-0.5, -0.25]]
    assert b.inputs["advantages"].tolist() == [[1.0, 0.0]]


@pytest.mark.parametrize("datum,loss_fn,message", [
    (ce_chunked([1, 2], [2, 3], [1, 1]), "importance_sampling", "missing advantages, logprobs; not read: weights"),
    (rl_datum([1, 2], [2, 3], [0, 0], [0, 0], mask=[1, 1]), "ppo", "not read: mask"),
    (ce_chunked([1, 2], [2, 3, 4], [1, 1]), "cross_entropy", "target_tokens has 3 entries for 2 tokens"),
    (ce_chunked([1, 2], [2, 3], [1]), "cross_entropy", "weights has 1 entries for 2 tokens"),
])
def test_invalid_datum_names_index_and_reason(datum, loss_fn, message):
    with pytest.raises(ValueError, match="datum 1: .*" + message.replace("(", r"\(")):
        convert_request([ce_chunked([1], [1], [1]) if loss_fn == "cross_entropy" else
                         rl_datum([1], [1], [0], [0]), datum], loss_fn)


def test_unserved_loss_and_empty_request_are_refused():
    with pytest.raises(ValueError, match="'cispo' is not served"):
        convert_request([ce_chunked([1], [1], [1])], "cispo")
    with pytest.raises(ValueError, match="no datums"):
        convert_request([], "cross_entropy")


def test_image_chunk_is_refused():
    d = Datum.model_validate({
        "model_input": {"chunks": [{"type": "image", "data": "aGk=", "expected_tokens": 4}]},
        "loss_fn_inputs": {"target_tokens": {"data": [1]}, "weights": {"data": [1]}},
    })
    with pytest.raises(ValueError, match="datum 0: image input"):
        convert_request([d], "cross_entropy")
