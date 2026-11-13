"""worker._micro_batch: a micro-batch is never narrower than MIN_MICRO_BATCH_WIDTH columns, since
Transformer Engine's fused attention refuses a backward at sequence length 1. Padding rows and
one-token rows run at that width; the added columns are zero, past every row's length."""
import pytest
import torch

pytest.importorskip("nemo_automodel")

from tinkercloud.training.backends.automodel.converter import Batch, LossFn
from tinkercloud.training.backends.automodel.worker import MIN_MICRO_BATCH_WIDTH, _micro_batch


def _batch(ids, lengths):
    ids = torch.tensor(ids, dtype=torch.long)
    return Batch(loss_fn=LossFn.CROSS_ENTROPY, input_ids=ids, lengths=torch.tensor(lengths),
                 inputs={"target_tokens": ids.clone(), "weights": (ids != 0).float()}, request_rows=())


def test_padding_rows_run_at_the_minimum_width():
    mb = _micro_batch(_batch([[7, 8, 9], [0, 0, 0]], [3, 0]), 1, 2)   # the padding row alone
    assert MIN_MICRO_BATCH_WIDTH == 2 and mb.input_ids.shape == (1, 2)
    assert int(mb.lengths[0]) == 0


def test_a_one_token_shard_is_padded_with_zero_columns():
    mb = _micro_batch(_batch([[5], [0]], [1, 0]), 0, 2)   # the shard itself is one column wide
    assert mb.input_ids.tolist() == [[5, 0], [0, 0]]
    assert mb.inputs["weights"].tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert mb.lengths.tolist() == [1, 0]


def test_a_longer_micro_batch_is_cut_to_its_longest_row():
    mb = _micro_batch(_batch([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]], [4, 2]), 0, 2)
    assert mb.input_ids.shape == (2, 4)
