"""A forward_backward result of millions of values in the futures store: the row keeps an array-valued
tensor once, in the proto view."""
import numpy as np

from tinkercloud.training.proto.wire import forward_backward_json, stored_forward_backward_result
from tinkercloud.training.storage.futures import FuturesStorage

VALUES = 2_000_000


def test_a_result_read_back_from_the_row_has_its_array_tensors_from_the_proto_view(tmp_path):
    """The stored result holds an array tensor's dtype and shape and no values. A store opened on the
    same file must still give a JSON client those values, as the proto view carries them, beside a
    list tensor's own numbers, a field one datum lacks, and every other field of the result."""
    logprobs = np.linspace(-9.0, 0.0, VALUES, dtype=np.float32)
    result = {"loss_fn_output_type": "cross_entropy", "metrics": {"loss:sum": 1.5}, "weight_version": 3,
              "loss_fn_outputs": [{"logprobs": {"data": logprobs, "dtype": "float32", "shape": [VALUES // 16, 16]}},
                                {"logprobs": {"data": logprobs[:16], "dtype": "float32", "shape": [1, 16]},
                                 "elementwise_loss": {"data": [1 / 7, 2 / 7], "dtype": "float32", "shape": [2]}}]}
    stored, blob = stored_forward_backward_result(result)
    assert "data" not in stored["loss_fn_outputs"][0]["logprobs"]
    store = FuturesStorage(tmp_path / "f.db")
    store.save_future("r1", "forward_backward", {"seq": 1}, model_id="m", seq_id=1)
    store.update_status("r1", "completed", stored, result_proto=blob)

    for fut in (store.get_future("r1"), FuturesStorage(tmp_path / "f.db").get_future("r1")):
        view = forward_backward_json(fut["result"], fut["result_proto"])
        assert view["weight_version"] == 3 and view["metrics"] == {"loss:sum": 1.5}
        long, short = view["loss_fn_outputs"]
        assert long == {"logprobs": {"data": logprobs.tolist(), "dtype": "float32", "shape": [VALUES // 16, 16]}}
        assert short["logprobs"]["data"] == logprobs[:16].tolist()
        assert short["elementwise_loss"]["data"] == [1 / 7, 2 / 7]   # a list keeps the backend's doubles
