"""Every async result is checked against its response model before it is stored."""
import pytest

from tinkercloud.training.models.responses import RESULT_MODELS, ResultShapeError, validate_result

FB = {
    "loss_fn_output_type": "cross_entropy",
    "loss_fn_outputs": [{"logprobs": {"data": [-0.1, -0.2], "shape": [2], "dtype": "float32"}}],
    "metrics": {"loss:mean": 0.15, "num_tokens:sum": 2},
    "loss": 0.15,
    "deferred": False,
}


def test_well_formed_results_round_trip_unchanged():
    cases = {
        "forward_backward": FB,
        "forward": {**FB, "type": "forward"},
        "optim_step": {"success": True, "grad_norm": 1.5, "learning_rates": [], "model_id": "m",
                       "metrics": {"grad_norm": 1.5}, "weight_version": 3},
        "create_model": {"model_id": "m", "base_model": "Qwen/x", "lora_config": {"rank": 8}, "status": "ready"},
        "save_weights": {"path": "tinker://m/weights/s1", "checkpoint_path": "/data/x", "step_id": 1,
                         "name": "s1", "type": "save_weights"},
        "save_weights_for_sampler": {"path": None, "sampling_session_id": "ss1", "type": "save_weights_for_sampler"},
        "save_weights_external": {"path": "tinker://m/external_weights/e1", "size_bytes": 12,
                                  "type": "save_weights_external"},
        "load_weights": {"type": "load_weights", "path": "tinker://m/weights/s1", "model_id": "m"},
        "sample": {"sequences": [{"stop_reason": "stop", "tokens": [1, 2], "logprobs": [-0.1, -0.2], "text": None}],
                   "prompt_logprobs": None, "weight_version": None, "latest_weight_version": None},
        "unload_model": {"model_id": "m", "type": "unload_model"},
        "create_sampling_client": {"sampling_client_id": "sc", "model_path": "tinker://m", "status": "ready"},
    }
    assert set(cases) | {"asample"} == set(RESULT_MODELS)
    for op, result in cases.items():
        out = validate_result(op, result)
        # coercion only: ints become floats in metrics, nothing added or dropped
        assert set(out) == set(result), op
        assert out == {**result, **({"metrics": {k: float(v) for k, v in result["metrics"].items()}}
                                    if "metrics" in result else {})}, op


def test_deferred_and_classification_shapes_pass():
    validate_result("forward_backward", {"loss_fn_output_type": "ppo", "loss_fn_outputs": [],
                                         "metrics": {}, "deferred": True, "loss": None})
    validate_result("forward", {"loss_fn_output_type": "classification", "metrics": {},
                                "loss_fn_outputs": [{"logits": {"data": [0.1, 0.9], "shape": [1, 2], "dtype": "float32"}}]})
    validate_result("optim_step", {"metrics": {}, "loss_fn_outputs": [{"logprobs": {"data": [], "shape": [0], "dtype": "float32"}}]})
    validate_result("sample", {"sequences": [{"stop_reason": "length", "tokens": []}]})  # logprobs optional


@pytest.mark.parametrize("op,result,loc", [
    ("forward_backward", {"loss_fn_outputs": [], "metrics": {}}, "loss_fn_output_type"),
    ("forward_backward", {**FB, "loss_fn_outputs": [{"logprobs": [-0.1]}]}, "loss_fn_outputs.0.logprobs"),
    ("forward_backward", {**FB, "metrics": {"note": "n/a"}}, "metrics.note"),
    ("optim_step", {"metrics": {"grad_norm": [1.0]}}, "metrics.grad_norm"),
    ("save_weights", {"checkpoint_path": "/data/x"}, "path"),
    ("sample", {"sequences": [{"tokens": [1]}]}, "sequences.0.stop_reason"),
    ("create_model", {"base_model": "Qwen/x"}, "model_id"),
])
def test_malformed_results_fail_naming_the_field(op, result, loc):
    with pytest.raises(ResultShapeError, match=rf"{op} result does not match \w+: .*{loc}"):
        validate_result(op, result)


def test_unregistered_operation_passes_through():
    assert validate_result("telemetry", {"anything": 1}) == {"anything": 1}
