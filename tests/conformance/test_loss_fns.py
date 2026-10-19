"""loss_fn names are whitelisted and loss_fn_config is validated per name and delivered to the backend."""
import pytest
from tinker import types

from .conftest import make_datum


def _fb_body(model_id, loss_fn, cfg=None):
    tokens = list(range(3, 9))
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {
            "weights": {"data": [1.0] * 5, "dtype": "float32", "shape": [5]},
            "target_tokens": {"data": tokens[1:], "dtype": "int64", "shape": [5]},
        },
    }
    inp = {"data": [datum], "loss_fn": loss_fn}
    if cfg is not None:
        inp["loss_fn_config"] = cfg
    return {"model_id": model_id, "forward_backward_input": inp}


@pytest.fixture(scope="module")
def model(service_client):
    return service_client.create_lora_training_client(base_model="fake/tiny", rank=4)


@pytest.mark.parametrize("loss_fn,cfg,detail", [
    ("ppo_loss", None, "Unknown loss_fn"),
    ("cross_entropy", {"beta": 0.1}, "not valid for"),
    ("ppo", {"beta": 0.1}, "not valid for"),
    ("cispo", {"clip_low_threshold": 1.5}, "clip thresholds"),
])
def test_invalid_loss_requests_are_400(server, model, loss_fn, cfg, detail):
    r = server.post("/api/v1/forward_backward", _fb_body(model.model_id, loss_fn, cfg))
    assert r.status_code == 400, (r.status_code, r.text)
    assert detail in r.json()["error"]
    r = server.post("/api/v1/forward", {"model_id": model.model_id,
                                        "forward_input": {**_fb_body(model.model_id, loss_fn, cfg)["forward_backward_input"]}})
    assert r.status_code == 400, (r.status_code, r.text)


@pytest.mark.parametrize("loss_fn,cfg", [
    ("cross_entropy", None),
    ("importance_sampling", None),
    ("ppo", {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}),
    ("cispo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}),
    ("dro", {"beta": 0.05}),
])
def test_supported_losses_reach_backend_with_config(service_client, server, model, loss_fn, cfg):
    tokens = list(range(20, 27))
    fb = model.forward_backward([make_datum(tokens)], loss_fn, loss_fn_config=cfg).result()
    model.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert list(fb.loss_fn_outputs[0]["logprobs"].data) == pytest.approx([-(t % 7) / 7.0 for t in tokens[:-1]], rel=1e-6)
    for k, v in (cfg or {}).items():
        assert fb.metrics[f"{k}:mean"] == v, fb.metrics  # fb metrics carry a ":<reduction>" suffix
    seen = [t for t in server.trace() if t["op"] == "forward_backward" and t["model_id"] == model.model_id]
    assert seen[-1]["loss_fn"] == loss_fn
    assert (seen[-1]["loss_fn_config"] or None) == cfg
