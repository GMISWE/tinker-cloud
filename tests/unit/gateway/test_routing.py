"""An id names its engine; the gateway reads it from the path, the query or the body without opening the rest."""
import orjson

from training.gateway import routing
from training.proto import tinker_public_pb2 as pb
from training.utils.helpers import engine_name_of

HEX = "0123456789abcdef"


def _name(*args):
    named = routing.named_engine(*args)
    return named[1] if named else None


def test_engine_name_of_reads_both_kinds_and_underscored_names():
    assert engine_name_of(f"model_fake_{HEX}") == "fake"
    assert engine_name_of(f"req_nemo_rl_{HEX}") == "nemo_rl"
    assert engine_name_of(f"model_megatron_bridge_{HEX}") == "megatron_bridge"


def test_engine_name_of_is_none_for_another_shape():
    for other in ("model_x", "req_123", f"model_fake_{HEX[:15]}", f"model-fake-{HEX}", f"sess_fake_{HEX}", ""):
        assert engine_name_of(other) is None, other


def test_path_ids():
    assert _name(f"/api/v1/retrieve_future/req_fake_{HEX}", b"", b"", b"", None) == "fake"
    assert _name(f"/api/v1/training_runs/model_nemo_rl_{HEX}/checkpoints", b"", b"", b"", None) == "nemo_rl"
    assert _name(f"/api/v1/external_weights/model_fake_{HEX}/hf1/adapter_config.json", b"", b"", b"", None) == "fake"
    assert _name("/api/v1/retrieve_future/req_123", b"", b"", b"", None) is None
    assert _name("/api/v1/create_model", b"", b"", b"", None) is None


def test_get_tokenizer_reads_the_query():
    assert _name("/api/v1/get_tokenizer", f"model_id=model_fake_{HEX}".encode(), b"", b"", None) == "fake"
    assert _name("/api/v1/get_tokenizer", b"", b"", b"", None) is None


def test_json_bodies_carry_model_or_request_id():
    ct = b"application/json"
    assert _name("/api/v1/optim_step", b"", ct, b"", orjson.dumps({"model_id": f"model_fake_{HEX}"})) == "fake"
    assert _name("/api/v1/cancel_future", b"", ct, b"", orjson.dumps({"request_id": f"req_fake_{HEX}"})) == "fake"
    assert _name("/api/v1/optim_step", b"", ct, b"", orjson.dumps({"model_id": "model_x"})) is None
    assert _name("/api/v1/optim_step", b"", ct, b"", b"not json") is None
    assert _name("/api/v1/optim_step", b"", ct, b"", b"[1, 2]") is None
    assert _name("/api/v1/optim_step", b"", ct, b"", b"") is None


def test_json_body_of_an_op_that_carries_no_id_is_not_opened():
    body = orjson.dumps({"model_id": f"model_fake_{HEX}"})
    assert _name("/api/v1/create_session", b"", b"application/json", b"", body) is None
    assert not routing.reads_body("/api/v1/create_session")
    assert routing.reads_body("/api/v1/forward_backward")


def test_proto_body_is_routed_by_its_skeleton():
    body = pb.ForwardBackwardRequest(model_id=f"model_fake_{HEX}", loss_fn="cross_entropy").SerializeToString()
    assert _name("/api/v1/forward_backward", b"", b"application/x-protobuf", b"", body) == "fake"
    assert _name("/api/v1/forward_backward", b"", b"application/x-protobuf", b"", b"\xff\xff\xff") is None


def test_the_id_itself_comes_back_with_the_name():
    assert routing.named_engine(f"/api/v1/retrieve_future/req_fake_{HEX}", b"", b"", b"", None) == (f"req_fake_{HEX}", "fake")
    body = orjson.dumps({"model_path": f"tinker://model_other_{HEX}/sampler_weights/x"})
    assert routing.named_engine("/api/v1/create_sampling_client", b"", b"application/json", b"", body) == (f"model_other_{HEX}", "other")
    body = orjson.dumps({"sampling_session_id": "ss", "model_path": f"tinker://model_fake_{HEX}/weights/x"})
    assert routing.named_engine("/api/v1/asample", b"", b"application/json", b"", body) == (f"model_fake_{HEX}", "fake")
    body = orjson.dumps({"sampling_session_id": "ss"})
    assert routing.named_engine("/api/v1/asample", b"", b"application/json", b"", body) is None
    body = orjson.dumps({"path": f"tinker://model_fake_{HEX}/weights/x"})
    assert routing.named_engine("/api/v1/load_weights", b"", b"application/json", b"", body) == (f"model_fake_{HEX}", "fake")
    body = orjson.dumps({"tinker_path": f"tinker://model_other_{HEX}/weights/x"})
    assert routing.named_engine("/api/v1/weights_info", b"", b"application/json", b"", body) == (f"model_other_{HEX}", "other")
    assert routing.named_engine("/api/v1/load_weights", b"", b"application/json", b"", orjson.dumps({"path": "tinker://nope/w/x"})) is None


def test_a_compressed_body_is_not_opened():
    body = orjson.dumps({"model_id": f"model_fake_{HEX}"})
    assert _name("/api/v1/forward_backward", b"", b"application/json", b"zstd", body) is None
