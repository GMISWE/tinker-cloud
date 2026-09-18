"""core.routing: one model -> endpoint table, lock-free reads over an immutable snapshot."""
import pytest

from tinkercloud.training.core.routing import InferenceEndpoint, RoutingError, RoutingTable


def test_publish_then_resolve_then_withdraw():
    t = RoutingTable()
    t.publish("m1", InferenceEndpoint("http://10.0.0.1:30000/"))
    assert t.endpoint_for("m1").base_url == "http://10.0.0.1:30000"  # trailing slash normalised
    assert t.withdraw("m1").base_url == "http://10.0.0.1:30000"
    with pytest.raises(RoutingError):
        t.endpoint_for("m1")


def test_unpublished_model_raises_not_none():
    t = RoutingTable()
    with pytest.raises(RoutingError):
        t.endpoint_for("never")
    assert t.withdraw("never") is None  # in-process engines have no route to withdraw


def test_republish_replaces_the_address():
    t = RoutingTable()
    t.publish("m", InferenceEndpoint("http://a:1"))
    t.publish("m", InferenceEndpoint("http://b:2"))
    assert t.endpoint_for("m").base_url == "http://b:2"


def test_snapshot_is_immutable_and_stable_across_writes():
    t = RoutingTable()
    t.publish("m1", InferenceEndpoint("http://a:1"))
    snap = t.snapshot()
    t.publish("m2", InferenceEndpoint("http://b:2"))
    t.withdraw("m1")
    assert dict(snap) == {"m1": InferenceEndpoint("http://a:1")}
    with pytest.raises(TypeError):
        snap["x"] = InferenceEndpoint("http://c:3")  # type: ignore[index]


def test_endpoint_must_be_http_url():
    with pytest.raises(ValueError):
        InferenceEndpoint("10.0.0.1:30000")
