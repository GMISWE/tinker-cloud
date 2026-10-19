"""core.routing: one immutable table, model -> endpoint (one or more base URLs)."""
import pytest

from tinkercloud.training.core.routing import InferenceEndpoint, RoutingError, RoutingTable


def test_publish_then_resolve_then_withdraw():
    t = RoutingTable()
    t.publish("m1", InferenceEndpoint(("http://10.0.0.1:30000/",)))
    assert t.endpoint_for("m1").base_urls == ("http://10.0.0.1:30000",)  # trailing slash normalised
    assert t.withdraw("m1").base_urls == ("http://10.0.0.1:30000",)
    assert t.withdraw("m1") is None


def test_unpublished_model_raises_not_none():
    t = RoutingTable()
    with pytest.raises(RoutingError):
        t.endpoint_for("ghost")


def test_republish_replaces_the_address():
    t = RoutingTable()
    t.publish("m", InferenceEndpoint(("http://a:1",)))
    t.publish("m", InferenceEndpoint(("http://b:2",)))
    assert t.endpoint_for("m").base_urls == ("http://b:2",)


def test_snapshot_is_immutable_and_stable_across_writes():
    t = RoutingTable()
    t.publish("m1", InferenceEndpoint(("http://a:1",)))
    snap = t.snapshot()
    t.publish("m2", InferenceEndpoint(("http://b:2",)))
    assert set(t.snapshot()) == {"m1", "m2"}
    assert dict(snap) == {"m1": InferenceEndpoint(("http://a:1",))}
    with pytest.raises(TypeError):
        snap["x"] = InferenceEndpoint(("http://c:3",))  # type: ignore[index]


def test_endpoint_keeps_every_leader_url_in_order():
    ep = InferenceEndpoint(("http://n1:8001/", "http://n2:8001"))
    assert ep.base_urls == ("http://n1:8001", "http://n2:8001")


def test_endpoint_must_be_http_urls_and_non_empty():
    with pytest.raises(ValueError):
        InferenceEndpoint(("10.0.0.1:30000",))
    with pytest.raises(ValueError):
        InferenceEndpoint(("http://a:1", "b:2"))
    with pytest.raises(ValueError):
        InferenceEndpoint(())
