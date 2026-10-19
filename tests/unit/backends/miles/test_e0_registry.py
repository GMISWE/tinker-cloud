"""E0 calibration registry resolution.

The co-batch guard threshold is a measured per-(model x parallel config)
constant. Resolution order: env override > registry entry (null = guard
open) > UNCALIBRATED (co-batching must be disabled). These tests pin that
contract and that the shipped registry parses.
"""

import json

import pytest

try:
    from training.backends.miles.e0_registry import (
        _REGISTRY_PATH,
        resolve_e0,
    )
except ImportError:  # pragma: no cover
    resolve_e0 = None

pytestmark = pytest.mark.skipif(
    resolve_e0 is None, reason="training package not importable"
)


@pytest.fixture()
def registry(tmp_path):
    p = tmp_path / "e0_calibration.json"
    p.write_text(json.dumps({
        "schema": 1,
        "entries": [
            {"model": "ModelA", "tp": 1, "dp": 2,
             "threshold_tokens_per_rank": 512,
             "measured_range_tokens_per_rank": [189, 1512]},
            {"model": "ModelB", "tp": 2, "dp": 1,
             "threshold_tokens_per_rank": None,
             "measured_range_tokens_per_rank": [189, 1512]},
            {"model": "ModelC", "tp": 1, "dp": 2,
             "threshold_tokens_per_rank": -1,
             "measured_range_tokens_per_rank": [189, 1512]},
        ],
    }))
    return p


def test_configured_override_wins(registry):
    res = resolve_e0("Org/ModelA", tp=1, dp=2, registry_path=registry, override=256)
    assert (res.source, res.threshold, res.disable_cobatch) == ("override", 256, False)


def test_override_zero_disables_guard_not_cobatch(registry):
    res = resolve_e0("Org/ModelA", tp=1, dp=2, registry_path=registry, override=0)
    assert (res.source, res.threshold, res.disable_cobatch) == ("override", 0, False)


def test_registry_threshold(registry):
    res = resolve_e0("Org/ModelA", tp=1, dp=2, registry_path=registry)
    assert (res.source, res.threshold, res.disable_cobatch) == ("registry", 512, False)


def test_registry_null_means_guard_open(monkeypatch, registry):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    res = resolve_e0("ModelB", tp=2, dp=1, registry_path=registry)
    assert (res.source, res.threshold, res.disable_cobatch) == ("registry", 0, False)


def test_registry_no_safe_region_disables_cobatch(monkeypatch, registry):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    res = resolve_e0("ModelC", tp=1, dp=2, registry_path=registry)
    assert (res.source, res.threshold, res.disable_cobatch) == ("registry", 0, True)


def test_parallel_shape_is_part_of_the_key(monkeypatch, registry):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    res = resolve_e0("ModelA", tp=2, dp=1, registry_path=registry)
    assert res.source == "uncalibrated"
    assert res.disable_cobatch is True


def test_missing_entry_disables_cobatch(monkeypatch, registry):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    res = resolve_e0("NeverMeasured", tp=1, dp=2, registry_path=registry)
    assert (res.source, res.disable_cobatch) == ("uncalibrated", True)


def test_unreadable_registry_is_uncalibrated(monkeypatch, tmp_path):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    res = resolve_e0("ModelA", tp=1, dp=2, registry_path=tmp_path / "absent.json")
    assert (res.source, res.disable_cobatch) == ("uncalibrated", True)


def test_shipped_registry_parses_and_seeds_resolve(monkeypatch):
    monkeypatch.delenv("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", raising=False)
    data = json.loads(_REGISTRY_PATH.read_text())
    assert data["schema"] == 1 and data["entries"]
    r05 = resolve_e0("Qwen/Qwen2.5-0.5B", tp=1, dp=2)
    assert (r05.source, r05.threshold) == ("registry", 512)
    r8b = resolve_e0("Qwen/Qwen3-8B-Base", tp=2, dp=1)
    assert (r8b.source, r8b.threshold, r8b.disable_cobatch) == ("registry", 0, False)
