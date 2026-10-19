"""Offline tests for the gate suite's pure layers: comparators, seg-sweep
verdict logic, arm parsing, runner verdict-JSON shape. No server/GPU.

Fidelity tests replay the recorded g1b results JSONs from specs/008 and
specs/013 (monorepo layout) and assert the migrated verdict logic
reproduces the probes' recorded verdicts byte-for-byte.
"""

import json
import math
from pathlib import Path

import pytest

from gates import comparators, invariants, runner
from gates.traces.seg_sweep import (
    DEFAULT_ARMS,
    SegSweep,
    parse_segmentations,
    verdict,
)
from tests.fakes.gates import CannedRows, StubDriver

MONOREPO = Path(__file__).resolve().parents[4]
RECORDED = [
    p
    for pattern in (
        "specs/008-q3-abstraction-tax/probes/results/g1b_*.json",
        "specs/013-a1-8b-spine/probes/results/g1b_8b_*.json",
    )
    for p in MONOREPO.glob(pattern)
]

E0 = comparators.REGISTRY["e0_bitwise"]


def rows_for(values, arms=None):
    arms = arms or [a for a, _ in DEFAULT_ARMS]
    segs = dict(DEFAULT_ARMS)
    return [
        {"arm": a, "segmentation": segs.get(a, [8]), "grad_norm": v}
        for a, v in zip(arms, values)
    ]


class TestComparators:
    def test_e0_bitwise(self):
        assert E0(1.5, 1.5).passed
        assert not E0(1.5, 1.5000001).passed
        assert E0(1.5, 1.5000001).delta == pytest.approx(1e-7 / 1.5, rel=1e-2)

    def test_e1_threshold(self):
        c = comparators.e1_threshold(2.80, 2.83, bar=0.05)
        assert c.passed and c.delta == pytest.approx(0.03) and c.threshold == 0.05
        assert not comparators.e1_threshold(2.80, 2.90, bar=0.05).passed

    def test_e2_envelope(self):
        assert comparators.e2_envelope(2.30, 2.25, 2.35).passed
        assert not comparators.e2_envelope(2.40, 2.25, 2.35).passed


class TestSegSweepVerdict:
    def test_all_bit_identical(self):
        v = verdict(rows_for([3.0] * 6), E0)
        assert v["outcome"] == "E0_ARBITRARY"
        assert v["passed"] is True
        assert v["determinism_ok"] and v["nonaligned_all_bit_identical"]
        assert v["nonaligned_reproducible"] is True

    def test_aligned_only(self):
        v = verdict(rows_for([3.0, 3.0, 3.0 + 1e-6, 3.0 + 2e-6, 3.0 + 1e-6, 3.0]), E0)
        assert v["outcome"] == "E0_ALIGNED_ONLY"
        assert v["passed"] is False
        assert v["aligned_bit_identical"] is True
        # the repeated [3,5] arm reproduced its value: segmentation-caused
        assert v["nonaligned_reproducible"] is True
        assert v["max_nonaligned_rel_diff"] == pytest.approx(2e-6 / 3.0)

    def test_broken_even_aligned(self):
        v = verdict(rows_for([3.0, 3.1, 3.2, 3.3, 3.2, 3.0]), E0)
        assert v["outcome"] == "E0_BROKEN"
        assert v["passed"] is False

    def test_nondeterministic_inconclusive(self):
        v = verdict(rows_for([3.0, 3.0, 3.0, 3.0, 3.0, 3.5]), E0)
        assert v["outcome"] == "INCONCLUSIVE"
        assert v["passed"] is None

    def test_verdict_string_formats_max_rel(self):
        v = verdict(rows_for([3.0, 3.0, 3.0 + 3e-6, 3.0 + 3e-6, 3.0 + 3e-6, 3.0]), E0)
        assert "max rel 1.000e-06" in v["verdict"]


class TestParseSegmentations:
    def test_probe_compatible_naming(self):
        arms = parse_segmentations("8;4,4;2,6;6,2;2,6;8", n=8)
        assert arms == [
            ("A1", [8]),
            ("ALIGNED", [4, 4]),
            ("S2+6", [2, 6]),
            ("S6+2", [6, 2]),
            ("S2+6_R", [2, 6]),
            ("A2", [8]),
        ]


@pytest.mark.skipif(not RECORDED, reason="monorepo specs/ results not present")
class TestRecordedReplay:
    @pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.stem)
    def test_reproduces_recorded_verdict(self, path):
        doc = json.loads(path.read_text())
        rows = [
            {k: v for k, v in r.items() if k not in ("bit_identical", "rel_diff")}
            for r in doc["arms"]
        ]
        v = verdict(rows, E0)
        assert v["verdict"] == doc["verdict"]
        assert v["determinism_ok"] == doc["determinism_ok"]
        assert v["aligned_bit_identical"] == doc["aligned_bit_identical"]
        assert v["nonaligned_all_bit_identical"] == doc["nonaligned_all_bit_identical"]
        assert v["nonaligned_reproducible"] == doc.get("nonaligned_reproducible")
        rec = doc["max_nonaligned_rel_diff"]
        assert v["max_nonaligned_rel_diff"] == rec or (
            math.isnan(v["max_nonaligned_rel_diff"]) and math.isnan(rec)
        )
        # recomputed per-arm annotations match what the probe wrote
        for new, old in zip(rows, doc["arms"]):
            if "bit_identical" in old:
                assert new["bit_identical"] == old["bit_identical"]


class StubTrace(CannedRows, SegSweep):
    pass


class TestRunner:
    def make_doc(self, values, expect=None, tmp_path=None):
        trace = StubTrace(rows_for(values))
        return runner.run(
            invariants.get("SPLIT_INV"),
            trace,
            tag="stub",
            out_dir=str(tmp_path) if tmp_path else None,
            expect=expect,
            driver=StubDriver(),
        )

    def test_verdict_doc_matches_schema(self, tmp_path):
        doc = self.make_doc([3.0] * 6, tmp_path=tmp_path)
        schema = json.loads(
            (Path(runner.__file__).parent / "verdicts" / "schema.json").read_text()
        )
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(doc, schema)
        assert doc["gate"] == "SPLIT_INV.seg_sweep"
        assert doc["claimed_level"] == "E0"
        assert doc["passed"] is True
        written = json.loads(
            (tmp_path / "split_inv.seg_sweep_stub.json").read_text()
        )
        assert written["artifacts"] == doc["artifacts"]

    def test_expect_grading(self):
        doc = self.make_doc(
            [3.0, 3.0, 3.0 + 1e-6, 3.0 + 2e-6, 3.0 + 1e-6, 3.0],
            expect="E0_ALIGNED_ONLY",
        )
        assert doc["passed"] is False
        assert doc["expect_met"] is True


class TestModelCleanup:
    """A gate owns the models it creates: miles does not auto-reap them, and
    a leftover model wedges the next gate's create_model in placement."""

    def _trace(self, rows, models):
        t = StubTrace(rows)
        t.created_models = list(models)
        return t

    def test_models_released_after_run(self):
        driver = StubDriver()
        trace = self._trace(rows_for([3.0] * 6), ["model_a"])
        runner.run(invariants.get("SPLIT_INV"), trace, tag="stub", driver=driver)
        assert driver.deleted == ["model_a"]
        assert trace.created_models == []

    def test_models_released_even_when_trace_raises(self):
        driver = StubDriver()
        trace = self._trace(rows_for([3.0] * 6), ["model_b"])

        def boom(_driver):
            trace.created_models.append("model_c")
            raise RuntimeError("arm exploded mid-sweep")

        trace.execute = boom
        with pytest.raises(RuntimeError, match="exploded"):
            runner.run(invariants.get("SPLIT_INV"), trace, tag="stub", driver=driver)
        assert driver.deleted == ["model_b", "model_c"]

    def test_cleanup_failure_does_not_mask_verdict(self):
        """A leaked model is an operational problem; losing the verdict that
        cost a cluster run is worse. Cleanup warns and the result stands."""

        class AngryDriver(StubDriver):
            def delete_model(self, model_id):
                raise RuntimeError("delete endpoint down")

        trace = self._trace(rows_for([3.0] * 6), ["model_d"])
        doc = runner.run(invariants.get("SPLIT_INV"), trace, tag="stub",
                         driver=AngryDriver())
        assert doc["outcome"] == "E0_ARBITRARY"
        assert doc["passed"] is True


class TestAlignedOnlySweep:
    """A sweep of only aligned arms asks nothing about split-invariance and
    must say so rather than crashing on an empty max()."""

    def test_no_nonaligned_arm_is_inconclusive(self):
        rows = [
            {"arm": "A1", "segmentation": [8], "grad_norm": 3.0},
            {"arm": "ALIGNED", "segmentation": [4, 4], "grad_norm": 3.0},
            {"arm": "A2", "segmentation": [8], "grad_norm": 3.0},
        ]
        v = verdict(rows, E0)
        assert v["outcome"] == "NO_NONALIGNED_ARM"
        assert v["passed"] is None
        assert v["delta"] == 0.0
        assert "cannot distinguish" in v["verdict"]

    def test_still_reports_a_failed_determinism_control(self):
        rows = [
            {"arm": "A1", "segmentation": [8], "grad_norm": 3.0},
            {"arm": "ALIGNED", "segmentation": [4, 4], "grad_norm": 3.0},
            {"arm": "A2", "segmentation": [8], "grad_norm": 3.5},
        ]
        v = verdict(rows, E0)
        assert v["outcome"] == "INCONCLUSIVE"
        assert v["passed"] is None
