"""Offline tests for the PARTITION_INV order-permutation trace.

The fidelity test replays all 13 recorded `order_perm_*.json` runs from
specs/014-gate-suite/results/ (monorepo layout). Those JSONs were written
by the probe's ORIGINAL binary verdict (order-invariant / not), so the
recorded verdict STRING is not comparable; what is asserted is that the
migrated three-way logic reproduces the recorded per-arm annotations and
numbers exactly, agrees with the recorded binary field where they overlap,
and lands each run in the class the investigation attributed it to.
"""

import json
import math
from pathlib import Path

import pytest

from gates import comparators, invariants, runner
from gates.traces.order_perm import OrderPerm, default_arms, verdict
from tests.fakes.gates import CannedRows, StubDriver

MONOREPO = Path(__file__).resolve().parents[4]
RESULTS = MONOREPO / "specs/014-gate-suite/results"
RECORDED = sorted(RESULTS.glob("order_perm_*.json"))

E0 = comparators.REGISTRY["e0_bitwise"]

# tag -> the class the investigation attributed the run to
# (INVESTIGATION-miles-split.md §DEFECT 2 / §THE nemo_rl PEER PROBE).
EXPECTED = {
    # miles pre-fix: the gradient is a function of the rank partition
    "d2_dp2_instr": "PARTITION_SENSITIVE",
    "dynON_dp2": "PARTITION_SENSITIVE",
    "dynoff_dp2": "PARTITION_SENSITIVE",
    "seed7_dp2": "PARTITION_SENSITIVE",
    # dp=1 (no partition exists) and every post-fix miles run
    "d2_dp1_instr": "INVARIANT",
    "d2_dp2_FIXED": "INVARIANT",
    "d2_dp4_FIXED": "INVARIANT",
    "d2_dp2_optB": "INVARIANT",
    "d2_dp2_optB2": "INVARIANT",
    "seed7_dp1": "INVARIANT",
    # nemo_rl: miles' exact inverse -- the rank-flip is bit-identical and
    # what moves is the concatenation order, at the fp floor
    "nemorl_dp1": "PARTITION_INVARIANT_ORDER_SENSITIVE",
    "nemorl_dp2": "PARTITION_INVARIANT_ORDER_SENSITIVE",
    "nemorl_dp4": "INVARIANT",
}


def rows_for(values, arms=None):
    arms = arms or [a for a, _ in default_arms(8)]
    orders = dict(default_arms(8))
    return [
        {"arm": a, "order": orders.get(a, list(range(8))), "grad_norm": v}
        for a, v in zip(arms, values)
    ]


class TestVerdict:
    def test_all_bit_identical(self):
        v = verdict(rows_for([3.0] * 5), E0)
        assert v["outcome"] == "INVARIANT"
        assert v["passed"] is True
        assert v["delta"] == 0.0 and v["max_rel_diff"] == 0.0

    def test_partition_sensitive(self):
        # SWAP_PAIRS and REVERSE induce the SAME partition: they agree with
        # each other while differing from IDENT -- the miles signature
        v = verdict(rows_for([3.0, 3.01, 3.01, 3.01, 3.0]), E0)
        assert v["outcome"] == "PARTITION_SENSITIVE"
        assert v["passed"] is False
        assert v["same_partition_arms_agree"] is True
        assert v["partition_reproducible"] is True
        assert v["delta"] == pytest.approx(0.01 / 3.0)
        assert "SAME rank partition" in v["verdict"]

    def test_partition_invariant_but_order_sensitive(self):
        # the rank-flip is bit-identical; only the reversal moves
        v = verdict(rows_for([3.0, 3.0, 3.0 + 1e-6, 3.0, 3.0]), E0)
        assert v["outcome"] == "PARTITION_INVARIANT_ORDER_SENSITIVE"
        # the partition does not reach the gradient, which is what
        # PARTITION_INV claims; the residual is reported, not a failure
        assert v["passed"] is True
        assert v["delta"] == 0.0
        assert v["max_rel_diff"] == pytest.approx(1e-6 / 3.0)
        assert "commutative but not associative" in v["verdict"]

    def test_unclassified_when_neither_signature_matches(self):
        # SWAP_PAIRS moves but REVERSE lands somewhere else: the partition
        # and the order cannot be separated, so the gate must not guess
        v = verdict(rows_for([3.0, 3.01, 3.02, 3.01, 3.0]), E0)
        assert v["outcome"] == "ORDER_SENSITIVE_UNCLASSIFIED"
        assert v["passed"] is None
        assert v["same_partition_arms_agree"] is False

    def test_nondeterministic_is_inconclusive(self):
        v = verdict(rows_for([3.0, 3.0, 3.0, 3.0, 3.5]), E0)
        assert v["outcome"] == "INCONCLUSIVE"
        assert v["passed"] is None

    def test_missing_determinism_control(self):
        rows = rows_for([3.0, 3.0, 3.0, 3.0], arms=["IDENT", "SWAP_PAIRS",
                                                    "REVERSE", "SWAP_PAIRS_R"])
        v = verdict(rows, E0)
        assert v["outcome"] == "NO_DETERMINISM_CONTROL"
        assert v["passed"] is None

    def test_no_partition_arm(self):
        # a sweep without the rank-flipping permutation asks nothing about
        # the partition, however bit-identical its other arms are
        rows = rows_for([3.0, 3.0, 3.0], arms=["IDENT", "REVERSE", "IDENT_R"])
        v = verdict(rows, E0)
        assert v["outcome"] == "NO_PARTITION_ARM"
        assert v["passed"] is None
        assert "cannot ask" in v["verdict"]


class TestDefaultArms:
    def test_probe_compatible_orders(self):
        assert default_arms(8) == [
            ("IDENT", [0, 1, 2, 3, 4, 5, 6, 7]),
            ("SWAP_PAIRS", [1, 0, 3, 2, 5, 4, 7, 6]),
            ("REVERSE", [7, 6, 5, 4, 3, 2, 1, 0]),
            ("SWAP_PAIRS_R", [1, 0, 3, 2, 5, 4, 7, 6]),
            ("IDENT_R", [0, 1, 2, 3, 4, 5, 6, 7]),
        ]

    def test_odd_n_rejected(self):
        # SWAP_PAIRS is what flips the partition; without pairs it is not
        # the arm the verdict logic thinks it is
        with pytest.raises(ValueError, match="even"):
            default_arms(7)


@pytest.mark.skipif(not RECORDED, reason="monorepo specs/014 results not present")
class TestRecordedReplay:
    @pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.stem)
    def test_reproduces_recorded_observations(self, path):
        doc = json.loads(path.read_text())
        rows = [
            {k: v for k, v in r.items() if k not in ("bit_identical", "rel_diff")}
            for r in doc["arms"]
        ]
        v = verdict(rows, E0)

        assert v["determinism_ok"] == doc["determinism_ok"]
        assert v["permutation_invariant"] == doc["order_invariant"]
        rec = doc["max_rel_diff"]
        assert v["max_rel_diff"] == rec or (
            math.isnan(v["max_rel_diff"]) and math.isnan(rec)
        )
        for new, old in zip(rows, doc["arms"]):
            if "bit_identical" in old:
                assert new["bit_identical"] == old["bit_identical"]
                assert new["rel_diff"] == pytest.approx(old["rel_diff"], rel=1e-12)

    @pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.stem)
    def test_classifies_recorded_run(self, path):
        tag = json.loads(path.read_text())["tag"]
        assert tag in EXPECTED, f"unattributed recorded run {tag}"
        doc = json.loads(path.read_text())
        v = verdict([dict(r) for r in doc["arms"]], E0)
        assert v["outcome"] == EXPECTED[tag]

    def test_every_expected_tag_has_a_recording(self):
        tags = {json.loads(p.read_text())["tag"] for p in RECORDED}
        assert tags == set(EXPECTED)


class StubTrace(CannedRows, OrderPerm):
    def execute(self, driver):
        self.created_models.append("model_perm")
        return super().execute(driver)


class TestRunner:
    def test_verdict_doc_matches_schema(self, tmp_path):
        driver = StubDriver()
        doc = runner.run(
            invariants.get("PARTITION_INV"),
            StubTrace(rows_for([3.0, 3.01, 3.01, 3.01, 3.0])),
            tag="stub",
            out_dir=str(tmp_path),
            expect="PARTITION_SENSITIVE",
            driver=driver,
        )
        schema = json.loads(
            (Path(runner.__file__).parent / "verdicts" / "schema.json").read_text()
        )
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(doc, schema)
        assert doc["gate"] == "PARTITION_INV.order_perm"
        assert doc["claimed_level"] == "E0"
        assert doc["passed"] is False and doc["expect_met"] is True
        assert driver.deleted == ["model_perm"]
        written = json.loads(
            (tmp_path / "partition_inv.order_perm_stub.json").read_text()
        )
        assert written["artifacts"] == doc["artifacts"]

    def test_registered_on_the_cli(self):
        assert "order_perm" in runner.TRACES
