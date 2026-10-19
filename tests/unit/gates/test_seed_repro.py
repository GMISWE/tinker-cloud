"""Offline tests for the SEED_REPRO trace.

No recorded JSONs exist to replay — the seeding gap was found by hand
(two same-width same-seed nemo_rl runs disagreeing on grad_norm,
3194.760 vs 3270.384) and fixed before a gate existed. The cases below
encode that observation and the miles one it is paired with: an engine
that ignores the client's seed and falls back to its own default (miles
pre-`995945e`, Megatron's --seed=1234) reproduces perfectly and must
still fail.
"""

import json
from pathlib import Path

import pytest

from gates import comparators, invariants, runner
from gates.traces.seed_repro import SeedRepro, default_arms, verdict

E0 = comparators.REGISTRY["e0_bitwise"]


def rows_for(values, seeds=(7, 7, 99)):
    arms = [a for a, _ in default_arms()]
    return [
        {"arm": a, "seed": s, "grad_norm": v}
        for a, s, v in zip(arms, seeds, values)
    ]


class TestVerdict:
    def test_seed_honored(self):
        v = verdict(rows_for([3141.8, 3141.8, 2900.0]), E0)
        assert v["outcome"] == "SEED_HONORED"
        assert v["passed"] is True
        assert v["delta"] == 0.0
        assert v["same_seed_bit_identical"] and v["different_seed_differs"]

    def test_seed_ignored_when_the_contrast_arm_agrees_too(self):
        # the failure the same-seed pair alone cannot see: perfectly
        # reproducible, and the client's value reaches nothing
        v = verdict(rows_for([3141.8, 3141.8, 3141.8]), E0)
        assert v["outcome"] == "SEED_IGNORED"
        assert v["passed"] is False
        assert v["same_seed_bit_identical"] is True
        assert v["different_seed_differs"] is False
        assert "accident of the engine's own default seed" in v["verdict"]

    def test_not_reproducible(self):
        # the recorded nemo_rl signature: same seed, two models, no seeding
        v = verdict(rows_for([3194.760, 3270.384, 2900.0]), E0)
        assert v["outcome"] == "NOT_REPRODUCIBLE"
        assert v["passed"] is False
        assert v["delta"] == pytest.approx(abs(3270.384 - 3194.760) / 3194.760)

    def test_no_contrast_arm_is_inconclusive(self):
        rows = rows_for([3141.8, 3141.8], seeds=(7, 7))[:2]
        v = verdict(rows, E0)
        assert v["outcome"] == "NO_CONTRAST_ARM"
        assert v["passed"] is None
        assert v["different_seed_rel_diff"] is None

    def test_no_repeat_arm_is_inconclusive(self):
        rows = [
            {"arm": "SEED_A", "seed": 7, "grad_norm": 3141.8},
            {"arm": "SEED_B", "seed": 99, "grad_norm": 2900.0},
        ]
        v = verdict(rows, E0)
        assert v["outcome"] == "NO_REPEAT_ARM"
        assert v["passed"] is None


class TestArms:
    def test_default_arms_pair_a_repeat_with_a_contrast(self):
        assert default_arms(7, 99) == [
            ("SEED_A", 7),
            ("SEED_A_R", 7),
            ("SEED_B", 99),
        ]

    def test_trace_params_carry_both_seeds(self):
        p = SeedRepro(seed=11, contrast_seed=12).params()
        assert p["seed"] == 11 and p["contrast_seed"] == 12
        assert [a["seed"] for a in p["arms"]] == [11, 11, 12]


class FakeClient:
    def __init__(self, model_id, grad_norm):
        self.model_id = model_id
        self._gn = grad_norm

    def forward_backward(self, datums, loss_fn):
        class _F:
            def result(self):
                return None

        return _F()


class FakeDriver:
    """Records the create/delete interleaving and serves canned grad_norms
    keyed by the seed each model was created with."""

    base_url = "stub://"

    def __init__(self, by_seed):
        self.by_seed = by_seed
        self.events = []
        self.live = set()
        self.grad_norms = {}

    def create_training_client(self, base_model, rank, seed):
        mid = f"model_{len(self.grad_norms) + 1}"
        self.events.append(("create", mid, seed))
        self.live.add(mid)
        self.grad_norms[mid] = self.by_seed[seed]
        return FakeClient(mid, self.grad_norms[mid])

    def optim_step(self, model_id, lr=0.0):
        return {"grad_norm": self.grad_norms[model_id]}

    def delete_model(self, model_id):
        self.events.append(("delete", model_id, None))
        self.live.discard(model_id)
        return True


class TestExecute:
    """Three models is one more than a 4-GPU pod can hold beside a live
    engine, so the arms must not overlap."""

    def _run(self, monkeypatch, by_seed):
        monkeypatch.setattr(
            "gates.traces.common.synthetic_lm_datums",
            lambda n, seq: [f"d{i}" for i in range(n)],
        )
        trace = SeedRepro(seed=7, contrast_seed=99)
        driver = FakeDriver(by_seed)
        return trace, driver, trace.execute(driver)

    def test_each_model_is_released_before_the_next_is_created(
        self, monkeypatch
    ):
        trace, driver, rows = self._run(monkeypatch, {7: 3141.8, 99: 2900.0})
        kinds = [e[0] for e in driver.events]
        assert kinds == ["create", "delete"] * 3
        assert driver.live == set() and trace.created_models == []
        assert [r["seed"] for r in rows] == [7, 7, 99]
        assert [r["grad_norm"] for r in rows] == [3141.8, 3141.8, 2900.0]

    def test_a_failed_arm_still_releases_its_model(self, monkeypatch):
        monkeypatch.setattr(
            "gates.traces.common.synthetic_lm_datums",
            lambda n, seq: [f"d{i}" for i in range(n)],
        )
        trace = SeedRepro()
        driver = FakeDriver({7: 3141.8, 99: 2900.0})

        def boom(model_id, lr=0.0):
            raise RuntimeError("step exploded")

        driver.optim_step = boom
        with pytest.raises(RuntimeError, match="exploded"):
            trace.execute(driver)
        assert driver.live == set() and trace.created_models == []


class StubDriver:
    base_url = "stub://"


class StubTrace(SeedRepro):
    def __init__(self, rows, **kw):
        super().__init__(**kw)
        self._rows = rows

    def execute(self, driver):
        return [dict(r) for r in self._rows]


class TestRunner:
    def test_verdict_doc_matches_schema(self, tmp_path):
        doc = runner.run(
            invariants.get("SEED_REPRO"),
            StubTrace(rows_for([3141.8, 3141.8, 2900.0])),
            tag="stub",
            out_dir=str(tmp_path),
            expect="SEED_HONORED",
            driver=StubDriver(),
        )
        schema = json.loads(
            (Path(runner.__file__).parent / "verdicts" / "schema.json").read_text()
        )
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(doc, schema)
        assert doc["gate"] == "SEED_REPRO.seed_repro"
        assert doc["passed"] is True and doc["expect_met"] is True
        assert (tmp_path / "seed_repro.seed_repro_stub.json").exists()

    def test_registered_on_the_cli(self):
        assert "seed_repro" in runner.TRACES
