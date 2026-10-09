"""SPLIT_INV segmentation sweep (G1b), consolidated from
specs/008-q3-abstraction-tax/probes/g1b_segmentation.py and
specs/013-a1-8b-spine/probes/g1b_8b.py — 0.5B and 8B are two parameter
points of this one trace. Datum generator, arm naming, and verdict
semantics are preserved token-for-token; recorded results JSONs remain
comparable.

All arms run at lr=0.0 so every arm sees identical weights. Outcomes:
  E0_ARBITRARY    every segmentation bit-identical (buffered prediction)
  E0_ALIGNED_ONLY aligned arm bit-identical, non-aligned differ at
                  ~fp-noise (eager prediction: Thm 1 holds, Prop 2 fails)
  E0_BROKEN       even the aligned arm differs
  INCONCLUSIVE    determinism control failed (A2 != A1)
"""

import time
from dataclasses import dataclass, field

from gates.comparators import Comparison, rel_diff

# (name, segmentation of the n datums); verdict logic keys on A1/A2/ALIGNED/_R
DEFAULT_ARMS: list[tuple[str, list[int]]] = [
    ("A1", [8]),
    ("ALIGNED", [4, 4]),
    ("NONALIGN", [3, 5]),
    ("NONALIGN2", [1, 7]),
    ("NONALIGN_R", [3, 5]),  # rerun: is the non-aligned value itself stable?
    ("A2", [8]),
]

OUTCOME_VERDICTS = {
    "INCONCLUSIVE": "INCONCLUSIVE: backend is not run-to-run deterministic (A2 != A1)",
    "NO_NONALIGNED_ARM": (
        "INCONCLUSIVE: the sweep contains no non-aligned segmentation, so it "
        "cannot distinguish E0 under arbitrary splits from E0 only when aligned"
    ),
    "E0_ARBITRARY": (
        "E0 UNDER ARBITRARY SEGMENTATION: every arm bit-identical "
        "(consistent with a segmentation-independent reduction order)"
    ),
    "E0_ALIGNED_ONLY": (
        "E0 ONLY WHEN ALIGNED: aligned arm bit-identical, non-aligned "
        "arms differ (max rel {max_rel:.3e}) -- Thm 1 holds, Prop 2 fails"
    ),
    "E0_BROKEN": (
        "E0 FAILS EVEN WHEN ALIGNED -- split-invariance is broken, not just bitwise"
    ),
}


def parse_segmentations(spec: str, n: int) -> list[tuple[str, list[int]]]:
    """'8;4,4;2,6;6,2;2,6;8' -> named arms. First = reference A1, last =
    determinism control A2. Needed on backends that DEADLOCK on
    rank-imbalanced splits (miles at dp=2 derives a different
    num_steps_per_rollout per rank and hangs): balanced-but-unaligned
    splits like 2,6 still ask the bitwise question."""
    segs = [[int(x) for x in part.split(",")] for part in spec.split(";")]
    names, seen = [], {}
    for i, s in enumerate(segs):
        if i == 0:
            names.append("A1")
        elif i == len(segs) - 1:
            names.append("A2")
        elif s == [n // 2, n // 2]:
            names.append("ALIGNED")
        else:
            key = tuple(s)
            seen[key] = seen.get(key, 0) + 1
            names.append("S" + "+".join(map(str, s)) + ("_R" if seen[key] > 1 else ""))
    return list(zip(names, segs))


def segments(ds: list, sizes: list[int]) -> list[list]:
    out, i = [], 0
    for s in sizes:
        out.append(ds[i : i + s])
        i += s
    assert i == len(ds), f"segmentation {sizes} does not cover {len(ds)} datums"
    return out


@dataclass
class SegSweep:
    model: str = "Qwen/Qwen2.5-0.5B"
    rank: int = 32
    seq: int = 64
    n: int = 8
    seed: int | None = None
    lr: float = 0.0
    arms: list[tuple[str, list[int]]] = field(
        default_factory=lambda: list(DEFAULT_ARMS)
    )
    name: str = "seg_sweep"
    # models this trace created; the runner deletes them when the gate ends
    created_models: list[str] = field(default_factory=list)

    def params(self) -> dict:
        return {
            "model": self.model,
            "rank": self.rank,
            "seq": self.seq,
            "n_datums": self.n,
            "seed": self.seed,
            "lr": self.lr,
            "arms": [{"arm": a, "segmentation": s} for a, s in self.arms],
        }

    def execute(self, driver) -> list[dict]:
        """One observation row per arm: {arm, segmentation, grad_norm, repr, s}."""
        from gates.traces.common import synthetic_lm_datums

        ds = synthetic_lm_datums(self.n, self.seq)
        tc = driver.create_training_client(
            base_model=self.model,
            rank=self.rank,
            seed=self.seed,
        )
        self.created_models.append(tc.model_id)
        rows = []
        for arm, sizes in self.arms:
            t0 = time.time()
            for part in segments(ds, sizes):
                tc.forward_backward(part, "cross_entropy").result()
            res = driver.optim_step(tc.model_id, lr=self.lr)
            gn = res.get("grad_norm")
            rows.append(
                {
                    "arm": arm,
                    "segmentation": sizes,
                    "grad_norm": gn,
                    "repr": repr(gn),
                    "s": round(time.time() - t0, 1),
                }
            )
            print(
                f"  {arm:10s} {str(sizes):8s} grad_norm={gn!r} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        return rows

    def verdict(self, rows: list[dict], comparator) -> dict:
        return verdict(rows, comparator)


def verdict(rows: list[dict], comparator) -> dict:
    """Pure function of the observation rows; mutates rows in place with
    per-arm bit_identical/rel_diff exactly as the 008/013 probes did."""
    ref = rows[0]["grad_norm"]
    for r in rows[1:]:
        cmp: Comparison = comparator(ref, r["grad_norm"])
        r["bit_identical"] = cmp.passed
        r["rel_diff"] = (
            rel_diff(ref, r["grad_norm"]) if ref else float("nan")
        )

    det = next(r for r in rows if r["arm"] == "A2")["bit_identical"]
    al = [r for r in rows if r["arm"] == "ALIGNED"]
    aligned = al[0]["bit_identical"] if al else None
    # "non-aligned" = every middle arm other than the aligned one
    nonalign = [r for r in rows[1:-1] if r["arm"] != "ALIGNED"]
    # A repeated segmentation that reproduces its own value pins the A1
    # difference on the segmentation, not run-to-run noise.
    nonalign_reproducible = None
    for r in nonalign:
        if r["arm"].endswith("_R"):
            base = next((q for q in nonalign if q["arm"] == r["arm"][:-2]), None)
            if base:
                nonalign_reproducible = comparator(
                    base["grad_norm"], r["grad_norm"]
                ).passed
    all_nonalign_bit = all(r["bit_identical"] for r in nonalign)
    max_rel = max((r["rel_diff"] for r in nonalign), default=0.0)

    if not det:
        outcome = "INCONCLUSIVE"
    elif not nonalign:
        # Aligned arms alone cannot separate "invariant under every
        # segmentation" from "invariant only where the split happens to
        # match the engine's batching" — the sweep asked nothing.
        outcome = "NO_NONALIGNED_ARM"
    elif all_nonalign_bit:
        outcome = "E0_ARBITRARY"
    elif aligned:
        outcome = "E0_ALIGNED_ONLY"
    else:
        outcome = "E0_BROKEN"

    return {
        "outcome": outcome,
        "verdict": OUTCOME_VERDICTS[outcome].format(max_rel=max_rel),
        # strict SPLIT_INV@E0; None = the sweep could not decide (determinism
        # control failed, or no non-aligned arm was submitted)
        "passed": (
            None
            if outcome in ("INCONCLUSIVE", "NO_NONALIGNED_ARM")
            else outcome == "E0_ARBITRARY"
        ),
        "delta": max_rel,
        "determinism_ok": det,
        "aligned_bit_identical": aligned,
        "nonaligned_reproducible": nonalign_reproducible,
        "nonaligned_all_bit_identical": all_nonalign_bit,
        "max_nonaligned_rel_diff": max_rel,
    }
