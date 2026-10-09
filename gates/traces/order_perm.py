"""PARTITION_INV order-permutation probe, consolidated from
specs/014-gate-suite/probes/order_perm_probe.py. Datum generator, arm
naming and verdict semantics are preserved, so the 13 recorded
`order_perm_*.json` runs replay against this logic (see
tests/test_gates_order_perm.py).

Strictly stronger than SPLIT_INV and it needs no segmentation at all:
ONE call, all n datums, only the submission order changes. miles assigns
sample i of a call to rank `i % dp`, so SWAP_PAIRS ([1,0,3,2,...]) moves
every datum to the OTHER rank while changing nothing else -- same call,
same count, no pad, same weights, lr=0. A pure-sum gradient is
order-invariant in exact arithmetic (Thm 1), so any deviation beyond the
fp floor is a defect.

The three-way verdict keys on the fact that SWAP_PAIRS and REVERSE induce
the SAME rank partition but different concatenation orders -- the two
failure modes a binary order-invariance verdict confounds (they coincided
on miles, which is why the confusion went unnoticed):
  INVARIANT                            every permuted arm bit-identical
  PARTITION_SENSITIVE                  SWAP_PAIRS != IDENT and == REVERSE:
                                       the gradient is a function of which
                                       datums share a rank (the miles
                                       signature, pre-`39add0a`)
  PARTITION_INVARIANT_ORDER_SENSITIVE  SWAP_PAIRS bit-identical, something
                                       else moves: fp addition is
                                       commutative but not associative
                                       (the buffered/nemo_rl signature)
  ORDER_SENSITIVE_UNCLASSIFIED         neither known signature
  INCONCLUSIVE                         determinism control failed

DP WIDTH IS NOT AN ARM AND CANNOT BE. Width is fixed by the server's
NUM_GPUS at launch, so one gate process cannot sweep it: invoke the runner
once per server configuration and compare the verdicts afterwards. On a
buffered backend the width comparison is not certifiable through the API
at all -- LoRA B is zero-initialized, so no step-0 observable can see A.

WHAT THE ARMS MEAN IS ITSELF WIDTH-DEPENDENT, and the trace is blind to
the width. Under `rank = position % dp`, the grouping an arm induces is
the partition of datum indices by position residue, so at n=8:
  dp=2  SWAP_PAIRS moves every datum to the other rank -- a real grouping
        change, which is the case the three-way verdict was validated on
  dp=4  every default arm preserves the grouping and changes only which
        rank holds which group (and the order within it), so a residual
        there is a reduction-order effect, not partition sensitivity
Hence a run whose arms disagree lands in ORDER_SENSITIVE_UNCLASSIFIED
rather than being forced into a signature: at dp=4 the gate genuinely
cannot tell the two apart, and saying so is the honest verdict. Seen
live -- miles dp=4 at the engine default seed puts SWAP_PAIRS 6.4e-08
off IDENT while REVERSE is bit-identical, reproducibly.
"""

import time
from dataclasses import dataclass, field

from gates.comparators import Comparison, rel_diff

# The determinism control; every other arm is a genuine permutation.
CONTROL_ARM = "IDENT_R"
# The partition arm: at dp>=2 it sends every datum to a different rank.
PARTITION_ARM = "SWAP_PAIRS"
# Same partition as PARTITION_ARM, different concatenation order.
ORDER_ARM = "REVERSE"


def default_arms(n: int) -> list[tuple[str, list[int]]]:
    """(name, submission order). SWAP_PAIRS flips every datum's rank at
    dp=2; REVERSE induces the same partition by a coarser route; the _R
    arms are the reproducibility and determinism controls."""
    if n % 2:
        raise ValueError(f"n must be even for {PARTITION_ARM}, got {n}")
    ident = list(range(n))
    swap = [i ^ 1 for i in range(n)]
    return [
        ("IDENT", ident),
        (PARTITION_ARM, swap),
        (ORDER_ARM, list(reversed(ident))),
        (PARTITION_ARM + "_R", list(swap)),
        (CONTROL_ARM, list(ident)),
    ]


OUTCOME_VERDICTS = {
    "INCONCLUSIVE": "INCONCLUSIVE: the repeated IDENT arm is not reproducible",
    "NO_DETERMINISM_CONTROL": (
        "INCONCLUSIVE: the sweep carries no repeated IDENT arm, so a moved "
        "gradient cannot be told from run-to-run noise"
    ),
    "NO_PARTITION_ARM": (
        "INCONCLUSIVE: the sweep contains no rank-flipping permutation, so it "
        "cannot ask whether the gradient depends on the partition"
    ),
    "INVARIANT": (
        "INVARIANT: permuting the round leaves the gradient bit-identical "
        "under every arm -- neither the rank partition nor the concatenation "
        "order reaches it"
    ),
    "PARTITION_SENSITIVE": (
        "PARTITION-SENSITIVE: two different permutations inducing the SAME "
        "rank partition agree bit-for-bit while differing from the reference "
        "by rel={partition_rel:.3e} -- the gradient is a function of which "
        "datums share a rank, and no segmentation is needed to reach it"
    ),
    "PARTITION_INVARIANT_ORDER_SENSITIVE": (
        "PARTITION-INVARIANT, order-sensitive at rel={max_rel:.3e}: the "
        "rank-flipping permutation is bit-identical, so the partition does "
        "not reach the gradient; what moves is the concatenation order, and "
        "floating-point addition is commutative but not associative -- a "
        "within-pair swap preserves the reduction tree, a reversal regroups it"
    ),
    "ORDER_SENSITIVE_UNCLASSIFIED": (
        "ORDER-SENSITIVE (unclassified): max rel={max_rel:.3e}; the "
        "partition/order arms do not match either known signature"
    ),
}

# Outcomes that report a deviation rather than a broken invariant: the
# partition does not reach the gradient, which is what PARTITION_INV
# claims. The residual order-sensitivity is carried in the detail fields.
PASSING_OUTCOMES = ("INVARIANT", "PARTITION_INVARIANT_ORDER_SENSITIVE")
UNDECIDED_OUTCOMES = (
    "INCONCLUSIVE",
    "NO_DETERMINISM_CONTROL",
    "NO_PARTITION_ARM",
    "ORDER_SENSITIVE_UNCLASSIFIED",
)


@dataclass
class OrderPerm:
    model: str = "Qwen/Qwen2.5-0.5B"
    rank: int = 32
    seq: int = 64
    n: int = 8
    seed: int | None = None
    lr: float = 0.0
    arms: list[tuple[str, list[int]]] = field(default_factory=lambda: default_arms(8))
    name: str = "order_perm"
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
            "arms": [{"arm": a, "order": o} for a, o in self.arms],
        }

    def execute(self, driver) -> list[dict]:
        """One observation row per arm: {arm, order, grad_norm, repr, s}."""
        from gates.traces.common import synthetic_lm_datums

        ds = synthetic_lm_datums(self.n, self.seq)
        tc = driver.create_training_client(
            base_model=self.model,
            rank=self.rank,
            seed=self.seed,
        )
        self.created_models.append(tc.model_id)
        rows = []
        for arm, order in self.arms:
            t0 = time.time()
            # ONE call: no segmentation, no pad -- only the order changes
            tc.forward_backward([ds[i] for i in order], "cross_entropy").result()
            res = driver.optim_step(tc.model_id, lr=self.lr)
            gn = res.get("grad_norm")
            rows.append(
                {
                    "arm": arm,
                    "order": list(order),
                    "grad_norm": gn,
                    "repr": repr(gn),
                    "s": round(time.time() - t0, 1),
                }
            )
            print(
                f"  {arm:13s} grad_norm={gn!r} ({time.time() - t0:.1f}s)",
                flush=True,
            )
        return rows

    def verdict(self, rows: list[dict], comparator) -> dict:
        return verdict(rows, comparator)


def verdict(rows: list[dict], comparator) -> dict:
    """Pure function of the observation rows; mutates rows in place with
    per-arm bit_identical/rel_diff exactly as order_perm_probe.py did."""
    ref = rows[0]["grad_norm"]
    for r in rows[1:]:
        cmp: Comparison = comparator(ref, r["grad_norm"])
        r["bit_identical"] = cmp.passed
        r["rel_diff"] = rel_diff(ref, r["grad_norm"]) if ref else float("nan")

    by = {r["arm"]: r for r in rows[1:]}
    control, swap, rev = by.get(CONTROL_ARM), by.get(PARTITION_ARM), by.get(ORDER_ARM)
    # the arms that actually permute something; the control is not one
    perms = [r for r in rows[1:] if r["arm"] != CONTROL_ARM]

    det = control["bit_identical"] if control else None
    max_rel = max((r["rel_diff"] for r in perms), default=0.0)
    partition_rel = swap["rel_diff"] if swap else None
    all_perm_bit = all(r["bit_identical"] for r in perms) if perms else False
    # A repeated permutation that reproduces its own value pins the IDENT
    # difference on the permutation, not on run-to-run noise.
    repeat = by.get(PARTITION_ARM + "_R")
    partition_reproducible = (
        comparator(swap["grad_norm"], repeat["grad_norm"]).passed
        if swap and repeat
        else None
    )
    # SWAP_PAIRS and REVERSE induce the SAME partition: agreeing bit-for-bit
    # while both differing from IDENT is the partition-sensitive signature.
    same_partition_agree = (
        comparator(swap["grad_norm"], rev["grad_norm"]).passed
        if swap and rev
        else None
    )

    if control is None:
        outcome = "NO_DETERMINISM_CONTROL"
    elif not det:
        outcome = "INCONCLUSIVE"
    elif swap is None:
        outcome = "NO_PARTITION_ARM"
    elif all_perm_bit:
        outcome = "INVARIANT"
    elif not swap["bit_identical"] and same_partition_agree:
        outcome = "PARTITION_SENSITIVE"
    elif swap["bit_identical"]:
        outcome = "PARTITION_INVARIANT_ORDER_SENSITIVE"
    else:
        outcome = "ORDER_SENSITIVE_UNCLASSIFIED"

    return {
        "outcome": outcome,
        "verdict": OUTCOME_VERDICTS[outcome].format(
            partition_rel=partition_rel or 0.0, max_rel=max_rel
        ),
        # PARTITION_INV asks whether the partition reaches the gradient, so a
        # bit-identical rank-flip passes even when the concatenation order
        # still moves the last ulp; None = the sweep could not decide.
        "passed": (
            None
            if outcome in UNDECIDED_OUTCOMES
            else outcome in PASSING_OUTCOMES
        ),
        # the invariant's own quantity: how far the rank-flip moved the gradient
        "delta": partition_rel,
        "determinism_ok": det,
        "partition_bit_identical": swap["bit_identical"] if swap else None,
        "partition_reproducible": partition_reproducible,
        "same_partition_arms_agree": same_partition_agree,
        "permutation_invariant": all_perm_bit,
        "partition_rel_diff": partition_rel,
        "max_rel_diff": max_rel,
    }
