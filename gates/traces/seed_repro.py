"""SEED_REPRO: is the client's `LoraConfig.seed` actually read?

Guards a silent failure. The seed was declared on the API, carried into
every backend's create_model, and read by NO training backend: miles was
reproducible *by accident* (Megatron's --seed defaults to 1234) while
ignoring the client's value, and nemo_rl's policy workers never seeded at
all. Fixed on both sides (tinker-cloud `995945e` PR #38, GavinZhu-GMI/RL
`faba0bfc3`), but nothing fails loudly the day an image is rebuilt without
the engine half -- hence this gate.

Three arms, each a fresh model given ONE identical call at lr=0:
  SEED_A, SEED_A_R   same seed twice
  SEED_B             a different seed

The contrast arm is what separates the two failure modes, and it is why
"two models, one seed" is not enough on its own: an engine that ignores the
client's seed and falls back to a fixed internal default reproduces
perfectly across SEED_A/SEED_A_R and looks healthy.

  SEED_HONORED     same seed agrees bitwise, different seed disagrees
  SEED_IGNORED     both pairs agree -- the seed value reaches nothing
  NOT_REPRODUCIBLE same seed disagrees -- nothing is seeded (or the engine
                   is not run-to-run deterministic)

grad_norm is the right observable even though LoRA B is zero-initialized:
B=0 makes the step-0 *forward* seed-independent, but dL/dB is a function of
A, so the gradient sees the seed. Do not substitute a forward observable.
"""

import time
from dataclasses import dataclass, field

from gates.comparators import Comparison, rel_diff

DEFAULT_SEED = 7
DEFAULT_CONTRAST_SEED = 99

REPEAT_ARM = "SEED_A_R"
CONTRAST_ARM = "SEED_B"

OUTCOME_VERDICTS = {
    "NO_REPEAT_ARM": (
        "INCONCLUSIVE: the sweep creates no second model at the same seed, so "
        "it cannot ask whether the seed reproduces"
    ),
    "NOT_REPRODUCIBLE": (
        "SEED NOT REPRODUCIBLE: two models created with the same seed disagree "
        "on grad_norm by rel={repeat_rel:.3e} -- the seed reaches no RNG, so "
        "nothing about a run at this seed is reproducible"
    ),
    "NO_CONTRAST_ARM": (
        "INCONCLUSIVE: the same seed reproduces, but with no different-seed "
        "arm this cannot be told from an engine that ignores the client's seed "
        "and falls back to a fixed internal default"
    ),
    "SEED_IGNORED": (
        "SEED IGNORED: two DIFFERENT seeds produce a bit-identical gradient -- "
        "the client's value reaches no RNG and the reproducibility above is an "
        "accident of the engine's own default seed"
    ),
    "SEED_HONORED": (
        "SEED HONORED: the same seed reproduces bit-identically and a "
        "different seed moves the gradient (rel={contrast_rel:.3e}) -- the "
        "client's seed is read"
    ),
}


def default_arms(
    seed: int = DEFAULT_SEED, contrast_seed: int = DEFAULT_CONTRAST_SEED
) -> list[tuple[str, int]]:
    return [("SEED_A", seed), (REPEAT_ARM, seed), (CONTRAST_ARM, contrast_seed)]


@dataclass
class SeedRepro:
    model: str = "Qwen/Qwen2.5-0.5B"
    rank: int = 32
    seq: int = 64
    n: int = 8
    seed: int = DEFAULT_SEED
    contrast_seed: int = DEFAULT_CONTRAST_SEED
    max_seq_len: int | None = None
    debug_train_only: bool = False
    lr: float = 0.0
    arms: list[tuple[str, int]] | None = None
    name: str = "seed_repro"
    # models this trace created; the runner deletes any the trace still holds
    created_models: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.arms is None:
            self.arms = default_arms(self.seed, self.contrast_seed)

    def params(self) -> dict:
        return {
            "model": self.model,
            "rank": self.rank,
            "seq": self.seq,
            "n_datums": self.n,
            "seed": self.seed,
            "contrast_seed": self.contrast_seed,
            "max_seq_len": self.max_seq_len,
            "debug_train_only": self.debug_train_only,
            "lr": self.lr,
            "arms": [{"arm": a, "seed": s} for a, s in self.arms],
        }

    def execute(self, driver) -> list[dict]:
        """One observation row per arm: {arm, seed, grad_norm, repr, s}.

        Each arm's model is released as soon as its observation is recorded:
        the arms are independent, and holding three models at once would ask
        for GPUs a 4-GPU pod does not have."""
        from gates.traces.common import synthetic_lm_datums

        ds = synthetic_lm_datums(self.n, self.seq)
        rows = []
        for arm, seed in self.arms:
            t0 = time.time()
            tc = driver.create_training_client(
                base_model=self.model,
                rank=self.rank,
                seed=seed,
                max_seq_len=self.max_seq_len,
                debug_train_only=self.debug_train_only,
            )
            self.created_models.append(tc.model_id)
            try:
                tc.forward_backward(ds, "cross_entropy").result()
                res = driver.optim_step(tc.model_id, lr=self.lr)
            finally:
                if driver.delete_model(tc.model_id):
                    self.created_models.remove(tc.model_id)
            gn = res.get("grad_norm")
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "grad_norm": gn,
                    "repr": repr(gn),
                    "s": round(time.time() - t0, 1),
                }
            )
            print(
                f"  {arm:10s} seed={seed:<6d} grad_norm={gn!r} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
        return rows

    def verdict(self, rows: list[dict], comparator) -> dict:
        return verdict(rows, comparator)


def verdict(rows: list[dict], comparator) -> dict:
    """Pure function of the observation rows; mutates rows in place with
    per-arm bit_identical/rel_diff against the first arm."""
    ref = rows[0]["grad_norm"]
    for r in rows[1:]:
        cmp: Comparison = comparator(ref, r["grad_norm"])
        r["bit_identical"] = cmp.passed
        r["rel_diff"] = rel_diff(ref, r["grad_norm"]) if ref else float("nan")

    by = {r["arm"]: r for r in rows[1:]}
    repeat, contrast = by.get(REPEAT_ARM), by.get(CONTRAST_ARM)
    reproducible = repeat["bit_identical"] if repeat else None
    contrast_differs = (not contrast["bit_identical"]) if contrast else None
    repeat_rel = repeat["rel_diff"] if repeat else None
    contrast_rel = contrast["rel_diff"] if contrast else None

    if repeat is None:
        outcome = "NO_REPEAT_ARM"
    elif not reproducible:
        outcome = "NOT_REPRODUCIBLE"
    elif contrast is None:
        outcome = "NO_CONTRAST_ARM"
    elif not contrast_differs:
        outcome = "SEED_IGNORED"
    else:
        outcome = "SEED_HONORED"

    return {
        "outcome": outcome,
        "verdict": OUTCOME_VERDICTS[outcome].format(
            repeat_rel=repeat_rel or 0.0, contrast_rel=contrast_rel or 0.0
        ),
        "passed": (
            None
            if outcome in ("NO_REPEAT_ARM", "NO_CONTRAST_ARM")
            else outcome == "SEED_HONORED"
        ),
        # the invariant's own quantity: how far two same-seed models drifted
        "delta": repeat_rel,
        "same_seed_bit_identical": reproducible,
        "different_seed_differs": contrast_differs,
        "same_seed_rel_diff": repeat_rel,
        "different_seed_rel_diff": contrast_rel,
    }
