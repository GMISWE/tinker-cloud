# Gate Suite

The checking layer for the TinkerCloud API as a shippable artifact.
Design, phasing (P1/P2/P3), and SOTA positioning:
`specs/014-gate-suite/design.md` in the monorepo.

**Status: P1 merged** (#39) — registry, trace/driver/oracle/comparator
split, runner + verdict schema, and three gates. Remaining P1 migrations
and P2/P3 are in the map below.

A gate is one composition — **invariant × trace × arm → verdict** —
instead of a 300-line script:

```python
from gates import invariants, runner
from gates.traces import SegSweep

runner.run(
    invariants.get("SPLIT_INV"),
    SegSweep(model="Qwen/Qwen3-8B-Base"),
    tag="nemo_rl_tp2",
    out_dir="results",
)
```

0.5B and 8B are two parameter points of one gate, not two files.

```bash
# on the server pod (TINKER_BASE_URL=http://localhost:8000)
python -m gates.runner seg_sweep --tag nemo_rl --expect E0_ARBITRARY
python -m gates.runner seg_sweep --tag miles --segmentations '8;4,4;2,6;6,2;2,6;8' \
    --expect E0_ALIGNED_ONLY   # miles deadlocks on rank-imbalanced splits at dp>1
python -m gates.runner seg_sweep --tag nemo_rl_tp2 --model Qwen/Qwen3-8B-Base \
    --expect E0_ARBITRARY

# PARTITION_INV: one call, only the submission order changes (no inference engine: boot the server with NEMORL_DEBUG_TRAIN_ONLY=1
# is required on miles below NUM_GPUS=4 -- start_engines dies on
# reordered_gpu_ids[gpu_index] at NUM_GPUS=2)
python -m gates.runner order_perm --tag miles_dp4 --seed 7 \
    --expect INVARIANT
python -m gates.runner order_perm --tag nemo_rl_dp2 --seed 7 \
    --expect PARTITION_INVARIANT_ORDER_SENSITIVE   # dp=4 is INVARIANT

# SEED_REPRO: does the backend read LoraConfig.seed at all
python -m gates.runner seed_repro --tag nemo_rl_dp4 \
    --expect SEED_HONORED
```

**DP width is not an arm anywhere in this suite.** Width is fixed by the
server's `NUM_GPUS` at launch, so one gate process cannot sweep it: run the
gate once per server configuration and diff the verdict JSONs. On a
buffered backend the width comparison is not certifiable through the API at
all — LoRA `B` is zero-initialized, so no step-0 observable can see `A`.

Exit code 0 iff `passed` (strict invariant at its claimed level) or, when
`--expect` is given, the outcome matches the prediction. Verdict JSONs
conform to `verdicts/schema.json`.

## Invariants

`invariants.py` is the registry. Each entry carries a statement, the
**claimed** equivalence level, the comparator that discharges it, where its
tolerance comes from, and the label it answers to in the manuscript.

| id | level | asks |
|---|---|---|
| `SPLIT_INV` | E0 | is a round's result independent of how the client segmented it |
| `PARTITION_INV` | E0 | is the gradient independent of which datums share a DP rank |
| `SEED_REPRO` | E0 | is the client's seed read, or defaulted behind its back |
| `DEFER_OBS` | E0 | is deferring `fb` to `optim_step` the identity on observables |
| `ISOLATION` | E0 | do co-located tenants observe what they would serialized |
| `PIN_DRIFT` | E0 | does a pinned sampler stay bit-stable as training moves |
| `ADAPTER_ORACLE` | E1 | do engine logprobs match an offline fp32 HF(+PEFT) oracle |
| `ENDPOINT_NLL` | E2 | does endpoint NLL land inside the measured rerun envelope |

`PARTITION_INV` is **strictly stronger than `SPLIT_INV`**: the DP-reduction
defect (`39add0a`) was invisible at `SPLIT_INV`'s aligned arm yet caught by
a single permuted call.

## Outcomes

Each trace owns a vocabulary; the runner records it verbatim and
`--expect` grades against it.

| trace | outcomes |
|---|---|
| `seg_sweep` | `E0_ARBITRARY` · `E0_ALIGNED_ONLY` · `E0_BROKEN` · `NO_NONALIGNED_ARM`⁰ · `INCONCLUSIVE`⁰ |
| `order_perm` | `INVARIANT` · `PARTITION_INVARIANT_ORDER_SENSITIVE` · `PARTITION_SENSITIVE` · `ORDER_SENSITIVE_UNCLASSIFIED`⁰ · `NO_PARTITION_ARM`⁰ · `NO_DETERMINISM_CONTROL`⁰ · `INCONCLUSIVE`⁰ |
| `seed_repro` | `SEED_HONORED` · `SEED_IGNORED` · `NOT_REPRODUCIBLE` · `NO_CONTRAST_ARM`⁰ · `NO_REPEAT_ARM`⁰ |

⁰ = `passed: null`. **"Could not decide" is a first-class outcome**, distinct
from False: a failed determinism control, an arm set that never asked the
question, or — for `order_perm` — arms that match neither known signature.
A gate that cannot answer must say so rather than guess.

Two outcomes are worth reading twice:

- `PARTITION_INVARIANT_ORDER_SENSITIVE` **passes**. The rank-flipping
  permutation is bit-identical, so the partition does not reach the
  gradient, which is what `PARTITION_INV` claims; the residual is
  concatenation order (fp addition is commutative but not associative) and
  is reported in `detail`, not treated as a failure.
- `SEED_IGNORED` needs the third arm to exist at all. An engine that
  ignores the client's seed and falls back to a fixed internal default
  reproduces perfectly across a same-seed pair — so `seed_repro` runs
  `SEED_A`/`SEED_A_R` at one seed **and `SEED_B` at another**.

## Contracts the runner enforces

- **A gate deletes the models it creates.** miles never reaps them, and one
  leftover wedges the next gate's `create_model` in Ray placement (four
  runs were lost to this before it was fixed). Cleanup is best-effort and
  never masks a verdict — losing a result that cost a cluster run is worse
  than leaking a model. `seed_repro` goes further and releases each arm's
  model *before* creating the next: three live models do not fit a 4-GPU
  pod.
- **A trace's verdict logic is a module-level pure function**, so recorded
  JSONs replay offline. SDK/torch imports stay inside `execute()`; the
  verdict layer loads without a GPU stack.
- **Recorded results stay comparable.** Migrated gates preserve datum
  generators, arm naming and verdict semantics token-for-token. The datum
  generator in `traces/common.py` is byte-identical to the probes' — do not
  "improve" it.

## What the suite has caught

Live runs and full numbers: `specs/014-gate-suite/RESULTS-P1-SMOKE.md` and
`RESULTS-P1-PARTITION-SEED.md` (monorepo).

- **A split-invariance failure on miles** contradicting this repo's own
  recorded artifacts — first live run of the migrated `seg_sweep`. Two
  engine defects behind it, both since fixed (`e37b8a2`, `39add0a`).
- **A merged-but-not-deployed backend.** `seed_repro`'s first run found the
  miles pod serving pre-`995945e` code: seeds 7 and 99 gave a bit-identical
  gradient. The pod was perfectly reproducible the whole time on Megatron's
  own default of 1234 — a same-seed pair alone would have passed.
- **An init-dependent residual.** `PARTITION_INV` is bitwise on both
  backends at dp=4 at seed 7, but at the engine default init `SWAP_PAIRS`
  sits 6.405e-08 off `IDENT`, reproducibly. Post-fix partition invariance
  is bitwise at the inits measured, not structurally.

## Layout

| Path | Role |
|---|---|
| `invariants.py` | registry: the paper's §3 contract as first-class objects |
| `traces/` | canonical trace generators (the admission clause made executable) |
| `drivers/` | run a trace through the public SDK/HTTP surface only |
| `oracles/` | fp32 HF(+PEFT) oracle, rerun-envelope estimator (P1: pointers) |
| `comparators.py` | E0 bitwise / E1 calibrated bar / E2 measured envelope |
| `runner.py` | composition + uniform verdict JSON + CLI |
| `verdicts/schema.json` | verdict schema (history in, certificate out) |
| `calibration.json` | (P3) the two measured populations per scale |
| `fixtures/` | (P3) frozen streams, content-addressed |

## Migration map

Standing gates and their source probes. Migrated gates preserve datum
generators, arm naming, and verdict semantics token-for-token, so recorded
results JSONs under `specs/*/probes/results/` remain comparable.

| Gate | Invariant | Source | Status |
|---|---|---|---|
| G1b segmentation sweep (0.5B + 8B) | SPLIT_INV | `specs/008 probes/g1b_segmentation.py`, `specs/013 probes/g1b_8b.py` | **migrated** → `traces/seg_sweep.py` |
| Order permutation (partition sensitivity) | PARTITION_INV | `specs/014 probes/order_perm_probe.py` | **migrated** → `traces/order_perm.py` |
| Seed reproducibility | SEED_REPRO | (none — found by hand, tinker-cloud `995945e` + `GavinZhu-GMI/RL` `faba0bfc3`) | **new** → `traces/seed_repro.py` |
| G1 grad-accum split | SPLIT_INV | `scripts/gates/g1_seam_parity.py` | pending (subsumed by seg_sweep default arms) |
| G1c admission sweep | SPLIT_INV (admission clause) | `specs/008 probes/g1c_admission_sweep.py` | pending |
| G2 client-order / pipelined | DEFER_OBS | `scripts/gates/g2_client_order.py`, `g2_pipelined.py` | pending |
| G3 30-step SFT e2e | ENDPOINT_NLL | `scripts/gates/g3_sl_basic.py` | pending |
| G4 pool isolation | ISOLATION | `scripts/gates/g4_pool_isolation.py` | pending |
| G5 pinned-v0 + routing | PIN_DRIFT | `scripts/gates/g5_pinned_v0.py` | pending |
| G6 batch-invariance | ISOLATION | `scripts/gates/g6_batch_invariance.py` | pending |
| G7 reorder-drain | ISOLATION | `scripts/gates/g7_reorder_merge.py` | pending |
| G8/G8e sampling + overlap | DEFER_OBS | `scripts/gates/g8_verl_sampling.py`, `g8e_overlap_check.py` | pending |
| G9 adapter interchange | ADAPTER_ORACLE | `scripts/gates/g9_adapter_interchange.py` | pending |
| G10 staleness-k | PIN_DRIFT (ver(S)) | `scripts/gates/g10_staleness.py` | pending |
| logprob HF parity | ADAPTER_ORACLE | `scripts/gates/g_lp_hf_parity.py` | pending |

One-off attribution probes (E2/RESID arms, `q4_rl_tenant.py`, analyze
scripts) stay in their spec directories as historical record, per
design.md §6 Q2.

## CI tiers (P3)

Smoke (0.5B, minutes) on every PR touching translation/backend code; full
parity nightly; 8B weekly/manual; mandatory admission run before any
engine-submodule bump.
