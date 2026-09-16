"""Runner: gate = invariant x trace x arm -> uniform verdict JSON
(gates/verdicts/schema.json). Replaces the per-spec probe scripts'
ad hoc verdict/result formats.

  from gates import invariants, runner
  from gates.traces import SegSweep
  runner.run(invariants.get("SPLIT_INV"), SegSweep(model="Qwen/Qwen3-8B-Base",
             max_seq_len=8192, debug_train_only=True), tag="nemo_rl_tp2")

CLI: python -m gates.runner seg_sweep --tag nemo_rl [--model ...] --out results

DP width is not an arm anywhere in this suite: it is fixed by the server's
NUM_GPUS at launch, so a width comparison means invoking the runner once per
server configuration and diffing the verdict JSONs.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from gates import GATES_VERSION, comparators, invariants
from gates.drivers import SDKDriver
from gates.invariants import Invariant


def _sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=repr).encode()
    ).hexdigest()


def _release_models(trace, driver) -> None:
    """Delete every model the trace created. Best-effort and never raises —
    a failed cleanup must not turn a real verdict into an exception."""
    models = trace.created_models
    if not models:
        return
    for model_id in models:
        try:
            driver.delete_model(model_id)
        except Exception as e:  # noqa: BLE001 - a leaked model must not eat a verdict
            print(f"WARNING: releasing {model_id} raised {e!r}", flush=True)
    models.clear()


def run(
    invariant: Invariant,
    trace,
    tag: str,
    out_dir: str | None = None,
    expect: str | None = None,
    driver=None,
) -> dict:
    driver = driver or SDKDriver()
    comparator = comparators.REGISTRY[invariant.comparator]

    print(f"[{tag}] gate {invariant.id}.{trace.name} ({invariant.level})", flush=True)
    try:
        rows = trace.execute(driver)
    finally:
        # A gate owns the models it creates. miles does not auto-reap them, so
        # skipping this wedges the next gate's create_model in placement.
        _release_models(trace, driver)
    v = trace.verdict(rows, comparator)

    ref = rows[0]
    print(f"\n[{tag}] vs {ref['arm']} = {ref['grad_norm']!r}"
          if "grad_norm" in ref else f"\n[{tag}]")
    for r in rows[1:]:
        if "bit_identical" in r:
            line = ("BIT-IDENTICAL" if r["bit_identical"]
                    else f"differs, rel={r['rel_diff']:.3e}")
            print(f"  {r['arm']:10s} {str(r.get('segmentation', '')):8s} {line}")

    verdict_doc = {
        "gate": f"{invariant.id}.{trace.name}",
        "invariant": invariant.id,
        "claimed_level": invariant.level,
        "arm": {"tag": tag, "expect": expect},
        "trace": {"name": trace.name, "params": trace.params()},
        "outcome": v["outcome"],
        "verdict": v["verdict"],
        "passed": v["passed"],
        "expect_met": (v["outcome"] == expect) if expect else None,
        "delta": v.get("delta"),
        "threshold": v.get("threshold"),
        "observations": rows,
        "detail": {
            k: val
            for k, val in v.items()
            if k not in ("outcome", "verdict", "passed", "delta", "threshold")
        },
        "artifacts": {
            "observations_sha256": _sha256(rows),
            "trace_sha256": _sha256(trace.params()),
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": driver.base_url,
            "gates_version": GATES_VERSION,
        },
    }

    print(f"\n[{tag}] {v['verdict']}")
    if expect:
        print(f"[{tag}] expected {expect}: "
              f"{'MET' if verdict_doc['expect_met'] else 'NOT MET'}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f"{invariant.id}.{trace.name}_{tag}.json".lower()
        )
        with open(path, "w") as f:
            json.dump(verdict_doc, f, indent=2)
        print(f"wrote {path}")
    return verdict_doc


def _build_seg_sweep(args) -> tuple[Invariant, object]:
    from gates.traces.seg_sweep import SegSweep, parse_segmentations

    trace = SegSweep(
        model=args.model,
        rank=args.rank,
        seq=args.seq,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        debug_train_only=args.debug_train_only,
    )
    if args.segmentations:
        trace.arms = parse_segmentations(args.segmentations, trace.n)
    return invariants.get("SPLIT_INV"), trace


def _build_order_perm(args) -> tuple[Invariant, object]:
    from gates.traces.order_perm import OrderPerm, default_arms

    trace = OrderPerm(
        model=args.model,
        rank=args.rank,
        seq=args.seq,
        n=args.n,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        debug_train_only=args.debug_train_only,
    )
    trace.arms = default_arms(trace.n)
    return invariants.get("PARTITION_INV"), trace


def _build_seed_repro(args) -> tuple[Invariant, object]:
    from gates.traces.seed_repro import DEFAULT_SEED, SeedRepro

    return invariants.get("SEED_REPRO"), SeedRepro(
        model=args.model,
        rank=args.rank,
        seq=args.seq,
        n=args.n,
        seed=args.seed if args.seed is not None else DEFAULT_SEED,
        contrast_seed=args.contrast_seed,
        max_seq_len=args.max_seq_len,
        debug_train_only=args.debug_train_only,
    )


TRACES = {
    "order_perm": _build_order_perm,
    "seed_repro": _build_seed_repro,
    "seg_sweep": _build_seg_sweep,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", choices=sorted(TRACES))
    ap.add_argument("--tag", required=True,
                    help="arm label: nemo_rl | nemo_rl_tp2 | verl | miles | ...")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--n", type=int, default=8,
                    help="datums per round (order_perm needs it even)")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix LoRA init so runs at different dp are comparable")
    ap.add_argument("--contrast-seed", type=int, default=99,
                    help="seed_repro: the different-seed arm")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--debug-train-only", action="store_true",
                    help="no inference engine (8B sweeps run this way)")
    ap.add_argument("--segmentations", default=None,
                    help="override arms, e.g. '8;4,4;2,6;6,2;2,6;8' "
                         "(first=reference, last=determinism control)")
    ap.add_argument("--expect", default=None,
                    help="grade the outcome against a prediction, "
                         "e.g. E0_ARBITRARY (buffered) / E0_ALIGNED_ONLY (eager)")
    args = ap.parse_args()

    inv, trace = TRACES[args.trace](args)
    doc = run(inv, trace, tag=args.tag, out_dir=args.out, expect=args.expect)
    raise SystemExit(0 if doc["passed"] or doc["expect_met"] else 1)


if __name__ == "__main__":
    main()
