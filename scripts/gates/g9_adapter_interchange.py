#!/usr/bin/env python3
"""G9: cross-backend LoRA adapter interchange gates (Q5 migration seams).

Two modes, both against a live TinkerCloud server plus a local fp32 HF oracle:

  export  train N steps, snapshot the observable on a frozen probe batch,
          save_state, then load the PUBLISHED hf_adapter/ into plain
          HF+peft (fp32) and compare logprobs.
          -> **G9a adapter round-trip fidelity**: does the exported adapter
          reproduce the training engine's own distribution? Independently
          checks rank slicing, alpha scaling, target-module coverage and
          state-dict key naming (specs/007 §3 open risk (a)).

  import  create a model FROM that checkpoint on a second backend, run the
          same frozen probe batch, and compare to the exporter's JSON.
          -> **G9b seam continuity**: the quantity the Q5 figure rests on.

PASS bar: E1 (corr > 0.999 and |mean logprob gap| < 0.03 nats) — the
cross-engine envelope measured in 006 (G_lp 0.004, G8a 0.031).

Usage:
  python3 -B scripts/gates/g9_adapter_interchange.py export \
      --steps 3 --name g9-seam --out /data/g9_export_<backend>.json
  # kubectl cp /data/checkpoints/<run_id>/<name> to the destination pod
  python3 -B scripts/gates/g9_adapter_interchange.py import \
      --resume tinker://<run_id>/weights/g9-seam \
      --reference /data/g9_export_<backend>.json --out /data/g9_import_<backend>.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

import torch

import tinker
from tinker import types

BASE_MODEL = os.environ.get("G9_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
RANK = int(os.environ.get("G9_RANK", "32"))
# 257 tokens => 256-token inputs after the target shift: miles' TE fused-attn
# backward fails with CUDNN_STATUS_BAD_PARAM on short/odd sequence lengths
# (63 crashed), so keep the probe on a power-of-two length.
SEQ = 257
N_TRAIN = 4          # datums per training step
PROBE_SEED = 20260803
CORR_BAR, GAP_BAR = 0.999, 0.03


def _tokens(seed: int, n: int, length: int) -> list[list[int]]:
    """Deterministic token batches — identical on every backend and in the
    offline oracle (no cookbook/dataset dependency in the seam gate)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, 40000, (n, length), generator=g).tolist()


def _datum(toks: list[int]) -> types.Datum:
    inp, tgt = toks[:-1], toks[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "weights": types.TensorData.from_torch(torch.ones(len(inp), dtype=torch.float32)),
            "target_tokens": types.TensorData.from_torch(torch.tensor(tgt, dtype=torch.long)),
        },
    )


def _per_datum_logprobs(outputs) -> list[list[float]]:
    """loss_fn_outputs entries are Pydantic LossFnOutput on some paths and
    plain dicts on others; both index, only one has .get. Kept per-datum:
    the datum is the alignment unit of the observation contract, and backends
    disagree on the length by one (miles forward() yields L-1 where its
    forward_backward yields L)."""
    per_datum: list[list[float]] = []
    for o in outputs or []:
        try:
            lp = o["logprobs"]
        except Exception:
            lp = getattr(o, "logprobs", None)
        if lp is not None:
            per_datum.append(list(getattr(lp, "data", lp)))
    return per_datum


def _probe_logprobs(tc, probe: list[list[int]]) -> list[list[float]]:
    """Backend's per-token logprobs on the frozen probe batch.

    Uses the forward-only primitive deliberately: forward_backward is not a
    read-only observable on every backend — R9-buffering backends (nemo_rl)
    execute nothing there and hand back zero placeholders, deferring the real
    values to optim_step (001-P3). forward() means the same thing everywhere
    and leaves no gradient state behind.
    """
    out = tc.forward([_datum(t) for t in probe], loss_fn="cross_entropy").result()
    lps = _per_datum_logprobs(out.loss_fn_outputs)
    if not lps or not any(any(d) for d in lps):
        raise SystemExit(
            "forward() returned no per-token logprobs — this backend's "
            "forward-only path does not carry the observable the seam gate needs"
        )
    return lps


def _train(tc, steps: int, lr: float) -> list[float]:
    """Move the adapter off its B=0 init so the export carries real weights."""
    norms = []
    for step in range(steps):
        batch = _tokens(PROBE_SEED + 1000 + step, N_TRAIN, SEQ)
        fb = tc.forward_backward([_datum(t) for t in batch], loss_fn="cross_entropy")
        st = tc.optim_step(types.AdamParams(learning_rate=lr))
        fb.result()
        res = st.result()
        # OptimStepResponse has no top-level grad_norm — it rides in metrics
        # (key name varies by backend: grad_norm / optim/grad_norm / ...).
        metrics = getattr(res, "metrics", None) or {}
        gn = next((v for k, v in metrics.items() if "grad_norm" in k), None)
        norms.append(float(gn) if gn is not None else float("nan"))
        print(f"  step {step}: grad_norm={norms[-1]}")
    return norms


def _compare(a: list[list[float]], b: list[list[float]], label_a: str, label_b: str) -> dict:
    """Align per datum — the unit of the observation contract. A length
    disagreement is REPORTED, never silently absorbed: it is itself a
    cross-path divergence (measured 2026-08-03: miles forward() returns
    L-1 per datum where its forward_backward returns L)."""
    assert len(a) == len(b), f"datum count {len(a)} vs {len(b)}"
    trimmed = [(len(x), len(y)) for x, y in zip(a, b) if len(x) != len(y)]
    if trimmed:
        print(f"  WARNING per-datum length disagreement {label_a}/{label_b}: {trimmed[:4]}")
    flat_a, flat_b = [], []
    for x, y in zip(a, b):
        n = min(len(x), len(y))
        flat_a.extend(x[:n])
        flat_b.extend(y[:n])
    ta = torch.tensor(flat_a, dtype=torch.float64)
    tb = torch.tensor(flat_b, dtype=torch.float64)
    corr = float(torch.corrcoef(torch.stack([ta, tb]))[0, 1])
    gap = float((ta.mean() - tb.mean()).abs())
    verdict = "PASS" if corr > CORR_BAR and gap < GAP_BAR else "FAIL"
    print(
        f"{label_a} vs {label_b}: n={ta.numel()} corr={corr:.6f} "
        f"mean_gap={gap:.6f} nats max|d|={float((ta - tb).abs().max()):.6f} -> {verdict}"
    )
    return {
        "n": int(ta.numel()), "length_mismatches": trimmed, "corr": corr, "mean_gap_nats": gap,
        "max_abs_delta": float((ta - tb).abs().max()),
        f"mean_lp_{label_a}": float(ta.mean()), f"mean_lp_{label_b}": float(tb.mean()),
        "verdict": verdict,
    }


def _offline_adapter_logprobs(adapter_dir: str, probe: list[list[int]]) -> list[list[float]]:
    """fp32 HF + peft oracle: base model with the EXPORTED adapter attached."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    lps: list[list[float]] = []
    with torch.no_grad():
        for toks in probe:
            logits = model(torch.tensor([toks])).logits[0, :-1].float()
            tgt = torch.tensor(toks[1:])
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            lps.append(lp.tolist())
    return lps


def _resolve_local(path: str) -> str:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from training.checkpoints import CheckpointRef, CheckpointStore
    from training.checkpoints.interchange import find_hf_adapter
    from training.config import get_config

    root = str(CheckpointStore(get_config().storage.checkpoint_base, metadata=None).root(CheckpointRef.parse(path)))
    adapter = find_hf_adapter(root)
    if adapter is None:
        raise SystemExit(
            f"G9a FAIL: no hf_adapter/ under {root} — this backend published no "
            f"interchange artifact, so the checkpoint cannot migrate."
        )
    return adapter


def run_export(args) -> dict:
    probe = _tokens(PROBE_SEED, N_TRAIN, SEQ)
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)

    print(f"training {args.steps} steps (lr={args.lr})")
    norms = _train(tc, args.steps, args.lr)

    lp_backend = _probe_logprobs(tc, probe)
    saved = tc.save_state(args.name).result()
    path = getattr(saved, "path", None) or args.name
    print(f"saved: {path}")

    adapter_dir = _resolve_local(path)
    print(f"interchange adapter: {adapter_dir}")
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        adapter_cfg = json.load(f)
    print(f"adapter_config: r={adapter_cfg.get('r')} alpha={adapter_cfg.get('lora_alpha')}")

    lp_oracle = _offline_adapter_logprobs(adapter_dir, probe)
    g9a = _compare(lp_backend, lp_oracle, "backend", "hf_peft_oracle")

    return {
        "gate": "G9a_adapter_roundtrip", "mode": "export",
        "base_model": BASE_MODEL, "rank": RANK, "steps": args.steps, "lr": args.lr,
        "checkpoint_path": path, "adapter_dir": adapter_dir, "adapter_config": adapter_cfg,
        "grad_norms": norms, "probe_seed": PROBE_SEED,
        "logprobs_backend": lp_backend, "result": g9a,
    }


def run_import(args) -> dict:
    probe = _tokens(PROBE_SEED, N_TRAIN, SEQ)
    with open(args.reference) as f:
        ref = json.load(f)

    sc = tinker.ServiceClient()
    # create + load_state: the only resume path (D9); nothing loads at create.
    tc = sc.create_lora_training_client(base_model=ref["base_model"], rank=ref["rank"])
    tc.load_state(args.resume).result()
    lp_dest = _probe_logprobs(tc, probe)

    g9b = _compare(ref["logprobs_backend"], lp_dest, "source_backend", "dest_backend")
    return {
        "gate": "G9b_seam_continuity", "mode": "import",
        "base_model": ref["base_model"], "rank": ref["rank"],
        "resume_path": args.resume, "reference": args.reference,
        "source_checkpoint": ref.get("checkpoint_path"),
        "logprobs_backend": lp_dest, "result": g9b,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "import"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="g9-seam")
    ap.add_argument("--resume")
    ap.add_argument("--reference")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.mode == "import" and not (args.resume and args.reference):
        ap.error("import mode needs --resume and --reference")

    result = run_export(args) if args.mode == "export" else run_import(args)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    print(f"{result['gate']}: {result['result']['verdict']}")
    return 0 if result["result"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
