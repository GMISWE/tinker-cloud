#!/usr/bin/env python3
"""G10 staleness-k gate (A4, specs/012).

Certifies the ver(S) contract through the public API on the nemo_rl backend,
at both the label level (version fields in responses) and the value level
(greedy logprobs bit-identical to the version the label claims).

Phase A — tenant declares staleness_k=1:
  A0: greedy sample S0 at v0 (pre-training).
  A1: fb+optim step 1 -> latest v1. Sample: served==0, latest==1 (deferred),
      and tokens+logprobs == S0 bit-identical (the engine really holds v0).
  A2: step 2 -> latest v2, staleness would be 2 -> refit. Sample S2:
      served==2, latest==2, and logprobs DIFFER from S0 (weights moved;
      the A1 match was not vacuous).
  A3: step 3 -> deferred again. Sample: served==2, latest==3, and
      tokens+logprobs == S2 bit-identical.
Phase B — fresh tenant, default staleness_k=0:
  B1: every post-step sample has served==latest (strict on-policy), and
      the post-step sample differs from the pre-step one.

PASS = all assertions. Run against a nemo_rl server (2 GPUs is enough).
"""
import os, sys, time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")
BASE = os.environ["TINKER_BASE_URL"]
KEY = os.environ["TINKER_API_KEY"]

import requests
import torch
import tinker
from tinker import types

BASE_MODEL = os.environ.get("G10_BASE_MODEL", "Qwen/Qwen2.5-0.5B")
RANK = 32
SEQ = 64
BATCH = 8
LR = 1e-4  # large on purpose: one step must move greedy logprobs measurably
PROMPT = [(1000 + j * 104729) % 50000 for j in range(24)]
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}
FAILS = []


def check(name, ok, detail=""):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}{' ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def make_datums(n, salt):
    out = []
    for i in range(n):
        toks = [(1000 + (salt * 31 + i) * 7919 + j * 104729) % 50000 for j in range(SEQ)]
        inp, tgt = toks[:-1], toks[1:]
        out.append(types.Datum(
            model_input=types.ModelInput.from_ints(inp),
            loss_fn_inputs={
                "weights": types.TensorData.from_torch(
                    torch.ones(len(inp), dtype=torch.float32)),
                "target_tokens": types.TensorData.from_torch(
                    torch.tensor(tgt, dtype=torch.long)),
            },
        ))
    return out


def await_future(rid, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        fr = requests.post(f"{BASE}/api/v1/retrieve_future/{rid}", headers=HDRS)
        if fr.status_code == 200:
            return fr.json()
        if fr.status_code == 408:
            time.sleep(2)
            continue
        print(f"FUTURE FAILED ({fr.status_code}):", fr.text[:2000])
        sys.exit(2)
    print("FUTURE TIMEOUT")
    sys.exit(2)


def raw_sample(session_id):
    """Greedy sample via raw HTTP so the ver(S) fields are readable. The
    sampler session serves the live engine (its pin only routes
    prompt-logprob reads, BUG-015)."""
    body = {
        "sampling_session_id": session_id,
        "prompt": {"tokens": PROMPT},
        "num_samples": 1,
        "sampling_params": {"temperature": 0.0, "max_tokens": 24, "top_p": 1.0},
    }
    r = requests.post(f"{BASE}/api/v1/asample", headers=HDRS, json=body)
    r.raise_for_status()
    res = await_future(r.json()["request_id"])
    seq = res["sequences"][0]
    return {
        "tokens": seq["tokens"],
        "logprobs": seq["logprobs"],
        "served": res.get("weight_version"),
        "latest": res.get("latest_weight_version"),
    }


def train_step(tc, salt):
    fb = tc.forward_backward(make_datums(BATCH, salt), "cross_entropy")
    op = tc.optim_step(types.AdamParams(learning_rate=LR))
    fb.result()
    op.result()


def run_phase(sc, k, steps):
    tc = sc.create_lora_training_client(
        base_model=BASE_MODEL, rank=RANK, staleness_k=k)
    model_id = tc.model_id
    print(f"model {model_id} staleness_k={k}")
    session_id = tc.save_weights_and_get_sampling_client()._sampling_session_id
    samples = [raw_sample(session_id)]  # index = after step i
    for t in range(1, steps + 1):
        train_step(tc, salt=t)
        samples.append(raw_sample(session_id))
    requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS,
                  json={"model_id": model_id})
    return samples


def bit_eq(a, b):
    return repr(a["tokens"]) == repr(b["tokens"]) and repr(a["logprobs"]) == repr(b["logprobs"])


def main():
    sc = tinker.ServiceClient(base_url=BASE, api_key=KEY)

    print("Phase A: staleness_k=1")
    s = run_phase(sc, k=1, steps=3)
    check("A0 pre-train versions", s[0]["served"] == 0 and s[0]["latest"] == 0,
          f"served={s[0]['served']} latest={s[0]['latest']}")
    check("A1 label: served 0, latest 1", s[1]["served"] == 0 and s[1]["latest"] == 1,
          f"served={s[1]['served']} latest={s[1]['latest']}")
    check("A1 value: sample == v0 bit-identical", bit_eq(s[1], s[0]))
    check("A2 label: served 2, latest 2 (refit)", s[2]["served"] == 2 and s[2]["latest"] == 2,
          f"served={s[2]['served']} latest={s[2]['latest']}")
    check("A2 value: sample != v0 (weights moved)", not bit_eq(s[2], s[0]))
    check("A3 label: served 2, latest 3 (deferred)", s[3]["served"] == 2 and s[3]["latest"] == 3,
          f"served={s[3]['served']} latest={s[3]['latest']}")
    check("A3 value: sample == v2 bit-identical", bit_eq(s[3], s[2]))

    print("Phase B: staleness_k=0 (default strict)")
    s = run_phase(sc, k=0, steps=2)
    ok = all(x["served"] == x["latest"] for x in s)
    check("B1 label: served == latest at every step", ok,
          " ".join(f"({x['served']},{x['latest']})" for x in s))
    check("B2 value: post-step sample != pre-step", not bit_eq(s[1], s[0]))

    print("RESULT:", "PASS" if not FAILS else f"FAIL ({FAILS})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
