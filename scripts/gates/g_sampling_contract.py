"""G-SC: sampling contract on a live server (specs/016 task 2.8; API-CONTRACT S4-S7).

Runs the same assertions as tests/protocol/test_sampling_contract.py against a
real backend with a real tokenizer, on a fresh model that is deleted at the end.

  TINKER_BASE_URL=http://127.0.0.1:8001 python scripts/gates/g_sampling_contract.py \
      --base-model Qwen/Qwen2.5-0.5B-Instruct --out /data/gsc_nemorl.json

Checks: token stop -> last token is the stop id; string stop -> string in decoded
output; max_tokens 8192 accepted (the old 4096 cap is gone); missing max_tokens
-> 422; prompt + max_tokens past the context -> 400 with the typed message;
len(tokens) == len(logprobs) everywhere.
"""
import argparse
import json
import os
import re
import sys
import time

import requests
from tinker import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from gates.drivers.sdk import SDKDriver  # noqa: E402

OUT = {"checks": [], "backend": None, "model_id": None}


def check(name, ok, detail=""):
    OUT["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    drv = SDKDriver()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base_model)
    prompt_ids = tok.encode("The quick brown fox jumps over the lazy dog because", add_special_tokens=True)
    prompt = types.ModelInput.from_ints(prompt_ids)

    t0 = time.time()
    tc = drv.create_training_client(a.base_model, rank=a.rank, max_seq_len=a.max_seq_len)
    OUT["model_id"] = tc.model_id
    OUT["backend"] = requests.get(drv.base_url + "/health", timeout=10).json()
    print(f"model {tc.model_id} booted in {time.time() - t0:.0f}s", flush=True)
    try:
        sc = tc.save_weights_and_get_sampling_client()

        def sample(**kw):
            return sc.sample(prompt=prompt, num_samples=1, sampling_params=types.SamplingParams(**kw)).result()

        # (i) token stop is the last token
        free = sample(max_tokens=16, seed=42, temperature=0.0)  # greedy: reproducible on every engine
        f = free.sequences[0]
        check("len(tokens)==len(logprobs) free", len(f.tokens) == len(f.logprobs), f"{len(f.tokens)} tokens, stop={f.stop_reason}")
        if len(f.tokens) >= 6:
            stop_id = f.tokens[4]
            st = sample(max_tokens=64, seed=42, temperature=0.0, stop=[stop_id]).sequences[0]
            check("token stop: last token is stop id", st.tokens and st.tokens[-1] == stop_id and st.stop_reason == "stop",
                  f"len={len(st.tokens)} last={st.tokens[-1] if st.tokens else None} stop_id={stop_id} stop_reason={st.stop_reason}")
            check("token stop: greedy prefix matches free sample", st.tokens == f.tokens[:5], f"{st.tokens[:6]} vs {f.tokens[:6]}")
            check("len(tokens)==len(logprobs) stopped", len(st.tokens) == len(st.logprobs))
        else:
            check("token stop", False, f"free sample too short ({len(f.tokens)}) to pick a stop id")

        # (ii) string stop is included in the output
        base = sample(max_tokens=40, seed=7, temperature=0.0).sequences[0]
        text = tok.decode(base.tokens)
        mid = len(text) // 2
        ss = text[mid:mid + 5]
        if ss.strip() and ss in text:
            sst = sample(max_tokens=40, seed=7, temperature=0.0, stop=[ss]).sequences[0]
            dec = tok.decode(sst.tokens)
            check("string stop: stop_reason", sst.stop_reason == "stop", f"stop_reason={sst.stop_reason} stop={ss!r}")
            check("string stop: string present in decoded output", ss in dec, f"tail={dec[-60:]!r}")
            check("string stop: shorter or equal", len(sst.tokens) <= len(base.tokens), f"{len(sst.tokens)} <= {len(base.tokens)}")
            check("len(tokens)==len(logprobs) string-stopped", len(sst.tokens) == len(sst.logprobs))
        else:
            check("string stop", False, f"no usable substring in {text!r}")

        # (iii) max_tokens above the old cap is accepted
        try:
            big = sample(max_tokens=8192, seed=3, temperature=1.0, stop=[stop_id] if len(f.tokens) >= 6 else None).sequences[0]
            check("max_tokens=8192 accepted", True, f"returned {len(big.tokens)} tokens, stop={big.stop_reason}")
            check("len(tokens)==len(logprobs) big", len(big.tokens) == len(big.logprobs))
        except Exception as e:  # noqa: BLE001
            check("max_tokens=8192 accepted", False, str(e)[:200])

        # (iv) missing max_tokens -> 422 at the request model (raw HTTP)
        body = {"sampling_session_id": sc._sampling_session_id, "seq_id": 990001, "num_samples": 1,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt_ids}]},
                "sampling_params": {"temperature": 1.0}}
        r = requests.post(drv.base_url + "/api/v1/asample", json=body, headers={"X-API-Key": drv.api_key}, timeout=30)
        check("missing max_tokens -> 422", r.status_code == 422 and "max_tokens" in r.text, f"http={r.status_code}")

        # (v) prompt + max_tokens over the context -> 400 with the typed message
        try:
            sample(max_tokens=10_000_000, seed=1, temperature=1.0)
            check("over-context -> 400", False, "request accepted")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            m = re.search(r"prompt_len (\d+) \+ max_tokens (\d+) = (\d+) exceeds the model context (\d+)", msg)
            check("over-context -> 400 with typed message", "400" in msg and m is not None, msg[:220])
            if m:
                OUT["context_length"] = int(m.group(4))
                ctx = int(m.group(4))
                # exactly one over the limit is also rejected; exactly at the limit is accepted
                try:
                    sample(max_tokens=ctx - len(prompt_ids) + 1, seed=1, temperature=1.0)
                    check("context+1 -> 400", False, "accepted")
                except Exception as e2:  # noqa: BLE001
                    check("context+1 -> 400", "400" in str(e2), str(e2)[:120])
                fit = sample(max_tokens=ctx - len(prompt_ids), seed=1, temperature=1.0,
                             stop=[stop_id] if len(f.tokens) >= 6 else None).sequences[0]
                check("context-exact fit accepted", True, f"returned {len(fit.tokens)} tokens, stop={fit.stop_reason}")
    finally:
        ok = drv.delete_model(tc.model_id)
        OUT["deleted"] = ok
        print(f"delete_model {tc.model_id}: {ok}", flush=True)
        OUT["all_ok"] = all(c["ok"] for c in OUT["checks"])
        with open(a.out, "w") as fh:
            json.dump(OUT, fh, indent=2)
        print(f"ALL_OK={OUT['all_ok']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
