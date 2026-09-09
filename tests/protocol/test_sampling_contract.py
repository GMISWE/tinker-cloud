"""Sampling contract (specs/016 task 2.8, API-CONTRACT S4-S7), driven by the SDK.

Shape ported from SkyRL tests/tinker/test_api.py (stop tokens, stop strings).
The fake backend decodes token t as its decimal digits, space-joined, so stop
strings are testable without a tokenizer; context_length == create_model's
max_seq_len.
"""
import pytest
from tinker import types


def _decode(tokens):
    return " ".join(str(t) for t in tokens)


def _client(service_client, name, max_seq_len=2048):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2, max_seq_len=max_seq_len)
    return tc, tc.save_weights_and_get_sampling_client(name)


def test_token_stop_is_last_token(service_client):
    tc, sc = _client(service_client, "c1")
    prompt = types.ModelInput.from_ints([11, 12])
    free = sc.sample(prompt=prompt, num_samples=1,
                     sampling_params=types.SamplingParams(max_tokens=10, seed=42)).result()
    stop_token = free.sequences[0].tokens[4]
    stopped = sc.sample(prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=50, seed=42, stop=[stop_token])).result()
    seq = stopped.sequences[0]
    assert len(seq.tokens) == 5
    assert seq.tokens[-1] == stop_token
    assert seq.stop_reason == "stop"
    assert len(seq.tokens) == len(seq.logprobs)


def test_string_stop_is_included_in_output(service_client):
    tc, sc = _client(service_client, "c2")
    prompt = types.ModelInput.from_ints([21, 22])
    baseline = sc.sample(prompt=prompt, num_samples=1,
                         sampling_params=types.SamplingParams(max_tokens=20, seed=7)).result()
    text = _decode(baseline.sequences[0].tokens)
    stop_string = text[len(text) // 2:][:3].strip()
    assert stop_string and stop_string in text
    stopped = sc.sample(prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=20, seed=7, stop=[stop_string])).result()
    seq = stopped.sequences[0]
    assert seq.stop_reason == "stop"
    assert len(seq.tokens) <= len(baseline.sequences[0].tokens)
    assert stop_string in _decode(seq.tokens)  # the stop string's tokens are returned
    assert len(seq.tokens) == len(seq.logprobs)


def test_max_tokens_above_4096_is_honoured(service_client):
    tc, sc = _client(service_client, "c3", max_seq_len=16384)
    out = sc.sample(prompt=types.ModelInput.from_ints([1, 2, 3]), num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=8192, seed=1)).result()
    seq = out.sequences[0]
    assert len(seq.tokens) == 8192 and seq.stop_reason == "length"
    assert len(seq.tokens) == len(seq.logprobs)


def test_missing_max_tokens_is_rejected(service_client, server):
    """The SDK cannot omit max_tokens without dropping it from the wire; the raw
    HTTP path shows what the server does then: 422 at the request model."""
    tc, sc = _client(service_client, "c4")
    body = {"sampling_session_id": sc._sampling_session_id, "seq_id": 999, "num_samples": 1,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
            "sampling_params": {"temperature": 1.0}}
    r = server.post("/api/v1/asample", body)
    assert r.status_code == 422, r.text
    assert "max_tokens" in r.text


def test_prompt_plus_max_tokens_over_context_is_400(service_client):
    tc, sc = _client(service_client, "c5", max_seq_len=64)
    prompt = types.ModelInput.from_ints(list(range(1, 41)))  # 40 tokens
    ok = sc.sample(prompt=prompt, num_samples=1,
                   sampling_params=types.SamplingParams(max_tokens=24, seed=1)).result()
    assert len(ok.sequences[0].tokens) == 24
    # A terminally failed future reaches the SDK as a 400 on retrieve_future
    # (not retried), surfaced as ValueError with the server's message.
    with pytest.raises(ValueError) as ei:
        sc.sample(prompt=prompt, num_samples=1,
                  sampling_params=types.SamplingParams(max_tokens=25, seed=1)).result()
    msg = str(ei.value)
    assert "400" in msg
    assert "prompt_len 40" in msg and "max_tokens 25" in msg and "context 64" in msg
