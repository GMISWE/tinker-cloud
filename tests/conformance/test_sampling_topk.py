"""SDK 0.33 sampling fields over HTTP on the fake backend: the unsupported ones
are refused by name (400) at submission, and topk_sample_logprobs /
topk_prompt_logprobs / prompt_logprobs_last_n travel the proto result the way
the SDK's response_conv reads them (sentinel-filled top-k rows, NaN prompt
logprobs before the last-N suffix). The raw-HTTP tests run under any SDK; the
SDK-driven one needs tinker >= 0.33."""
import math

import numpy as np
import pytest
from tinker import types

from tinkercloud.training.proto import tinker_public_pb2 as pb
from tinkercloud.training.proto.wire import PROTO_CONTENT_TYPE

ACCEPT_PROTO = {"Accept": PROTO_CONTENT_TYPE}
MASK = (0, -99999.0)   # the SDK's undefined top-k cell
PROMPT = [3, 4, 5, 6, 7, 8]


def _sampler_path(service_client, name):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    return tc.save_weights_for_sampler(name).result().path


def _asample(server, path, **fields):
    return server.post("/api/v1/asample", {
        "model_path": path, "prompt": {"tokens": PROMPT}, "num_samples": 1,
        "sampling_params": {"max_tokens": 3, "seed": 1}, **fields,
    })


def _retrieve_proto(server, rid):
    r = server.post_raw("/api/v1/retrieve_future", b'{"request_id": "%s", "allow_metadata_only": true}' % rid.encode(),
                        {"Content-Type": "application/json", **ACCEPT_PROTO}, timeout=60)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith(PROTO_CONTENT_TYPE)
    out = pb.SampleResponse()
    out.ParseFromString(r.content)
    return out


def _topk_rows(msg):
    """TopkLogprobs -> rows of (token_id, logprob), None where every cell is the sentinel."""
    n, k = msg.length, msg.k
    ids = np.frombuffer(msg.token_ids, dtype=np.int32).reshape(n, k).tolist()
    lps = np.frombuffer(msg.logprobs, dtype=np.float32).reshape(n, k).tolist()
    rows = []
    for row_ids, row_lps in zip(ids, lps):
        cells = [(t, lp) for t, lp in zip(row_ids, row_lps) if (t, lp) != MASK]
        rows.append(cells or None)
    return rows


@pytest.mark.parametrize("fields, named", [
    ({"target_prompt_logprobs": {"data": [4, -1], "dtype": "int64", "shape": [1, 2]}}, "target_prompt_logprobs"),
    ({"prompt_alt_tokens_k": 2}, "prompt_alt_tokens_k"),
    ({"prompt_logprobs_last_n": 2}, "prompt_logprobs_last_n requires prompt_logprobs"),
    ({"prompt_logprobs": True, "prompt_logprobs_last_n": 6}, "prompt_logprobs_last_n must be between 1 and len(prompt) - 1 = 5"),
])
def test_unsupported_or_invalid_fields_are_400_by_name(service_client, server, fields, named):
    path = _sampler_path(service_client, "t400")
    r = _asample(server, path, **fields)
    assert r.status_code == 400, r.text
    assert named in r.json()["error"]  # the server reports HTTPException.detail as "error"


def test_topk_above_the_engine_cap_fails_the_future_with_400(service_client, server):
    path = _sampler_path(service_client, "tcap")
    rid = _asample(server, path, topk_sample_logprobs=21).json()["request_id"]
    r = server.post("/api/v1/retrieve_future", {"request_id": rid, "allow_metadata_only": True}, timeout=60)
    assert r.status_code == 400, r.text
    assert "topk_sample_logprobs 21 exceeds the engine cap 20" in r.text


def test_topk_and_last_n_on_the_proto_wire(service_client, server):
    path = _sampler_path(service_client, "twire")
    r = _asample(server, path, num_samples=2, prompt_logprobs=True, topk_prompt_logprobs=2,
                 topk_sample_logprobs=2, prompt_logprobs_last_n=3)
    assert r.status_code == 200, r.text
    out = _retrieve_proto(server, r.json()["request_id"])
    assert len(out.sequences) == 2
    for seq in out.sequences:
        tokens = np.frombuffer(seq.tokens, dtype=np.int32).tolist()
        logprobs = np.frombuffer(seq.logprobs, dtype=np.float32).tolist()
        assert (seq.topk_sampled_logprobs.length, seq.topk_sampled_logprobs.k) == (len(tokens), 2)
        rows = _topk_rows(seq.topk_sampled_logprobs)
        # the fake engine ranks the sampled token first at its own logprob
        assert [row[0][0] for row in rows] == tokens
        assert [row[0][1] for row in rows] == pytest.approx(logprobs)
        assert all(len(row) == 2 and row[0][1] > row[1][1] for row in rows)
    plp = np.frombuffer(out.prompt_logprobs, dtype=np.float32)
    assert len(plp) == len(PROMPT)
    assert all(math.isnan(v) for v in plp[:3]) and not any(math.isnan(v) for v in plp[3:])
    prompt_rows = _topk_rows(out.topk_prompt_logprobs)
    assert (out.topk_prompt_logprobs.length, out.topk_prompt_logprobs.k) == (len(PROMPT), 2)
    assert prompt_rows[:3] == [None, None, None]
    assert all(len(row) == 2 for row in prompt_rows[3:])
    assert [row[0][1] for row in prompt_rows[3:]] == pytest.approx(plp[3:].tolist())


def test_plain_sampling_carries_no_topk(service_client, server):
    path = _sampler_path(service_client, "tplain")
    out = _retrieve_proto(server, _asample(server, path, prompt_logprobs=True).json()["request_id"])
    assert not out.sequences[0].HasField("topk_sampled_logprobs")
    assert not out.HasField("topk_prompt_logprobs")
    assert not math.isnan(np.frombuffer(out.prompt_logprobs, dtype=np.float32)[1])   # no last_n: whole prompt scored


def test_sdk_sample_with_topk_and_last_n(service_client):
    if "topk_sample_logprobs" not in types.SampleRequest.model_fields:
        pytest.skip("tinker SDK predates topk_sample_logprobs (>= 0.33 needed)")
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sc = tc.save_weights_and_get_sampling_client("tsdk")
    res = sc.sample(prompt=types.ModelInput.from_ints(PROMPT), num_samples=2,
                    sampling_params=types.SamplingParams(max_tokens=3, seed=1),
                    include_prompt_logprobs=True, topk_prompt_logprobs=2, topk_sample_logprobs=2,
                    prompt_logprobs_last_n=3).result()
    for seq in res.sequences:
        assert seq.topk_logprobs_np.token_ids.shape == (len(seq.tokens), 2)
        assert [row[0][0] for row in seq.topk_logprobs] == seq.tokens
    assert res.prompt_logprobs[:3] == [None, None, None] and None not in res.prompt_logprobs[3:]
    assert res.topk_prompt_logprobs[:3] == [None, None, None]
    assert all(len(row) == 2 for row in res.topk_prompt_logprobs[3:])
    # compute_logprobs rides the same fields
    lp = sc.compute_logprobs(types.ModelInput.from_ints(PROMPT), prompt_logprobs_last_n=2).result()
    assert lp[:4] == [None] * 4 and None not in lp[4:]
