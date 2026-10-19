"""FakeBackend: deterministic top-k sampling that validates against the result model."""
import asyncio

from tinkercloud.training.backends.fake.backend import FakeBackend, logprobs_for, topk_for
from tinkercloud.training.models.responses import validate_result


PROMPT = [1, 2, 3, 4, 5]


def _fake_model():
    b = FakeBackend()
    return b, asyncio.run(b.create_model("m", "r", "fake/tiny", 0))


def test_fake_topk_is_deterministic_and_validates():
    b, h = _fake_model()
    kw = dict(sampling_params={"max_tokens": 3, "seed": 1}, prompt_logprobs=True,
              topk_sample_logprobs=2, topk_prompt_logprobs=2)
    a = asyncio.run(b.sample(h, "r", PROMPT, 2, **kw))
    assert a == asyncio.run(b.sample(h, "r", PROMPT, 2, **kw))
    for seq in a["sequences"]:
        rows = seq["topk_sample_logprobs"]
        assert len(rows) == len(seq["tokens"])
        # rank 0 is the sampled token at its own logprob; ranks are best first
        assert [row[0] for row in rows] == list(zip(seq["tokens"], seq["logprobs"]))
        assert all(row[0][1] > row[1][1] for row in rows)
    assert a["topk_prompt_logprobs"][0] is None
    assert a["topk_prompt_logprobs"][1:] == topk_for(PROMPT, 2)[1:]
    assert [row[0][1] for row in a["topk_prompt_logprobs"][1:]] == logprobs_for(PROMPT)[1:]
    plain = asyncio.run(b.sample(h, "r", PROMPT, 1, sampling_params={"max_tokens": 3, "seed": 1}))
    assert "topk_sample_logprobs" not in plain["sequences"][0] and "topk_prompt_logprobs" not in plain
    out = validate_result("asample", a)
    assert out["sequences"][0]["topk_sample_logprobs"] == a["sequences"][0]["topk_sample_logprobs"]
    assert out["topk_prompt_logprobs"] == a["topk_prompt_logprobs"]
