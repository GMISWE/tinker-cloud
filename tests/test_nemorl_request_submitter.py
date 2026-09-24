"""NemoRLRequestSubmitter (specs/019): one generate_rows_async call per sample
with per-row seeds; a cancelled or failed sample spares the others."""
import asyncio

import pytest
import torch

from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle
from tinkercloud.training.backends.nemo_rl.generation import NemoRLRequestSubmitter

EOS = 2


class FakeTokenizer:
    eos_token_id = EOS

    def decode(self, toks, skip_special_tokens=False):
        return "".join(chr(97 + t % 26) for t in toks)


class FakeEngine:
    """generate_rows_async stand-in: every row's output = prompt + reply, rows
    returned in reverse order. A call whose max_tokens is HANG never returns;
    one whose first seed is FAIL raises. `ended` counts calls that ended by
    return, error or cancellation."""
    HANG = 4096
    FAIL = 77

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.ended = 0
        self.started = asyncio.Event()

    async def generate_rows_async(self, data, greedy):
        self.calls.append(data)
        self.started.set()
        try:
            if data["_tinker_seed"][0] == self.FAIL:
                raise RuntimeError("engine boom")
            if data["_tinker_max_new_tokens"][0] == self.HANG:
                await asyncio.sleep(3600)
            rows = []
            for i in reversed(range(len(data["input_lengths"]))):
                plen = int(data["input_lengths"][i])
                ids = torch.cat([data["input_ids"][i, :plen], torch.tensor(self.reply)]).unsqueeze(0)
                rows.append((i, {
                    "output_ids": ids,
                    "logprobs": torch.full((1, ids.shape[1]), -0.5 - i),
                    "generation_lengths": torch.tensor([len(self.reply)]),
                }))
            return rows
        finally:
            self.ended += 1


def handle_with(engine):
    return NemoRLHandle(model_id="m", backend_type="nemo_rl",
                        policy_generation=engine, tokenizer=FakeTokenizer())


def submit(sub, h, **kw):
    args = dict(request_id="r", prompt_tokens=[5, 6, 7], num_samples=1,
                sampling_params={"max_tokens": 8, "temperature": 1.0}, prompt_logprobs=False)
    args.update(kw)
    return sub.submit(h, **args)


def test_one_call_per_sample_with_per_row_seeds_and_ordered_sequences():
    eng = FakeEngine(reply=[0, 1, EOS])
    res = asyncio.run(submit(NemoRLRequestSubmitter(8), handle_with(eng),
                             num_samples=3, sampling_params={"max_tokens": 8, "seed": 10}))
    (call,) = eng.calls
    assert call["_tinker_seed"] == [10, 11, 12]
    assert call["input_ids"].shape == (3, 3) and call["input_lengths"].tolist() == [3, 3, 3]
    assert call["_tinker_max_new_tokens"] == [8, 8, 8]
    assert [s["tokens"] for s in res["sequences"]] == [[0, 1, EOS]] * 3
    assert [s["logprobs"][0] for s in res["sequences"]] == [-0.5, -1.5, -2.5]  # row order restored
    assert {s["stop_reason"] for s in res["sequences"]} == {"stop"}
    assert res["prompt_logprobs"] is None


def test_length_stop_and_stop_string_truncation():
    eng = FakeEngine(reply=[0, 1, 2, 3])          # decodes to "abcd"
    res = asyncio.run(submit(NemoRLRequestSubmitter(8), handle_with(eng),
                             sampling_params={"max_tokens": 4, "stop": ["c"]}))
    seq = res["sequences"][0]
    assert seq["tokens"] == [0, 1, 2] and seq["stop_reason"] == "stop" and seq["text"] == "abc"
    assert eng.calls[0]["stop_strings"] == [["c"]]
    res = asyncio.run(submit(NemoRLRequestSubmitter(8), handle_with(FakeEngine(reply=[0, 1, 3, 4])),
                             sampling_params={"max_tokens": 4}))
    assert res["sequences"][0]["stop_reason"] == "length"


def test_prompt_logprobs_skip_position_zero():
    res = asyncio.run(submit(NemoRLRequestSubmitter(8), handle_with(FakeEngine(reply=[0])),
                             prompt_logprobs=True))
    assert res["prompt_logprobs"] == [None, -0.5, -0.5]


def test_cancel_ends_its_call_and_spares_others():
    async def run():
        eng = FakeEngine(reply=[0, EOS])
        h = handle_with(eng)
        sub = NemoRLRequestSubmitter(8)
        hanging = asyncio.create_task(submit(sub, h, num_samples=2, sampling_params={"max_tokens": FakeEngine.HANG}))
        await eng.started.wait()
        await asyncio.sleep(0)
        other = await submit(sub, h)                 # unaffected while the other hangs
        assert other["sequences"][0]["tokens"] == [0, EOS]
        assert eng.ended == 1
        hanging.cancel()
        with pytest.raises(asyncio.CancelledError):
            await hanging
        assert eng.ended == 2
    asyncio.run(run())


def test_one_failed_call_fails_only_its_sample():
    async def run():
        eng = FakeEngine(reply=[0, EOS])
        h = handle_with(eng)
        sub = NemoRLRequestSubmitter(8)
        with pytest.raises(RuntimeError, match="engine boom"):
            await submit(sub, h, sampling_params={"max_tokens": 8, "seed": FakeEngine.FAIL})
        ok = await submit(sub, h, sampling_params={"max_tokens": 8, "seed": 1})
        assert ok["sequences"][0]["tokens"] == [0, EOS]
    asyncio.run(run())


def test_in_flight_calls_are_bounded():
    async def run():
        eng = FakeEngine(reply=[0, EOS])
        h = handle_with(eng)
        sub = NemoRLRequestSubmitter(max_in_flight=1)
        t1 = asyncio.create_task(submit(sub, h, sampling_params={"max_tokens": FakeEngine.HANG}))
        t2 = asyncio.create_task(submit(sub, h))
        await eng.started.wait()
        await asyncio.sleep(0.05)
        assert len(eng.calls) == 1                    # the second call waits on the bound
        t1.cancel()
        await asyncio.gather(t1, return_exceptions=True)
        assert (await t2)["sequences"][0]["tokens"] == [0, EOS]
    asyncio.run(run())
