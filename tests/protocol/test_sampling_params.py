"""Every SamplingParams field reaches the backend for every sample; seeds reproduce."""
from tinker import types


def _last_sample(server, model_id):
    return [t for t in server.trace() if t["op"] == "sample" and t["model_id"] == model_id][-1]


def test_all_params_reach_backend(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sc = tc.save_weights_and_get_sampling_client("p")
    params = types.SamplingParams(max_tokens=5, temperature=0.3, top_p=0.8, top_k=7, seed=42, stop=["END", "\n"])
    sc.sample(prompt=types.ModelInput.from_ints([1, 2]), num_samples=2, sampling_params=params).result()
    seen = _last_sample(server, tc.model_id)["sampling_params"]
    assert seen["max_tokens"] == 5 and seen["temperature"] == 0.3 and seen["top_p"] == 0.8
    assert seen["top_k"] == 7 and seen["seed"] == 42 and seen["stop"] == ["END", "\n"]
    # integer stop entries travel as stop_token_ids
    params = types.SamplingParams(max_tokens=5, stop=[9, 10])
    sc.sample(prompt=types.ModelInput.from_ints([1, 2]), num_samples=1, sampling_params=params).result()
    assert _last_sample(server, tc.model_id)["sampling_params"]["stop_token_ids"] == [9, 10]


def test_seed_reproduces_and_samples_differ(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sc = tc.save_weights_and_get_sampling_client("s")
    prompt = types.ModelInput.from_ints([3, 4, 5])
    a = sc.sample(prompt=prompt, num_samples=3, sampling_params=types.SamplingParams(max_tokens=6, seed=7)).result()
    b = sc.sample(prompt=prompt, num_samples=3, sampling_params=types.SamplingParams(max_tokens=6, seed=7)).result()
    c = sc.sample(prompt=prompt, num_samples=3, sampling_params=types.SamplingParams(max_tokens=6, seed=8)).result()
    def toks(r):
        return [s.tokens for s in r.sequences]
    assert toks(a) == toks(b)
    assert toks(a) != toks(c)
    assert len({tuple(t) for t in toks(a)}) == 3  # the samples of one request are distinct
    assert all(len(s.tokens) == 6 and s.stop_reason == "length" for s in a.sequences)


def test_stop_token_ends_generation(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    sc = tc.save_weights_and_get_sampling_client("t")
    prompt = types.ModelInput.from_ints([3])
    free = sc.sample(prompt=prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=8, seed=1)).result()
    stop_id = free.sequences[0].tokens[2]
    stopped = sc.sample(prompt=prompt, num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=8, seed=1, stop=[stop_id])).result()
    # the stop token itself is returned (API-CONTRACT S5): free[:3] ends with stop_id
    assert stopped.sequences[0].tokens == free.sequences[0].tokens[:3]
    assert stopped.sequences[0].tokens[-1] == stop_id
    assert stopped.sequences[0].stop_reason == "stop"
