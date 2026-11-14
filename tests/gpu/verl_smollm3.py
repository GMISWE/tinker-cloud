"""Shared by the verl GPU suites: the SmolLM3 student and teachers created with
full weights (LoRA rank 0), real-text tokens, [N, K] datums and an HF
reference for their logprobs."""
import subprocess

import ray
import torch
from ray.util.placement_group import placement_group_table

from tinkercloud.training.backends.verl.backend import (
    SHARED_POOL_NAME_PREFIX, VLLM_SERVER_NAME_PREFIX, WORKER_NAME_PREFIX,
)
from tinkercloud.training.models.requests import AdamParams

from ..unit.backends.verl.datums import nk_datum as datum  # noqa: F401  (the suites import it from here)

FAMILY = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-"
STUDENT = FAMILY + "MixSFT"
TEACHER_OF_DOMAIN = {"math": FAMILY + "RL-Math", "code": FAMILY + "RL-Code", "if": FAMILY + "RL-IF"}
TEACHERS = list(TEACHER_OF_DOMAIN.values())
FULL_WEIGHTS = {"rank": 0}
NUM_GPUS = 4
K = 16

TEXT = (
    "The lighthouse keeper climbed the spiral stairs every evening before the sun went down. He carried a small "
    "brass lamp, a notebook with a cracked leather cover, and a thermos of black tea. From the gallery he could "
    "see the fishing boats returning to the harbour, their hulls low in the water when the catch had been good. "
    "He wrote down the name of each boat, the hour it passed the rocks, and the colour of the sky behind it. "
    "On clear nights he counted the stars above the northern headland and compared the number with the one in "
    "his notebook from the year before. When the fog came in from the sea he lit the great lamp early, wound the "
    "clockwork that turned the lens, and listened for the horn of the mail steamer. The steamer was always late "
    "in autumn. Its captain blamed the tides, the coal, and the harbour master, in that order, and the keeper "
    "wrote those down as well. In forty years he had never missed an evening, and he did not intend to start now."
)


def create_full_weights(loop, backend, model_id: str, base_model: str):
    """`base_model` created at full weights on NUM_GPUS; the handle."""
    return loop.run_until_complete(backend.create_model(
        model_id, "create", base_model, NUM_GPUS, lora_config=FULL_WEIGHTS, optimizer="adamw"))


def adam(learning_rate: float) -> AdamParams:
    return AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                      grad_clip_norm=0.0)


def text_tokens(count: int) -> list:
    from transformers import AutoTokenizer

    tokens = AutoTokenizer.from_pretrained(STUDENT)(TEXT)["input_ids"]
    assert len(tokens) >= count, f"the text gives {len(tokens)} tokens, the test wants {count}"
    return tokens[:count]


def candidate_ids(tokens: list, generator: torch.Generator) -> torch.Tensor:
    """[len(tokens) - 1, K] candidate ids: the next token in column 0, random ids beside it."""
    ids = torch.randint(0, 30000, (len(tokens) - 1, K), generator=generator)
    ids[:, 0] = torch.tensor(tokens[1:])
    return ids


def logprobs_of(result: dict) -> torch.Tensor:
    """The first datum's logprobs of a forward or forward_backward result, in their wire shape."""
    out = result["loss_fn_outputs"][0]["logprobs"]
    return torch.tensor(out["data"], dtype=torch.float32).reshape(out["shape"])


def hf_log_softmax(model_name: str, tokens: list, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Plain HF: log_softmax (float32, on CPU) of the logits after each of `tokens`, [len(tokens), V]."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    with torch.no_grad():
        logits = model(torch.tensor([tokens], device=device)).logits[0]
    return torch.log_softmax(logits.float().cpu(), dim=-1)


def _recorded_datum(recorded: dict, row: int):
    """Row `row` of a recorded batch as an SDK datum with `[N, 16]` targets, and its prompt length."""
    from tinker import types

    attention = recorded["attention_mask"][row].bool()
    tokens = recorded["input_ids"][row][attention]
    prompt_len = int(attention[:recorded["prompts"].shape[1]].sum())
    targets = torch.zeros(len(tokens) - 1, K, dtype=torch.long)
    targets[prompt_len - 1:] = recorded["student_top_k_ids"][row, :len(tokens) - prompt_len]
    return types.Datum(model_input=types.ModelInput.from_ints(tokens[:-1].tolist()),
                       loss_fn_inputs={"target_tokens": types.TensorData.from_torch(targets)}), prompt_len


def recorded_update(path: str, shards: int, shard_rows: int, update_rows: int, update: int):
    """Update number `update` of a recorded batch (the file `test_verl_fit.py` describes: `shards`
    ranks of `shard_rows` rows, `update_rows` of each rank to an update) as SDK datums with
    `[N, 16]` targets: (datums, the first row's prompt tokens, each row's prompt length)."""
    rows = [shard_rows * shard + update_rows * update + i for shard in range(shards) for i in range(update_rows)]
    recorded = torch.load(path, weights_only=True)["tensors"]
    data, prompt_lens = zip(*(_recorded_datum(recorded, row) for row in rows), strict=True)
    return list(data), data[0].model_input.to_ints()[:prompt_lens[0]], list(prompt_lens)


def recorded_by_domain(path: str) -> dict:
    """A recorded batch's rows as SDK datums with `[N, 16]` targets, by the domain the file names for each row."""
    recorded = torch.load(path, weights_only=True)
    data: dict = {}
    for row, domain in enumerate(recorded["domain"]):
        data.setdefault(str(domain), []).append(_recorded_datum(recorded["tensors"], row)[0])
    return data


def gpu_mib() -> list:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def verl_actors() -> list:
    """Named actors a verl model or its sampler leaves alive (workers, vLLM servers)."""
    return sorted(n for n in ray.util.list_named_actors() if n.startswith((WORKER_NAME_PREFIX, VLLM_SERVER_NAME_PREFIX)))


def vllm_actors() -> list:
    return [n for n in verl_actors() if n.startswith(VLLM_SERVER_NAME_PREFIX)]


def live_placement_groups() -> list:
    return sorted(info["name"] for info in placement_group_table().values()
                  if info["name"].startswith(SHARED_POOL_NAME_PREFIX) and info["state"] != "REMOVED")
