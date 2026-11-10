"""Shared by the verl GPU suites: the SmolLM3 student and teachers created with
full weights (LoRA rank 0), real-text tokens, [N, K] datums and an HF
reference for their logprobs."""
import subprocess

import ray
import torch
from ray.util.placement_group import placement_group_table

from tinkercloud.training.models.requests import AdamParams, Datum

FAMILY = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-"
STUDENT = FAMILY + "MixSFT"
TEACHERS = [FAMILY + name for name in ("RL-Math", "RL-Code", "RL-IF")]
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


def adam(learning_rate: float) -> AdamParams:
    return AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
                      grad_clip_norm=0.0)


def text_tokens(count: int) -> list:
    from transformers import AutoTokenizer

    tokens = AutoTokenizer.from_pretrained(STUDENT)(TEXT)["input_ids"]
    assert len(tokens) >= count, f"the text gives {len(tokens)} tokens, the test wants {count}"
    return tokens[:count]


def datum(model_tokens: list, targets: torch.Tensor, weights: torch.Tensor) -> Datum:
    """A cross-entropy datum whose targets and weights are [N] or [N, K] tensors."""
    def wire(t):
        return {"data": t.reshape(-1).tolist(), "shape": list(t.shape)}
    return Datum.model_validate({"model_input": {"tokens": model_tokens}, "loss_fn_inputs": {
        "target_tokens": wire(targets), "weights": wire(weights)}})


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


def gpu_mib() -> list:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def verl_actors() -> list:
    """Named actors a verl model or its sampler leaves alive (workers, vLLM servers)."""
    return sorted(n for n in ray.util.list_named_actors() if n.startswith("verl_") or "vllm" in n)


def vllm_actors() -> list:
    return [n for n in verl_actors() if "vllm" in n]


def live_placement_groups() -> list:
    return sorted(info["name"] for info in placement_group_table().values()
                  if info["name"].startswith("verl_") and info["state"] != "REMOVED")
