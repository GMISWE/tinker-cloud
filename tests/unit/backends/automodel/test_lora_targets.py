"""The LoRA the automodel worker builds, on the real models built on the meta device
the way the worker builds them: its element count is Tinker's for every flag set,
and Qwen3.5's `in_proj_qkv` carries one independent LoRA pair per output slice."""
import pytest
import torch

pytest.importorskip("nemo_automodel")
hyperparam_utils = pytest.importorskip("tinker_cookbook.hyperparam_utils")

from accelerate import init_empty_weights  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from nemo_automodel import NeMoAutoModelForCausalLM  # noqa: E402
from nemo_automodel._transformers import model_init  # noqa: E402

from tinkercloud.training.backends.automodel.lora import lora_spec, model_lora  # noqa: E402
from tinkercloud.training.backends.automodel.peft import SlicedLinearLoRA, apply_lora, build_overrides, patch_sliced, peft_config  # noqa: E402
from tinkercloud.training.models.requests import LoraConfig  # noqa: E402
from tinkercloud.training.utils.model_config import read_raw_hf_config  # noqa: E402

MODELS = ("Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B")
FLAGS = [(True, False), (False, True), (True, True)]   # (train_attn, train_mlp)
# modules Tinker's counts leave out
NOT_ADAPTED = ("in_proj_b", "in_proj_a", "lm_head", "visual", "mtp")


def _snapshot(model: str) -> str:
    return snapshot_download(model, local_files_only=True)


def _build_on_meta(path: str, raw: dict) -> torch.nn.Module:
    """The model class and build arguments NeMoAutoModelForCausalLM.from_pretrained uses
    in the worker, with no weights."""
    with init_empty_weights():
        _, model = model_init._init_model(NeMoAutoModelForCausalLM, path, "sdpa", torch.bfloat16, None, False,
                                          **build_overrides(raw))
    return model


@pytest.mark.parametrize("train_attn,train_mlp", FLAGS, ids=lambda f: str(f))
@pytest.mark.parametrize("model", MODELS)
def test_lora_elements_are_tinkers(model, train_attn, train_mlp):
    path = _snapshot(model)
    raw = read_raw_hf_config(path)
    lora = model_lora(lora_spec(LoraConfig(rank=1, train_attn=train_attn, train_mlp=train_mlp), "adamw"), raw)
    built = _build_on_meta(path, raw)
    apply_lora(built, lora)
    adapted = {n: p.numel() for n, p in built.named_parameters() if ".lora_A." in n or ".lora_B." in n}
    want = hyperparam_utils.get_lora_param_count(model, lora_rank=1, train_attn=train_attn, train_mlp=train_mlp,
                                                 train_unembed=False)
    print(f"{model} train_attn={train_attn} train_mlp={train_mlp}: {sum(adapted.values())} elements, Tinker {want}")
    assert sum(adapted.values()) == want
    for name in adapted:
        assert not set(name.split(".")) & set(NOT_ADAPTED), name


def test_qwen3_5_in_proj_qkv_is_sliced_by_the_config():
    raw = read_raw_hf_config(_snapshot("Qwen/Qwen3.5-9B-Base"))
    lora = model_lora(lora_spec(LoraConfig(rank=4), "adamw"), raw)
    assert dict(lora.slices) == {"in_proj_qkv": (2048, 2048, 4096)}
    dense = model_lora(lora_spec(LoraConfig(rank=4), "adamw"), read_raw_hf_config(_snapshot("Qwen/Qwen3-8B")))
    assert dense.slices == ()


def test_in_proj_qkv_slices_are_independent_lora_pairs():
    raw = read_raw_hf_config(_snapshot("Qwen/Qwen3.5-9B-Base"))
    lora = model_lora(lora_spec(LoraConfig(rank=4, alpha=8), "adamw"), raw)
    widths = dict(lora.slices)["in_proj_qkv"]
    torch.manual_seed(0)
    linear = torch.nn.Linear(raw["text_config"]["hidden_size"], sum(widths), bias=False)
    module = patch_sliced(linear, widths, peft_config(lora))
    assert isinstance(module, SlicedLinearLoRA) and not module.weight.requires_grad
    for b in module.lora_B:
        torch.nn.init.normal_(b.weight)   # B starts at 0; a nonzero B makes every slice's term visible
    x = torch.randn(2, 3, linear.in_features)
    out = module(x)
    scale = lora.alpha / lora.rank
    want = torch.nn.functional.linear(x, module.weight) + torch.cat(
        [torch.nn.functional.linear(torch.nn.functional.linear(x, a.weight) * scale, b.weight)
         for a, b in zip(module.lora_A, module.lora_B)], dim=-1)
    assert (out - want).abs().max().item() <= 1e-6

    module.train()
    out[..., :widths[0]].sum().backward()   # reads the q slice only
    q_a, q_b = module.lora_A[0].weight.grad, module.lora_B[0].weight.grad
    assert q_a is not None and q_b is not None and q_a.abs().sum() > 0 and q_b.abs().sum() > 0
    for i in (1, 2):   # k, v
        for p in (module.lora_A[i].weight, module.lora_B[i].weight):
            assert p.grad is None or torch.count_nonzero(p.grad) == 0


@pytest.mark.parametrize("raw", [{"model_type": "jamba", "attn_layer_period": 8}, {"model_type": "nemotron_h"}, {}])
def test_a_model_without_layer_types_outside_the_dense_families_is_refused(raw):
    with pytest.raises(ValueError, match="declares no layer_types"):
        model_lora(lora_spec(LoraConfig(rank=4), "adamw"), raw)


def test_a_per_slice_lora_refuses_dora():
    lora = model_lora(lora_spec(LoraConfig(rank=4), "adamw"), read_raw_hf_config(_snapshot("Qwen/Qwen3.5-9B-Base")))
    config = peft_config(lora)
    config.use_dora = True
    with pytest.raises(ValueError, match="DoRA"):
        patch_sliced(torch.nn.Linear(8, 8, bias=False), (4, 4), config)
