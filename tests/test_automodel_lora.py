"""LoraConfig -> LoraSpec: Tinker's target flags, alpha default, tenant seed."""
import pytest

from tinkercloud.training.backends.automodel.lora import ATTN_MODULES, MLP_MODULES, lora_spec, tenant_seed
from tinkercloud.training.models.requests import LoraConfig


def test_defaults_target_attention_and_mlp_and_alpha_is_rank():
    spec = lora_spec(LoraConfig(rank=16))
    assert spec.target_modules == ATTN_MODULES + MLP_MODULES
    assert (spec.rank, spec.alpha, spec.dropout) == (16, 16, 0.0)
    assert spec.signature == (16, 16, ATTN_MODULES + MLP_MODULES)


@pytest.mark.parametrize("attn,mlp,want", [(True, False, ATTN_MODULES), (False, True, MLP_MODULES)])
def test_flags_select_targets(attn, mlp, want):
    assert lora_spec(LoraConfig(rank=8, alpha=32, train_attn=attn, train_mlp=mlp)).target_modules == want


def test_train_unembed_is_accepted_and_not_applied():
    on = lora_spec(LoraConfig(rank=8, train_unembed=True))
    off = lora_spec(LoraConfig(rank=8, train_unembed=False))
    assert on == off


def test_rank_zero_and_no_targets_are_refused():
    with pytest.raises(ValueError, match="rank is 0"):
        lora_spec(LoraConfig(rank=0))
    with pytest.raises(ValueError, match="neither attention nor MLP"):
        lora_spec(LoraConfig(rank=8, train_attn=False, train_mlp=False))


def test_tenant_seed_is_the_request_seed_else_stable_per_model():
    assert tenant_seed("m1", 7) == 7
    assert tenant_seed("m1", None) == tenant_seed("m1", None) != tenant_seed("m2", None)
    assert 0 <= tenant_seed("m1", None) < 2**63
