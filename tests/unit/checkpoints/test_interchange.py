"""Unit tests for the cross-backend HF PEFT adapter interchange.

Filesystem-level contract only (no engine): publish/stage round-trip,
atomicity of publish, PEFT key normalization, and the "native artifact wins"
rule. Where a root lives is the store's (test_checkpoint_store.py).
Design: specs/007-q5-migration/HANDOFF.md §2.1.
"""
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinkercloud.training.checkpoints import interchange as ci

BARE_KEY = "model.layers.0.self_attn.q_proj.lora_A.weight"


def _write_adapter(d, r=32, alpha=32, key=BARE_KEY, value=1.0):
    """A minimal but REAL adapter: the publish path parses safetensors."""
    os.makedirs(d, exist_ok=True)
    save_file(
        {key: torch.full((2, 2), value)},
        os.path.join(d, ci.ADAPTER_WEIGHTS_FILE),
        metadata={"format": "pt"},
    )
    with open(os.path.join(d, ci.ADAPTER_CONFIG_FILE), "w") as f:
        json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha}, f)
    return d


class TestSingleTenantNativePublish:
    """save_checkpoint must publish the interchange adapter in single-tenant
    mode too — the hook rode the pool-only adapter_save_dir since the
    multi-LoRA merge, silently breaking the miles->X interchange edge."""

    def test_newest_iter_adapter_is_published(self, tmp_path):
        from tinkercloud.training.backends.miles.backend import _publish_native_adapter

        save_root = tmp_path / "save"
        for i, val in ((1, 1.0), (2, 2.0)):
            d = save_root / f"iter_{i:07d}" / "adapter"
            os.makedirs(d)
            torch.save(
                {"model.layers.0.self_attn.q_proj.lora_A.weight": torch.full((2, 2), val)},
                os.path.join(d, ci.ADAPTER_WEIGHTS_TORCH_FILE),
            )
            os.utime(d, (i * 1000, i * 1000))

        class Args:
            save = str(save_root)

        root = str(tmp_path / "ckpt")
        _publish_native_adapter(Args(), root)
        out = load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE))
        assert float(next(iter(out.values()))[0, 0]) == 2.0

    def test_missing_save_root_is_a_warning_not_a_crash(self, tmp_path):
        from tinkercloud.training.backends.miles.backend import _publish_native_adapter

        class Args:
            save = str(tmp_path / "nonexistent")

        _publish_native_adapter(Args(), str(tmp_path / "ckpt"))
        assert not os.path.exists(os.path.join(tmp_path / "ckpt", ci.HF_ADAPTER_DIRNAME))


class TestPeftKeyNormalization:
    def test_bare_megatron_keys_gain_the_peft_prefix(self, tmp_path):
        """Miles/Bridge exports bare module paths; PeftModel.from_pretrained
        silently loads NOTHING from those (measured: corr 0.806 vs 0.99964)."""
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "src")), root)
        keys = list(load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE)))
        assert keys == [ci.PEFT_PREFIX + BARE_KEY]

    def test_already_peft_keyed_weights_are_left_alone(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        src = _write_adapter(str(tmp_path / "src"), key=ci.PEFT_PREFIX + BARE_KEY)
        ci.export_hf_adapter(src, root)
        keys = list(load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE)))
        assert keys == [ci.PEFT_PREFIX + BARE_KEY]   # not double-prefixed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
