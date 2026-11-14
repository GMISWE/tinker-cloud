"""
veRL argument builder: HF model + tinker knobs -> config dict.

Returns a plain dict; VerlBackend materializes veRL dataclasses lazily on the
server (where verl is importable). FSDP2 strategy first; Megatron later.
"""
import os
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder

# Sequences per GPU in one micro-batch of a training pass and of a forward-only pass. The engine holds the
# full logits of a micro-batch, so its memory grows with the tokens in it; at 1 a pass of 13K-token
# sequences fits an 80 GB GPU with four models on it (`tests/gpu/test_verl_fit.py`). Read 2026-11-12: a
# forward at 4 asked for 30 GiB in one allocation and failed; training at 2 fits with one model and is not
# measured with four. The gradient is the sum over datums either way.
TRAIN_MICRO_BATCH_PER_GPU = 1
INFER_MICRO_BATCH_PER_GPU = 1


class VerlArgumentBuilder(ArgumentBuilder):
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}

    def build_args(
        self,
        base_model: str,
        num_gpus: int = 4,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        parallelism = parallelism or {}
        lora_config = lora_config or {}
        max_seq_len = int(kwargs.get("max_seq_len", 2048))

        hf_path = self.resolve_hf_path(base_model)

        # request payloads carry explicit None values — coalesce, don't .get()-default
        lora_rank = int(lora_config.get("rank") or lora_config.get("lora_rank") or 0)
        # verl's structured config types lora_alpha as int (omegaconf rejects
        # 32.0 at the vLLM server-side merge); keep integral alphas as int
        lora_alpha = float(lora_config.get("alpha") or lora_config.get("lora_alpha") or 32)
        if lora_alpha.is_integer():
            lora_alpha = int(lora_alpha)
        target_modules = lora_config.get("target_modules") or "all-linear"

        cfg: Dict[str, Any] = {
            "hf_path": hf_path,
            "num_gpus": num_gpus,
            "model": {
                "path": hf_path,
                "use_remove_padding": True,
                # FSDP LoRA namespace ONLY (model.lora_rank); the Megatron
                # namespace (model.lora.rank) must stay 0 — duplicated-knob
                # trap, specs/006 q2-plan.md
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
            },
            "engine": {
                "strategy": self.overrides.get("strategy", "fsdp2"),
                "fsdp_size": num_gpus,
                "ulysses_sequence_parallel_size": int(parallelism.get("sp_size", 1)),
                # fixed micro-batch (dynamic_bsz = batch-composition-dependent
                # micro-partitioning; pure-sum makes it E0-safe but keep the
                # default deterministic)
                "use_dynamic_bsz": bool(self.overrides.get("use_dynamic_bsz", False)),
                "micro_batch_size_per_gpu": TRAIN_MICRO_BATCH_PER_GPU,
                "max_token_len_per_gpu": max_seq_len * 4,
                "infer_micro_batch_size_per_gpu": INFER_MICRO_BATCH_PER_GPU,
                "infer_max_token_len_per_gpu": max_seq_len * 8,
                "param_offload": bool(self.overrides.get("param_offload", False)),
                "optimizer_offload": bool(self.overrides.get("optimizer_offload", False)),
            },
            "optim": {
                # placeholder; client owns LR per optim_step (OptimStepParams)
                "lr": 1e-6,
            },
            "max_seq_len": max_seq_len,
            # vLLM rollout servers colocated with the training engine
            # (hybrid timeshare: replicas sleep during training ops, wake +
            # weight-sync for sampling). load_format stays "dummy": the
            # backend syncs the trainer's weights before the first sample.
            "rollout": {
                "name": "vllm",
                "mode": "async",
                "prompt_length": max_seq_len,
                "response_length": int(
                    (rollout_config or {}).get("max_response_len")
                    or self.overrides.get("response_length", 1024)
                ),
                "tensor_model_parallel_size": int((rollout_config or {}).get("tp_size") or 1),
                "gpu_memory_utilization": float(
                    self.overrides.get("rollout_gpu_memory_utilization", 0.5)
                ),
            },
        }
        return cfg

    @staticmethod
    def resolve_hf_path(base_model: str) -> str:
        """The directory `base_model` loads from, or the name itself (an absolute path, else a hub id).

        Raises:
            FileNotFoundError: TINKERCLOUD_MODEL_ROOT is set and holds no such model. The operator
                named where models live; another copy of the same name may differ in its config.
        """
        if os.path.isabs(base_model):
            return base_model
        root = os.environ.get("TINKERCLOUD_MODEL_ROOT")
        if root:
            candidate = os.path.join(root, base_model)
            if os.path.isdir(candidate):
                return candidate
            raise FileNotFoundError(f"TINKERCLOUD_MODEL_ROOT is {root} and holds no directory {base_model}")
        for root in ("/data/models", "/root/models"):
            candidate = os.path.join(root, base_model)
            if os.path.isdir(candidate):
                return candidate
        return base_model
