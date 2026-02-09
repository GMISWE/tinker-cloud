"""
NeMo RL argument builder — maps HuggingFace model config + Tinker
create_model args to NeMo RL PolicyConfig dict.

Returns (config_dict, hf_path) similar to MilesArgumentBuilder.
"""
import logging
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder

logger = logging.getLogger(__name__)


class NemoRLArgumentBuilder(ArgumentBuilder):
    """Builds NeMo RL PolicyConfig + loss config from Tinker API parameters."""

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
    ) -> Any:
        """
        Build NeMo RL config from Tinker API parameters.

        Returns:
            Tuple of (config_dict, hf_path) where config_dict contains:
            - "policy": PolicyConfig dict
            - "loss_fn": ClippedPGLossConfig dict
            - "cluster": cluster config (bundle_ct_per_node_list, etc.)
        """
        debug_train_only = kwargs.get("debug_train_only", False)
        max_batch_size = kwargs.get("max_batch_size", 4096)
        max_seq_len = kwargs.get("max_seq_len", 2048)
        rlve_config = kwargs.get("rlve_config")
        wandb_config = kwargs.get("wandb_config")

        hf_path = base_model

        # Warn about Miles-only RLVE server-side features
        if rlve_config and rlve_config.get("enabled", False):
            miles_only_keys = [
                "custom_prompt_preprocessor", "answer_marker_type",
                "difficulty_sliding_window_size", "min_metric_to_increase_difficulty",
                "min_prompts_before_difficulty_check", "over_sampling_batch_size",
                "use_dynamic_sampling_filter", "partial_rollout", "balance_data",
            ]
            unsupported = [k for k in miles_only_keys if k in rlve_config]
            if unsupported:
                logger.warning(
                    "RLVE server-side args ignored on NeMo RL backend (Miles-only): %s. "
                    "RLVE in Tinker mode is client-driven — these settings have no effect.",
                    unsupported,
                )

        # Parallelism config
        tp_size = 1
        pp_size = 1
        cp_size = 1
        dp_size = num_gpus  # default: all GPUs for data parallelism
        if parallelism:
            tp_size = parallelism.get("tensor_parallel", 1)
            pp_size = parallelism.get("pipeline_parallel", 1)
            cp_size = parallelism.get("context_parallel", 1)
            model_parallel = tp_size * pp_size * cp_size
            dp_size = max(1, num_gpus // model_parallel)

        # Micro-batch size calculation
        # NeMo RL train_global_batch_size = total samples per train() call
        # train_micro_batch_size = samples per GPU per forward/backward pass
        train_global_batch_size = max_batch_size
        train_micro_batch_size = max(1, train_global_batch_size // dp_size)

        # Policy config (maps to NeMo RL PolicyConfig TypedDict)
        policy_config = {
            "model_name": base_model,
            "tokenizer": {
                "name": base_model,
            },
            "train_global_batch_size": train_global_batch_size,
            "train_micro_batch_size": train_micro_batch_size,
            "logprob_batch_size": train_micro_batch_size,
            "precision": "bfloat16",
            "max_total_sequence_length": max_seq_len,
            "make_sequence_length_divisible_by": 1,
            "max_grad_norm": 1.0,
            "offload_optimizer_for_logprob": False,
            # Use DTensor V2 backend (recommended by NeMo RL)
            "dtensor_cfg": {
                "enabled": True,
                "_v2": True,
                "cpu_offload": False,
                "sequence_parallel": False,
                "activation_checkpointing": False,
                "tensor_parallel_size": tp_size,
                "context_parallel_size": cp_size,
                "custom_parallel_plan": None,
            },
            "megatron_cfg": {
                "enabled": False,
            },
            "dynamic_batching": {
                "enabled": False,
            },
            "sequence_packing": {
                "enabled": False,
            },
            # Generation config (colocated by default)
            "generation": {
                "backend": "vllm",
                "max_new_tokens": max_seq_len,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "stop_token_ids": None,
                "stop_strings": None,
                "vllm_cfg": {
                    "async_engine": False,
                    "precision": "bfloat16",
                    "kv_cache_dtype": "auto",
                    "tensor_parallel_size": tp_size,
                    "pipeline_parallel_size": pp_size,
                    "expert_parallel_size": 1,
                    "gpu_memory_utilization": 0.6,
                    "max_model_len": max_seq_len,
                    "enforce_eager": False,
                    "use_deep_gemm": False,
                    "num_last_layers_in_bf16": 0,
                    "num_first_layers_in_bf16": 0,
                    "enable_vllm_metrics_logger": False,
                    "vllm_metrics_logger_interval": 0.5,
                },
                "vllm_kwargs": {},
                "colocated": {
                    "enabled": True,
                },
            },
            # Optimizer (AdamW defaults matching Miles)
            "optimizer": {
                "name": "torch.optim.AdamW",
                "kwargs": {
                    "lr": 5.0e-6,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "foreach": False,
                    "fused": False,
                },
            },
        }

        # LoRA config
        if lora_config and lora_config.get("rank", 0) > 0:
            policy_config["dtensor_cfg"]["lora_cfg"] = {
                "enabled": True,
                "dim": lora_config.get("rank", 8),
                "alpha": lora_config.get("alpha", lora_config.get("rank", 8)),
                "dropout": lora_config.get("dropout", 0.0),
                "dropout_position": "pre",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "exclude_modules": [],
                "lora_A_init": "kaiming",
            }
        else:
            policy_config["dtensor_cfg"]["lora_cfg"] = {"enabled": False}

        # Loss function config (ClippedPGLossConfig for GRPO/PPO)
        loss_fn_config = {
            "reference_policy_kl_penalty": 0.001,
            "reference_policy_kl_type": "k1",
            "kl_input_clamp_value": None,
            "kl_output_clamp_value": None,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "token_level_loss": True,
            "sequence_level_importance_ratios": False,
            "disable_ppo_ratio": False,
            "force_on_policy_ratio": False,
        }

        # Override loss config from rl_config or rlve_config
        if rl_config:
            if "kl_penalty_coef" in rl_config:
                loss_fn_config["reference_policy_kl_penalty"] = rl_config["kl_penalty_coef"]
            if "eps_clip" in rl_config:
                loss_fn_config["ratio_clip_min"] = rl_config["eps_clip"]
                loss_fn_config["ratio_clip_max"] = rl_config["eps_clip"]

        # Cluster config
        cluster_config = {
            "bundle_ct_per_node_list": [num_gpus],
            "num_gpus_per_node": num_gpus,
            "max_colocated_worker_groups": 2,  # policy + generation
        }

        # Checkpointing config (passed to Policy.save_checkpoint)
        checkpointing_config = {
            "model_save_format": "safetensors",
            "save_consolidated": False,
        }
        if checkpoint_config:
            checkpointing_config.update(checkpoint_config)

        # Apply overrides
        config_dict = {
            "policy": policy_config,
            "loss_fn": loss_fn_config,
            "cluster": cluster_config,
            "checkpointing": checkpointing_config,
            "dp_size": dp_size,
            "debug_train_only": debug_train_only,
            "wandb_config": wandb_config,
            "rlve_config": rlve_config,
        }

        # Deep merge user overrides
        if self.overrides:
            _deep_merge(config_dict, self.overrides)

        logger.info(
            "NeMo RL config built: model=%s, num_gpus=%d, tp=%d, dp=%d, gbs=%d, mbs=%d",
            base_model, num_gpus, tp_size, dp_size,
            train_global_batch_size, train_micro_batch_size,
        )

        return config_dict, hf_path


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
