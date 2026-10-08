"""NeMo RL backend knobs: every NEMORL_* / NRL_* variable the backend reads.

`backend_overrides` keys that are not fields here are deep-merged into the
NeMo RL MasterConfig dict by the argument builder (raw config overrides).
"""
from typing import Any, Dict, Optional, Tuple

from pydantic import Field, model_validator

from ..env_config import EnvConfig


class NemoRLConfig(EnvConfig):
    max_seq_len_cap: int = Field(32768, description="Ceiling when sizing max_seq_len up to the model's context")
    default_tp: Optional[int] = Field(None, description="Force tensor parallel size")
    train_mbs: int = Field(1, description="Static micro-batch size when dynamic batching is off")
    train_mb_tokens: Optional[int] = Field(None, description="Dynamic-batch token budget; 0 disables; None = min(max_seq_len, 8192)")
    megatron: bool = Field(False, description="Use the Megatron worker instead of DTensor")
    megatron_lr: float = Field(2e-4, description="Megatron scheduler lr0")
    megatron_lr_decay_iters: int = Field(222, description="Megatron scheduler decay iterations")
    megatron_gbs: int = Field(128, description="Megatron config train_global_batch_size (scheduler units)")
    megatron_a_init: str = Field("xavier", description="Megatron LoRA A init method")
    megatron_precision_aware: bool = Field(True, description="use_precision_aware_optimizer")
    refit_buffer_memory_ratio: float = Field(0.3, description="Share of free GPU memory for the IPC refit buffer")
    vllm_max_connections: Optional[int] = Field(None, description="Concurrent connections per vLLM worker server; None = unbounded (engine queue)")
    # specs/021: vLLM either shares the training GPUs (sleeps during training,
    # IPC refit) or owns `inference_gpus` of create_model's num_gpus (NCCL refit).
    colocated: bool = Field(True, description="vLLM shares the training GPUs; False = separate inference GPUs")
    inference_gpus: Optional[int] = Field(None, description="GPUs vLLM owns when colocated=False, taken from num_gpus")
    inference_tp: int = Field(1, description="vLLM tensor parallel size when colocated=False; leaders = inference_gpus / inference_tp")

    ENV = {
        "max_seq_len_cap": "TINKERCLOUD_MAX_SEQ_LEN_CAP",
        "default_tp": "NEMORL_DEFAULT_TP",
        "train_mbs": "NEMORL_TRAIN_MBS",
        "train_mb_tokens": "NEMORL_TRAIN_MB_TOKENS",
        "megatron": "NEMORL_MEGATRON",
        "megatron_lr": "NEMORL_MEGATRON_LR",
        "megatron_lr_decay_iters": "NEMORL_MEGATRON_LR_DECAY_ITERS",
        "megatron_gbs": "NEMORL_MEGATRON_GBS",
        "megatron_a_init": "NEMORL_MEGATRON_A_INIT",
        "megatron_precision_aware": "NEMORL_MEGATRON_PRECISION_AWARE",
        "refit_buffer_memory_ratio": "NRL_REFIT_BUFFER_MEMORY_RATIO",
        "vllm_max_connections": "TINKERCLOUD_VLLM_MAX_CONNECTIONS",
        "colocated": "NEMORL_COLOCATED",
        "inference_gpus": "NEMORL_INFERENCE_GPUS",
        "inference_tp": "NEMORL_INFERENCE_TP",
    }

    @property
    def inference_gpu_count(self) -> int:
        """0 while colocated; the validated count otherwise."""
        if self.colocated:
            return 0
        assert self.inference_gpus is not None  # _check_inference_split
        return self.inference_gpus

    @model_validator(mode="after")
    def _check_inference_split(self) -> "NemoRLConfig":
        if self.colocated:
            if self.inference_gpus is not None:
                raise ValueError("inference_gpus has no meaning while colocated=True")
            return self
        if self.inference_gpus is None or self.inference_gpus < 1:
            raise ValueError("colocated=False needs inference_gpus >= 1")
        if self.inference_tp < 1 or self.inference_gpus % self.inference_tp != 0:
            raise ValueError(
                f"inference_gpus={self.inference_gpus} is not a multiple of inference_tp={self.inference_tp}"
            )
        return self

    @classmethod
    def split_overrides(cls, overrides: Optional[Dict[str, Any]]) -> Tuple["NemoRLConfig", Dict[str, Any]]:
        """(config from env + known override keys, remaining raw NeMo RL config overrides)."""
        known = {k: v for k, v in (overrides or {}).items() if k in cls.model_fields}
        raw = {k: v for k, v in (overrides or {}).items() if k not in cls.model_fields}
        return cls.from_env(known), raw
