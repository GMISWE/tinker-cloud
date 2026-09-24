"""NeMo RL backend knobs: every NEMORL_* / NRL_* variable the backend reads.

`backend_overrides` keys that are not fields here are deep-merged into the
NeMo RL MasterConfig dict by the argument builder (raw config overrides).
"""
from typing import Any, Dict, Optional, Tuple

from pydantic import Field

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
    sample_max_in_flight_per_worker: int = Field(512, description="Sample calls in flight per vLLM worker; Ray queues async-actor tasks past ~1000")

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
        "sample_max_in_flight_per_worker": "NEMORL_SAMPLE_MAX_IN_FLIGHT_PER_WORKER",
    }

    @classmethod
    def split_overrides(cls, overrides: Optional[Dict[str, Any]]) -> Tuple["NemoRLConfig", Dict[str, Any]]:
        """(config from env + known override keys, remaining raw NeMo RL config overrides)."""
        known = {k: v for k, v in (overrides or {}).items() if k in cls.model_fields}
        raw = {k: v for k, v in (overrides or {}).items() if k not in cls.model_fields}
        return cls.from_env(known), raw
