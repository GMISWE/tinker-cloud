"""Miles backend knobs: every SLIME_* / TINKERCLOUD_MILES_* variable the backend reads."""
from typing import Optional

from pydantic import Field

from ..env_config import EnvConfig

# Tinker's importance_sampling is unclipped. The IS ratio is >= 0, so a lower
# bound of 0 (eps 1.0) and an upper bound of 1e6 cannot bind. A ppo deployment
# overrides via SLIME_EPS_CLIP / SLIME_EPS_CLIP_HIGH (backend._check_miles_clip_config).
NO_CLIP_EPS = 1.0
NO_CLIP_EPS_HIGH = 1.0e6


class MilesConfig(EnvConfig):
    # Multi-LoRA pool (explicit opt-in; needs a LoRA rank on the model)
    multilora_slots: int = Field(0, description="Adapter slots in the pool; 0 = single-tenant boot")
    train_gpus: int = Field(0, description="Training GPUs in pool mode; 0 = half of the GPUs")
    cobatch_max_samples: int = Field(0, description="Co-batch cap across tenants; 0 = off")
    cobatch_reorder: bool = Field(False, description="Reorder co-batched samples by length")
    cobatch_e0_tokens: Optional[int] = Field(None, description="E0 guard threshold; None = calibration registry")
    # Parallelism overrides (None = auto-detect from model size / sequence length)
    tp: Optional[int] = Field(None, description="Force tensor parallel size after auto-detect")
    default_tp: Optional[int] = Field(None, description="Auto-detect override for TP")
    default_cp: Optional[int] = Field(None, description="Auto-detect override for CP")
    rlve_tp: int = Field(2, description="TP in RLVE mode")
    rlve_cp: int = Field(2, description="CP in RLVE mode")
    # Megatron / Slime CLI values
    data_pad_size_multiplier: int = Field(512, description="--data-pad-size-multiplier")
    advantage_estimator: str = Field("grpo", description="--advantage-estimator")
    eps_clip: float = Field(NO_CLIP_EPS, description="--eps-clip; 1.0 = lower bound 0, cannot bind")
    eps_clip_high: float = Field(NO_CLIP_EPS_HIGH, description="--eps-clip-high; 1e6 cannot bind")
    entropy_coef: float = Field(0.0, description="--entropy-coef")
    sglang_deterministic: bool = Field(False, description="Boot SGLang with deterministic inference (per-request seeds)")
    weight_decay: float = Field(0.0, description="--weight-decay")
    no_offload: bool = Field(False, description="Drop --colocate/--offload (attribution arm)")
    use_kl_loss: bool = Field(False, description="--use-kl-loss (needs a real reference model)")
    kl_loss_coef: float = Field(0.1, description="--kl-loss-coef")
    lora_a_init: Optional[str] = Field(None, description="lora_A_init_method override")
    dyn_batch: bool = Field(True, description="use_dynamic_batch_size")
    max_tokens_per_gpu: Optional[int] = Field(None, description="max_tokens_per_gpu; None = max(8192, max_seq_len)")
    enable_wandb: bool = Field(False, description="Slime-side wandb logging when the client sends none")

    ENV = {
        "multilora_slots": "TINKERCLOUD_MILES_MULTILORA_SLOTS",
        "train_gpus": "TINKERCLOUD_MILES_TRAIN_GPUS",
        "cobatch_max_samples": "TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES",
        "cobatch_reorder": "TINKERCLOUD_MILES_COBATCH_REORDER",
        "cobatch_e0_tokens": "TINKERCLOUD_MILES_COBATCH_E0_TOKENS",
        "tp": "SLIME_TP",
        "default_tp": "SLIME_DEFAULT_TP",
        "default_cp": "SLIME_DEFAULT_CP",
        "rlve_tp": "SLIME_RLVE_TP",
        "rlve_cp": "SLIME_RLVE_CP",
        "data_pad_size_multiplier": "SLIME_DATA_PAD_MULT",
        "advantage_estimator": "SLIME_ADVANTAGE_ESTIMATOR",
        "eps_clip": "SLIME_EPS_CLIP",
        "eps_clip_high": "SLIME_EPS_CLIP_HIGH",
        "entropy_coef": "SLIME_ENTROPY_COEF",
        "sglang_deterministic": "SLIME_SGLANG_DETERMINISTIC",
        "weight_decay": "SLIME_WEIGHT_DECAY",
        "no_offload": "SLIME_NO_OFFLOAD",
        "use_kl_loss": "SLIME_USE_KL_LOSS",
        "kl_loss_coef": "SLIME_KL_LOSS_COEF",
        "lora_a_init": "SLIME_LORA_A_INIT",
        "dyn_batch": "SLIME_DYN_BATCH",
        "max_tokens_per_gpu": "SLIME_MAX_TOKENS_PER_GPU",
        "enable_wandb": "SLIME_ENABLE_WANDB",
    }
