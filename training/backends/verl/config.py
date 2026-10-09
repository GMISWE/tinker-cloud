"""verl backend knobs that used to arrive on create_model (specs/025, D17)."""
from pydantic import Field

from ..env_config import EnvConfig


class VerlConfig(EnvConfig):
    debug_train_only: bool = Field(False, description="Boot without the rollout engine")
    max_seq_len: int = Field(2048, ge=1, description="Rollout prompt_length and token budgets")

    ENV = {
        "debug_train_only": "TINKERCLOUD_VERL_DEBUG_TRAIN_ONLY",
        "max_seq_len": "TINKERCLOUD_VERL_MAX_SEQ_LEN",
    }
