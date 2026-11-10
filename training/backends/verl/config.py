"""verl backend knobs that used to arrive on create_model (specs/025, D17)."""
from pydantic import Field

from ..env_config import EnvConfig


class VerlConfig(EnvConfig):
    models_per_gpu: int = Field(
        1, ge=1,
        description="Models resident on each GPU at once (a student and its teachers); every model keeps its "
                    "weights there, so the limit is GPU memory")
    max_seq_len: int = Field(2048, ge=1, description="Rollout prompt_length and token budgets")

    ENV = {
        "models_per_gpu": "TINKERCLOUD_VERL_MODELS_PER_GPU",
        "max_seq_len": "TINKERCLOUD_VERL_MAX_SEQ_LEN",
    }
