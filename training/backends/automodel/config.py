"""Automodel backend knobs: every AUTOMODEL_* variable the backend reads.

A training group is shared by every LoRA tenant with the same base model and
LoRA signature, and one vLLM serves every tenant of a base model, so their
sizes are server configuration: no request may set them. Fields without a
default are required; a server started without them fails at startup.
"""
from pydantic import Field, model_validator

from ..env_config import EnvConfig

# The ranks vLLM 0.29 accepts for --max-lora-rank (vllm/config/lora.py). An
# int field checked against this set, not a Literal: env values arrive as
# strings and pydantic does not coerce a string into an int Literal.
VLLM_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


class AutomodelConfig(EnvConfig):
    train_gpus: int = Field(ge=1, description="FSDP2 ranks (one GPU each) per training group")
    max_loras: int = Field(ge=1, description="vLLM --max-loras: adapters resident on the GPU at once")
    max_lora_rank: int = Field(description="vLLM --max-lora-rank; create_model refuses a larger LoRA rank")
    max_cpu_loras: int = Field(ge=1, description="vLLM --max-cpu-loras: adapters cached in host memory")

    ENV = {
        "train_gpus": "AUTOMODEL_TRAIN_GPUS",
        "max_loras": "AUTOMODEL_MAX_LORAS",
        "max_lora_rank": "AUTOMODEL_MAX_LORA_RANK",
        "max_cpu_loras": "AUTOMODEL_MAX_CPU_LORAS",
    }

    @model_validator(mode="after")
    def _check_lora_limits(self) -> "AutomodelConfig":
        if self.max_lora_rank not in VLLM_MAX_LORA_RANKS:
            raise ValueError(f"max_lora_rank={self.max_lora_rank} is not one vLLM accepts: {VLLM_MAX_LORA_RANKS}")
        # vLLM refuses this at boot; failing here names the variable instead
        if self.max_cpu_loras < self.max_loras:
            raise ValueError(f"max_cpu_loras={self.max_cpu_loras} must be >= max_loras={self.max_loras}")
        return self
