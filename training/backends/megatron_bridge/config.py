"""Megatron-Bridge (Evo2 classifier) recipe knobs.

These used to travel in the SDK's head_config dict: server-side file paths and
training-loop settings smuggled through the client (specs/025, D17). They are
operator configuration now.
"""
from typing import Optional

from pydantic import Field

from ..env_config import EnvConfig


class MegatronBridgeConfig(EnvConfig):
    base_ckpt_dir: Optional[str] = Field(None, description="Evo2 base checkpoint dir; None = the model's base_model path")
    tokenizer_path: Optional[str] = Field(None, description="Tokenizer dir; None = the recipe default")
    train_jsonl: Optional[str] = Field(None, description="Recipe train split (recipe-internal eval only)")
    val_jsonl: Optional[str] = Field(None, description="Recipe validation split")
    test_jsonl: Optional[str] = Field(None, description="Recipe test split")
    result_dir: Optional[str] = Field(None, description="Recipe result dir; None = the model's native_root")
    model_size: str = Field("evo2_1b_base", description="Evo2 size preset")
    seq_length: int = Field(1024, ge=1, description="Classifier and backbone sequence length")
    pool: str = Field("mean", description="Sequence pooling for the head")
    classifier_dropout: float = Field(0.1, ge=0.0, le=1.0)
    lr: float = Field(5e-4, gt=0)
    min_lr: float = Field(5e-5, ge=0)
    warmup_iters: int = Field(30, ge=0)
    train_iters: int = Field(1000, ge=1)
    global_batch_size: int = Field(32, ge=1)
    micro_batch_size: int = Field(8, ge=1)

    ENV = {
        "base_ckpt_dir": "MEGATRON_BRIDGE_BASE_CKPT_DIR",
        "tokenizer_path": "MEGATRON_BRIDGE_TOKENIZER_PATH",
        "train_jsonl": "MEGATRON_BRIDGE_TRAIN_JSONL",
        "val_jsonl": "MEGATRON_BRIDGE_VAL_JSONL",
        "test_jsonl": "MEGATRON_BRIDGE_TEST_JSONL",
        "result_dir": "MEGATRON_BRIDGE_RESULT_DIR",
        "model_size": "MEGATRON_BRIDGE_MODEL_SIZE",
        "seq_length": "MEGATRON_BRIDGE_SEQ_LENGTH",
        "pool": "MEGATRON_BRIDGE_POOL",
        "classifier_dropout": "MEGATRON_BRIDGE_CLASSIFIER_DROPOUT",
        "lr": "MEGATRON_BRIDGE_LR",
        "min_lr": "MEGATRON_BRIDGE_MIN_LR",
        "warmup_iters": "MEGATRON_BRIDGE_WARMUP_ITERS",
        "train_iters": "MEGATRON_BRIDGE_TRAIN_ITERS",
        "global_batch_size": "MEGATRON_BRIDGE_GLOBAL_BATCH_SIZE",
        "micro_batch_size": "MEGATRON_BRIDGE_MICRO_BATCH_SIZE",
    }
