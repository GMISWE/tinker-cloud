"""create_model(resume_from) on Miles is a weights-only load: the builder
must pair Megatron's --load with --no-load-optim / --no-load-rng / --finetune,
or the resume silently restores optimizer state (a full resume is
load_weights(optimizer=true), which goes through the train group instead).
--save is the model's own native area, never the shared default."""
from argparse import Namespace

import pytest

from test_miles_pack_length import MODEL_CONFIG, _load_builder_module


@pytest.fixture(scope="module")
def builder_mod():
    return _load_builder_module()


def _configure(builder_mod, load_dir, save_dir=None):
    b = builder_mod.MilesArgumentBuilder(default_save_dir="/tmp/ckpt")
    return b._configure_model_args(
        Namespace(), base_model="/tmp/model", megatron_checkpoint_path="/tmp/mcore",
        lora_config={"rank": 8}, debug_train_only=False, load_dir=load_dir,
        model_config=MODEL_CONFIG, parallel_config={"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "context_parallel_size": 1, "num_gpus": 1}, save_dir=save_dir,
    )


def test_resume_at_create_is_weights_only(builder_mod):
    args = _configure(builder_mod, "/data/ckpt/m/native/iter_0000003")
    assert args.load == "/data/ckpt/m/native/iter_0000003"
    assert args.no_load_optim is True and args.no_load_rng is True and args.finetune is True


def test_no_resume_sets_no_load(builder_mod):
    args = _configure(builder_mod, None)
    assert not getattr(args, "load", None)


def test_save_is_the_models_native_area(builder_mod):
    assert _configure(builder_mod, None, save_dir="/data/ckpt/m/native").save == "/data/ckpt/m/native"
    assert _configure(builder_mod, None).save == "/tmp/ckpt"
