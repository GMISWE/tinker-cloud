"""The miles argument builder: the CLI it hands miles' parse_args.

The DP gradient reduction must match the optimizer that consumes it, so the builder never pins use_distributed_optimizer.
The pack length must clear the kernel's batch-invariance threshold. create_model(resume_from) is a weights-only load
(--load with --no-load-optim / --no-load-rng / --finetune); --save is the model's own native area. Parallel-config keys
are required, never defaulted."""
import inspect
import re
from argparse import Namespace

import pytest

from tinkercloud.training.backends.miles import builder

from .model_shapes import QWEN25_05B


@pytest.fixture(scope="module")
def builder_src():
    return inspect.getsource(builder)


def test_builder_does_not_pin_use_distributed_optimizer(builder_src):
    hits = re.findall(r"^\s*args\.use_distributed_optimizer\s*=.*$",
                      builder_src, flags=re.MULTILINE)
    assert not hits, (
        "the builder assigns use_distributed_optimizer:\n  "
        + "\n  ".join(h.strip() for h in hits)
        + "\nThe engine derives this flag for the DDP config separately, so a "
          "second derivation here can disagree with it -- and when it does, DDP "
          "reduce-scatters gradients for an optimizer that needs an all-reduce "
          "and every rank steps with a partly unreduced gradient."
    )


def test_builder_does_not_pass_the_flag_on_the_cli(builder_src):
    # The CLI is the other way the service could pin it, and it reaches the
    # actors (unlike post-parse args.X), so it would be just as wrong.
    assert "--use-distributed-optimizer" not in builder_src
    assert "--no-use-distributed-optimizer" not in builder_src


def test_client_seed_rides_the_cli(builder_src):
    # LoraConfig.seed was accepted by the API and read by nobody. Megatron's own
    # default is 1234, so miles was reproducible by accident while ignoring
    # whatever the client declared. specs/014-gate-suite §THE nemo_rl PEER PROBE.
    assert "'--seed'" in builder_src, "the client's seed must reach the engine"
    assert "lora_config or {}).get('seed')" in builder_src, (
        "--seed must carry the client's declared value, not a constant"
    )


FC2_BATCH_INVARIANCE_THRESHOLD = 352


def _build(builder_mod):
    """The CLI list the builder hands miles' parse_args, for a small model."""
    return builder_mod.MilesArgumentBuilder(
        default_save_dir="/tmp/ckpt"
    )._build_minimal_args(
        hf_model_path="/tmp/model",
        model_config=QWEN25_05B,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        megatron_checkpoint_path="/tmp/mcore",
    )


@pytest.fixture(scope="module")
def cli():
    return _build(builder)


def _flag_value(cli, name):
    """The value following `name`, or None if the flag is absent."""
    return cli[cli.index(name) + 1] if name in cli else None


def test_pad_multiplier_is_on_the_cli(cli):
    # Not a post-parse args.X assignment: those never reach the Ray actors,
    # which is how the AdamParams and checkpoint_path defects happened.
    assert "--data-pad-size-multiplier" in cli


def test_pad_multiplier_clears_the_fc2_threshold(cli):
    value = int(_flag_value(cli, "--data-pad-size-multiplier"))
    assert value > FC2_BATCH_INVARIANCE_THRESHOLD, (
        f"pack lengths are multiples of {value}, so a call can pack to {value} "
        f"-- at or below the {FC2_BATCH_INVARIANCE_THRESHOLD}-token switch "
        f"point of linear_fc2, which makes the gradient depend on how the "
        f"client segmented its round"
    )


def test_pad_multiplier_divides_the_dynamic_budget(cli):
    # max_tokens_per_gpu floors at 8192. A multiplier that divides it adds no
    # padding at all to production-sized packs; one that does not taxes every
    # full microbatch.
    value = int(_flag_value(cli, "--data-pad-size-multiplier"))
    assert 8192 % value == 0, f"{value} does not divide the 8192-token budget"


def test_pad_multiplier_is_overridable(monkeypatch):
    monkeypatch.setenv("SLIME_DATA_PAD_MULT", "1024")
    cli = _build(builder)
    assert _flag_value(cli, "--data-pad-size-multiplier") == "1024"


@pytest.fixture(scope="module")
def builder_mod():
    return builder


def _configure(builder_mod, load_dir, save_dir=None):
    b = builder_mod.MilesArgumentBuilder(default_save_dir="/tmp/ckpt")
    return b._configure_model_args(
        Namespace(), base_model="/tmp/model", megatron_checkpoint_path="/tmp/mcore",
        lora_config={"rank": 8}, debug_train_only=False, load_dir=load_dir,
        model_config=QWEN25_05B, parallel_config={"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "context_parallel_size": 1, "num_gpus": 1}, save_dir=save_dir,
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


def test_miles_parallel_config_keys_are_required():
    b = builder.MilesArgumentBuilder(default_save_dir="/tmp/ckpt")
    with pytest.raises(KeyError, match="tensor_parallel_size"):
        b._configure_model_args(
            Namespace(), base_model="/tmp/model", megatron_checkpoint_path="/tmp/mcore",
            lora_config={"rank": 8}, debug_train_only=False, load_dir=None,
            model_config=QWEN25_05B, parallel_config={"num_gpus": 4}, save_dir=None,
        )
