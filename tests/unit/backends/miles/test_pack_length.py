"""The miles pack length must clear the kernel's batch-invariance threshold.

`get_batch` rounds each microbatch's packed token stream up to a multiple of
`tp_size * data_pad_size_multiplier`, so that multiplier decides the M handed
to every GEMM in the round. `linear_fc2` (K=4864 -> N=896, the model's only
large-K GEMM) selects a different cuBLAS K-reduction below M=352, which makes
the client's own segmentation pick the kernel: at the stock 128, fb(3) packs
to 256 and fb(5) to 384, and a datum's returned logprobs move by up to 0.38
nats with how the client chunked its round.

Measured on the cluster (specs/014-gate-suite/INVESTIGATION-miles-split.md):
SPLIT_INV goes 5.169e-02 -> 2.882e-07 and the forward becomes bit-identical
once every pack clears the threshold.

Two things are asserted, and the second is the one that has bitten before:
the value must clear 352, and it must ride the CLI list — post-parse
namespace mutations never reach the actors (the same root cause as the
dropped AdamParams and the ignored checkpoint_path).
"""

import pytest

# The measured switch point for linear_fc2 on H200 / TE 2.17 / Qwen2.5-0.5B.
# Not a constant of nature — re-measure with probes/bi_fc2_bisect.py when the
# model shape, GPU or cuBLAS changes.
FC2_BATCH_INVARIANCE_THRESHOLD = 352


def _load_builder_module():
    from tinkercloud.training.backends.miles import builder
    return builder


# Qwen2.5-0.5B, the shape every number in the investigation was measured on.
MODEL_CONFIG = {
    "num_layers": 24,
    "hidden_size": 896,
    "ffn_hidden_size": 4864,
    "num_attention_heads": 14,
    "num_query_groups": 2,
    "kv_channels": 64,
    "vocab_size": 151936,
    "norm_epsilon": 1e-6,
    "rotary_base": 1000000,
    "tie_word_embeddings": True,
}


def _build(builder_mod):
    """The CLI list the builder hands miles' parse_args, for a small model."""
    return builder_mod.MilesArgumentBuilder(
        default_save_dir="/tmp/ckpt"
    )._build_minimal_args(
        hf_model_path="/tmp/model",
        model_config=MODEL_CONFIG,
        tp_size=1,
        pp_size=1,
        cp_size=1,
        megatron_checkpoint_path="/tmp/mcore",
    )


@pytest.fixture(scope="module")
def cli():
    return _build(_load_builder_module())


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
    cli = _build(_load_builder_module())
    assert _flag_value(cli, "--data-pad-size-multiplier") == "1024"
