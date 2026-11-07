"""NemoRLArgumentBuilder: the config dict handed to NeMo RL (parallelism, cluster layout, LoRA, RL knobs, overrides)."""
import pytest

# what routers/models.py passes: LoraConfig.dict()
LORA16 = {"rank": 16, "alpha": None, "dropout": 0.0, "seed": None,
          "train_unembed": True, "train_mlp": True, "train_attn": True}


class TestNemoRLArgumentBuilder:
    """Test NemoRLArgumentBuilder config generation."""

    @pytest.fixture(autouse=True)
    def model_config_on_disk(self, monkeypatch):
        # The builder sizes itself from the model's raw config.json (D17); no hub here.
        from tinkercloud.training.backends.nemo_rl import builder as mod
        monkeypatch.setattr(mod, "read_raw_hf_config", lambda base_model: {
            "architectures": ["LlamaForCausalLM"], "max_position_embeddings": 131072,
        })

    @pytest.fixture
    def builder(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        return NemoRLArgumentBuilder()

    def test_build_args_returns_tuple(self, builder):
        config_dict, hf_path = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert isinstance(config_dict, dict)
        assert isinstance(hf_path, str)
        assert hf_path == "meta-llama/Llama-3.1-8B"

    def test_config_has_required_sections(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert "policy" in config_dict
        assert "loss_fn" in config_dict
        assert "cluster" in config_dict

    def test_parallelism_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=8,
            parallelism={"tensor_parallel": 2, "pipeline_parallel": 1},
        )
        assert config_dict["policy"]["dtensor_cfg"]["tensor_parallel_size"] == 2
        # The train DP is NeMo RL's to derive (handle.train_dp); the builder
        # only guarantees the split divides.
        assert "dp_size" not in config_dict
        assert config_dict["cluster"]["bundle_ct_per_node_list"] == [8]

    def test_model_parallel_must_divide_train_gpus(self, builder):
        with pytest.raises(ValueError, match="not divisible"):
            builder.build_args(
                base_model="meta-llama/Llama-3.1-8B",
                num_gpus=4,
                parallelism={"tensor_parallel": 3},
            )

    def test_colocated_layout_is_one_cluster(self, builder):
        config_dict, _ = builder.build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=4)
        cluster = config_dict["cluster"]
        assert cluster["bundle_ct_per_node_list"] == [4]
        assert cluster["max_colocated_worker_groups"] == 2
        assert cluster["inference"] is None
        gen = config_dict["policy"]["generation"]
        assert gen["colocated"]["enabled"] is True
        assert gen["colocated"]["resources"] == {"gpus_per_node": None, "num_nodes": None}
        assert config_dict["num_gpus"] == 4

    def test_split_layout_gives_vllm_its_own_cluster(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"colocated": False, "inference_gpus": 2, "inference_tp": 2}, environ={})
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=6, parallelism={"tensor_parallel": 2},
        )
        cluster = config_dict["cluster"]
        assert cluster["bundle_ct_per_node_list"] == [4]            # 6 - 2
        assert cluster["max_colocated_worker_groups"] == 1
        assert cluster["inference"] == {"bundle_ct_per_node_list": [2], "num_gpus_per_node": 2}
        gen = config_dict["policy"]["generation"]
        assert gen["colocated"] == {"enabled": False, "resources": {"gpus_per_node": 2, "num_nodes": 1}}
        assert gen["vllm_cfg"]["tensor_parallel_size"] == 2        # inference_tp, not the trainer's
        assert gen["vllm_cfg"]["pipeline_parallel_size"] == 1
        assert config_dict["policy"]["dtensor_cfg"]["tensor_parallel_size"] == 2

    def test_megatron_trains_at_the_client_lr_without_clipping(self):
        # The worker's set_learning_rate pins each group's max/min lr per step; that
        # holds only under a scheduler without warmup. Grad clipping is off by default.
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        from tinkercloud.training.backends.nemo_rl.loss_config import NO_GRAD_CLIP
        cfg = NemoRLConfig.from_env({"megatron": True}, environ={})
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config=LORA16,
        )
        meg = config_dict["policy"]["megatron_cfg"]
        assert meg["scheduler"]["lr_decay_style"] == "constant"
        assert meg["scheduler"]["lr_warmup_iters"] == 0
        assert meg["optimizer"]["clip_grad"] == NO_GRAD_CLIP
        assert config_dict["policy"]["dtensor_cfg"] == {"enabled": False}

    @pytest.mark.parametrize("flags, targets", [
        ({}, ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "output_layer"]),
        ({"train_unembed": False}, ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]),
        ({"train_attn": False, "train_mlp": False}, ["output_layer"]),
    ])
    def test_megatron_lora_targets_follow_the_sdk_flags(self, flags, targets):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ={})
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config={**LORA16, **flags},
        )
        peft = config_dict["policy"]["megatron_cfg"]["peft"]
        assert peft["target_modules"] == targets
        assert peft["experts_shared_outer_loras"] is True
        assert peft["experts_shared_outer_separate_gate_up"] is True
        assert peft["split_qkv_adapters"] is True

    def test_megatron_lora_refuses_no_target(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ={})
        off = {"train_attn": False, "train_mlp": False, "train_unembed": False}
        with pytest.raises(ValueError, match="all false"):
            NemoRLArgumentBuilder(config=cfg).build_args(
                base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config={**LORA16, **off},
            )

    def test_moe_model_gets_expert_parallel_and_the_providers_dispatcher(self, monkeypatch):
        from tinkercloud.training.backends.nemo_rl import builder as mod
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        monkeypatch.setattr(mod, "read_raw_hf_config", lambda base_model: {
            "architectures": ["GptOssForCausalLM"], "max_position_embeddings": 131072,
            "num_local_experts": 128,
        })
        cfg = NemoRLConfig.from_env({"megatron": True, "megatron_ep": 8}, environ={})
        config_dict, _ = mod.NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="openai/gpt-oss-120b", num_gpus=8, lora_config=LORA16,
        )
        meg = config_dict["policy"]["megatron_cfg"]
        assert meg["expert_model_parallel_size"] == 8
        assert meg["expert_tensor_parallel_size"] == 1
        assert meg["moe_token_dispatcher_type"] == "alltoall"
        assert meg["moe_permute_fusion"] is True
        assert config_dict["policy"]["generation"]["vllm_kwargs"]["moe_backend"] == "triton"

    @pytest.mark.parametrize("environ, expected", [({}, False), ({"NEMORL_MEGATRON_RECOMPUTE": "1"}, True)])
    def test_megatron_recompute_follows_its_knob(self, environ, expected):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ=environ)
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config=LORA16,
        )
        meg = config_dict["policy"]["megatron_cfg"]
        assert meg["activation_checkpointing"] is expected
        assert meg["recompute_granularity"] == "full"

    @pytest.mark.parametrize("environ, chunk", [({}, None), ({"NEMORL_MEGATRON_LOGPROB_CHUNK_SIZE": "2048"}, 2048)])
    def test_megatron_logprob_chunking_defers_fp32_logits(self, environ, chunk):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ=environ)
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config=LORA16,
        )
        assert config_dict["policy"]["logprob_chunk_size"] == chunk
        assert config_dict["policy"]["megatron_cfg"]["defer_fp32_logits"] is (chunk is not None)

    @pytest.mark.parametrize("environ, env_vars", [
        ({}, {}),
        ({"NEMORL_MEGATRON_ALLOC_CONF": "expandable_segments:True"},
         {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    ])
    def test_megatron_alloc_conf_reaches_the_worker_env(self, environ, env_vars):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ=environ)
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config=LORA16,
        )
        assert config_dict["policy"]["megatron_cfg"]["env_vars"] == env_vars

    @pytest.mark.parametrize("overrides", [{"megatron_recompute": True}, {"megatron_logprob_chunk_size": 2048},
                                           {"megatron_alloc_conf": "expandable_segments:True"}])
    def test_megatron_memory_knobs_are_refused_on_dtensor(self, overrides):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env(overrides, environ={})
        with pytest.raises(ValueError, match="need the Megatron worker"):
            NemoRLArgumentBuilder(config=cfg).build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=2)

    def test_dense_model_keeps_the_dense_moe_settings(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"megatron": True}, environ={})
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=2, lora_config=LORA16,
        )
        meg = config_dict["policy"]["megatron_cfg"]
        assert meg["expert_model_parallel_size"] == 1
        assert meg["moe_token_dispatcher_type"] == "allgather"
        assert "moe_backend" not in config_dict["policy"]["generation"]["vllm_kwargs"]

    @pytest.mark.parametrize("overrides, match", [
        ({"megatron": True, "megatron_ep": 2}, "does not divide the model's 0 experts"),
        ({"megatron_ep": 2}, "needs the Megatron worker"),
    ])
    def test_expert_parallel_is_refused_where_it_cannot_apply(self, overrides, match):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env(overrides, environ={})
        with pytest.raises(ValueError, match=match):
            NemoRLArgumentBuilder(config=cfg).build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=2)

    def test_sequence_packing_is_refused(self):
        # the step's per-sample logprobs are not collected from packed microbatches
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        builder = NemoRLArgumentBuilder(overrides={"policy": {"sequence_packing": {"enabled": True}}})
        with pytest.raises(ValueError, match="sequence_packing"):
            builder.build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=2)

    @pytest.mark.parametrize("inference_tp, layout", [(None, (2, 1)), (4, (4, 1))])
    def test_colocated_vllm_tp_follows_the_trainer_unless_set(self, inference_tp, layout):
        # unset, each engine spans one training model-parallel group; set, its own TP
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"inference_tp": inference_tp}, environ={})
        config_dict, _ = NemoRLArgumentBuilder(config=cfg).build_args(
            base_model="meta-llama/Llama-3.1-8B", num_gpus=8, parallelism={"tensor_parallel": 2},
        )
        vllm = config_dict["policy"]["generation"]["vllm_cfg"]
        assert (vllm["tensor_parallel_size"], vllm["pipeline_parallel_size"]) == layout
        assert config_dict["policy"]["dtensor_cfg"]["tensor_parallel_size"] == 2

    def test_colocated_vllm_tp_must_divide_the_gpus(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"inference_tp": 3}, environ={})
        with pytest.raises(ValueError, match="vLLM TP"):
            NemoRLArgumentBuilder(config=cfg).build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=8)

    def test_split_layout_refuses_when_no_training_gpu_remains(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
        cfg = NemoRLConfig.from_env({"colocated": False, "inference_gpus": 2}, environ={})
        with pytest.raises(ValueError, match="leaves 0 training GPU"):
            NemoRLArgumentBuilder(config=cfg).build_args(base_model="meta-llama/Llama-3.1-8B", num_gpus=2)

    def test_lora_config_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
            lora_config={"rank": 32, "alpha": 64, "dropout": 0.1},
        )
        lora_cfg = config_dict["policy"]["dtensor_cfg"]["lora_cfg"]
        assert lora_cfg["enabled"] is True
        assert lora_cfg["dim"] == 32
        assert lora_cfg["alpha"] == 64
        assert lora_cfg["dropout"] == 0.1

    def test_rl_config_mapping(self, builder):
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
            rl_config={"kl_penalty_coef": 0.01, "eps_clip": 0.3},
        )
        assert config_dict["loss_fn"]["reference_policy_kl_penalty"] == 0.01
        assert config_dict["loss_fn"]["ratio_clip_min"] == 0.3

    def test_overrides_applied(self):
        from tinkercloud.training.backends.nemo_rl.builder import NemoRLArgumentBuilder
        builder = NemoRLArgumentBuilder(overrides={
            "policy": {"precision": "float32"},
        })
        config_dict, _ = builder.build_args(
            base_model="meta-llama/Llama-3.1-8B",
            num_gpus=4,
        )
        assert config_dict["policy"]["precision"] == "float32"


# Keys of the installed NeMo RL's SFT exemplar the builder leaves out, each read optionally by
# NeMo RL 0.7 (so absence is a real state there); any other missing key is drift to fix.
_OPTIONAL_UPSTREAM_KEYS = {
    "tokenizer.chat_template",          # presence-checked (algorithms/utils.py:373); the client tokenizes
    "tokenizer.chat_template_kwargs",   # presence-checked (algorithms/utils.py:392)
    "dtensor_cfg.env_vars",             # .get (models/policy/lm_policy.py:165)
    "dtensor_cfg.dp_replicate_size",    # .get (models/automodel/setup.py:464)
    "dtensor_cfg.automodel_kwargs",     # .get (models/automodel/setup.py:678)
    "dtensor_cfg.lora_cfg.match_all_linear",  # PeftConfig.from_dict default; target_modules are explicit
}


def _missing_keys(reference: dict, ours: dict, prefix: str = "") -> list:
    """Keys of reference absent from ours, descending into sections ours enables."""
    missing = []
    for key, value in reference.items():
        if key not in ours:
            missing.append(prefix + key)
        elif isinstance(value, dict) and isinstance(ours[key], dict) and ours[key].get("enabled", True) is not False:
            missing += _missing_keys(value, ours[key], f"{prefix}{key}.")
    return missing


@pytest.mark.parametrize("megatron", [False, True])
def test_policy_config_carries_every_key_the_installed_nemo_rl_reads(megatron, monkeypatch):
    # NeMo RL reads many policy keys unconditionally; one missing key is a worker crash at
    # create_model. Its own SFT exemplar lists them, so diff against the installed copy.
    nemo_rl = pytest.importorskip("nemo_rl")
    import os

    import yaml
    exemplar = os.path.join(os.path.dirname(os.path.dirname(nemo_rl.__file__)), "examples", "configs", "sft.yaml")
    if not os.path.isfile(exemplar):
        pytest.skip(f"no NeMo RL exemplar at {exemplar}")
    with open(exemplar) as f:
        reference = yaml.safe_load(f)["policy"]
    from tinkercloud.training.backends.nemo_rl import builder as mod
    from tinkercloud.training.backends.nemo_rl.config import NemoRLConfig
    monkeypatch.setattr(mod, "read_raw_hf_config", lambda base_model: {
        "architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768,
    })
    cfg = NemoRLConfig.from_env({"megatron": megatron}, environ={})
    config_dict, _ = mod.NemoRLArgumentBuilder(config=cfg).build_args(
        base_model="Qwen/Qwen2.5-0.5B-Instruct", num_gpus=2, lora_config=LORA16,
    )
    drift = sorted(set(_missing_keys(reference, config_dict["policy"])) - _OPTIONAL_UPSTREAM_KEYS)
    assert not drift, f"policy keys {exemplar} has and the builder does not write: {drift}"
