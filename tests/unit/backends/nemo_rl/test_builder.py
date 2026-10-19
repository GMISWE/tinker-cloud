"""NemoRLArgumentBuilder: the config dict handed to NeMo RL (parallelism, cluster layout, LoRA, RL knobs, overrides)."""
import pytest


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
