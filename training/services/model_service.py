"""
Model Service - Business Logic for Model Management

Handles:
- Model creation (Ray actors, placement groups, rollout managers)
- Model deletion (cleanup GPU resources)
- Model metadata retrieval
"""
import asyncio
import logging
import ray
from datetime import datetime
from typing import Dict, Any, Optional

from ..core import SlimeArgumentBuilder
from ..storage import MetadataStorage
from ..utils.model_config import extract_model_name, detect_architecture

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model lifecycle and resources."""

    def __init__(self):
        """Initialize ModelService."""
        pass

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        lora_config: Optional[Dict[str, Any]],
        debug_train_only: bool,
        checkpoint_path: Optional[str],
        parallelism_config: Optional[Dict[str, Any]],
        slime_builder: SlimeArgumentBuilder,
        metadata_storage: MetadataStorage,
        training_clients: Dict[str, Dict[str, Any]],
        training_runs_metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new training model with Ray actors and GPU resources.

        Args:
            model_id: Unique identifier for the model
            request_id: Unique identifier for the request
            base_model: Base model path (HuggingFace or local)
            lora_config: Optional LoRA configuration dict
            debug_train_only: Whether to run in SFT mode (no rollout manager)
            checkpoint_path: Optional checkpoint path to resume from
            parallelism_config: Optional parallelism configuration
            slime_builder: Slime argument builder instance
            metadata_storage: Metadata storage instance
            training_clients: Global training clients dict
            training_runs_metadata: Global training runs metadata dict

        Returns:
            Dict with model_id, base_model, lora_config, status

        Raises:
            RuntimeError: If actor initialization fails
        """
        logger.info(f"[{request_id}] Creating model {model_id}")
        print(f"[DEBUG model_service] Creating model {model_id}", flush=True)

        # Build Slime arguments using builder
        # CRITICAL: build_args() contains blocking operations (HF AutoConfig, Megatron arg parsing)
        # Must run in thread pool to avoid blocking the async event loop
        logger.debug(f"[{request_id}] About to call build_args() in thread pool")
        print(f"[DEBUG model_service] About to call build_args()", flush=True)
        args, hf_path = await asyncio.to_thread(
            slime_builder.build_args,
            base_model=base_model,
            lora_config=lora_config,
            debug_train_only=debug_train_only,
            checkpoint_path=checkpoint_path,
            parallelism_config=parallelism_config
        )
        print(f"[DEBUG model_service] build_args() returned, hf_path={hf_path}", flush=True)
        logger.debug(f"[{request_id}] build_args() completed")

        # Log configuration
        print(f"[DEBUG model_service] About to log Model config", flush=True)
        logger.info(f"[{request_id}] Model config: base={base_model}, hf_path={hf_path}")
        if lora_config:
            logger.info(f"[{request_id}] LoRA config: {lora_config}")

        # Create training run metadata (use model_id as training_run_id for compatibility)
        training_run_id = model_id
        metadata = {
            "training_run_id": model_id,
            "model_id": model_id,
            "base_model": base_model,
            "hf_path": hf_path,
            "lora_config": lora_config,
            "created_at": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "model_owner": "kgateway-user",
            "is_lora": lora_config is not None,
            "lora_rank": lora_config.get("rank", 0) if lora_config else 0,
            "corrupted": False,
            "last_request_time": datetime.now().isoformat(),
            "last_checkpoint": None,
            "last_sampler_checkpoint": None
        }

        # Save to storage
        print(f"[DEBUG model_service] Saving metadata", flush=True)
        metadata_storage.save_training_run(model_id, metadata)
        training_runs_metadata[model_id] = metadata
        logger.info(f"[{request_id}] Metadata saved successfully")
        print(f"[DEBUG model_service] Metadata saved", flush=True)

        # Get or create Ray actor
        print(f"[DEBUG model_service] About to get or create Ray actor", flush=True)
        try:
            # Try to get existing actor first (with timeout to avoid blocking)
            try:
                train_group = ray.get_actor(f"RayTrainGroup-{model_id}", namespace="training")
                logger.info(f"[{request_id}] Found existing Ray actor for {model_id}")
                print(f"[DEBUG model_service] Found existing actor", flush=True)
            except:
                # Actor doesn't exist, create new one
                raise
        except:
            # Create new actor
            print(f"[DEBUG model_service] Creating new Ray actor", flush=True)
            logger.info(f"[{request_id}] Creating new Ray actor for {model_id}")
            print(f"[DEBUG model_service] About to import RayTrainGroup", flush=True)

            # Import and create Miles actors
            from miles.ray.actor_group import RayTrainGroup

            # Create placement group for GPU allocation
            # Matching working Qwen2.5-0.5B config: 4 GPUs with TP=2
            num_nodes = 1
            num_gpus_per_node = 4

            bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_nodes * num_gpus_per_node)]
            logger.info(f"[{request_id}] Creating placement group with bundles={bundles}, strategy=PACK")
            pg = ray.util.placement_group(bundles, strategy="PACK")

            # Use async API with timeout protection
            logger.info(f"[{request_id}] Waiting for placement group to be ready (timeout=120s)")
            await asyncio.wait_for(
                asyncio.wrap_future(pg.ready().future()),
                timeout=120.0  # 2 minutes max wait for GPU resource allocation
            )
            logger.info(f"[{request_id}] Placement group ready!")

            # Create reordered bundle indices (identity mapping for simple case)
            reordered_indices = list(range(len(bundles)))

            # Create RayTrainGroup
            # CRITICAL: Use fractional GPU (0.8) to leave room for SGLangEngines (0.2 each)
            # Total per bundle: 0.8 (training) + 0.2 (SGLang) = 1.0 GPU
            logger.info(f"[{request_id}] Creating RayTrainGroup with {num_gpus_per_node} GPUs")
            train_group = RayTrainGroup(
                args=args,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                pg=(pg, reordered_indices),
                num_gpus_per_actor=0.8,  # Leave 0.2 for SGLangEngine colocated on same bundle
                role="actor"
            )
            logger.info(f"[{request_id}] RayTrainGroup object created, initializing actors...")

            # Initialize actors - CRITICAL: missing in original refactored version
            init_refs = train_group.async_init(args, role="actor", with_ref=False)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in init_refs]),
                    timeout=180.0  # 3 minutes max for actor initialization (model loading)
                )
                logger.info(f"[{request_id}] Actor initialization completed")
            except asyncio.TimeoutError:
                logger.error(f"[{request_id}] Actor initialization timeout")
                # Clean up placement group
                ray.util.remove_placement_group(pg)
                raise RuntimeError(
                    "Actor initialization timeout after 180s. "
                    "Possible causes: model loading failure, network issues, or insufficient memory. "
                    "Check Ray logs for details."
                )

            # Create RolloutManager for SGLang sampling (only for RL, not for SFT)
            rollout_manager = None
            router_ip = None
            router_port = None

            print(f"[DEBUG model_service] About to check debug_train_only={debug_train_only}", flush=True)
            if not debug_train_only:
                # RL mode: Create RolloutManager and SGLang engines
                print(f"[DEBUG model_service] RL mode - creating RolloutManager", flush=True)
                logger.info(f"[{request_id}] Creating RolloutManager with SGLang (RL mode)")

                from miles.ray.rollout import RolloutManager

                # Share the same placement group with training (colocated mode)
                print(f"[DEBUG model_service] About to create RolloutManager.remote()", flush=True)
                rollout_manager = RolloutManager.options(
                    num_cpus=1,
                    num_gpus=0,
                ).remote(args, (pg, reordered_indices))
                print(f"[DEBUG model_service] RolloutManager.remote() created", flush=True)

                # Connect training actors to rollout manager
                print(f"[DEBUG model_service] Setting rollout_manager on train_group", flush=True)
                train_group.set_rollout_manager(rollout_manager)
                print(f"[DEBUG model_service] rollout_manager set on train_group", flush=True)

                # Initialize SGLang memory state (match Miles create_rollout_manager + train.py flow)
                # Step 1: Offload immediately after creation to initialize memory_saver state
                print(f"[DEBUG model_service] Starting SGLang init, offload_rollout={args.offload_rollout}", flush=True)
                if args.offload_rollout:
                    print(f"[DEBUG model_service] Step 1: Initial offload", flush=True)
                    logger.info(f"[{request_id}] Initial offload to initialize memory_saver state")
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.offload.remote()))
                    print(f"[DEBUG model_service] Step 1: Offload complete", flush=True)

                # Step 2: Onload weights before update_weights
                from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE
                try:
                    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
                except ImportError:
                    GPU_MEMORY_TYPE_CUDA_GRAPH = None

                if args.offload_rollout:
                    print(f"[DEBUG model_service] Step 2: Onload weights", flush=True)
                    logger.info(f"[{request_id}] Onloading weights for update_weights")
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])))
                    print(f"[DEBUG model_service] Step 2: Onload weights complete", flush=True)

                # Step 3: Push initial weights to SGLang
                print(f"[DEBUG model_service] Step 3: Push weights", flush=True)
                logger.info(f"[{request_id}] Pushing initial weights to SGLang")
                await asyncio.to_thread(train_group.update_weights)
                print(f"[DEBUG model_service] Step 3: Push weights complete", flush=True)

                # Step 4: Onload CUDA graphs and KV cache
                if args.offload_rollout:
                    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                        print(f"[DEBUG model_service] Step 4a: Onload CUDA graphs", flush=True)
                        logger.info(f"[{request_id}] Onloading CUDA graphs")
                        await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])))
                    print(f"[DEBUG model_service] Step 4b: Onload KV cache", flush=True)
                    logger.info(f"[{request_id}] Onloading KV cache")
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])))
                    logger.info(f"[{request_id}] SGLang memory state initialized")
                    print(f"[DEBUG model_service] SGLang init complete!", flush=True)

                # Get router address from RolloutManager
                try:
                    logger.info(f"[{request_id}] Getting router address from RolloutManager")
                    router_address_ref = rollout_manager.get_router_address.remote()
                    router_ip, router_port = await asyncio.wrap_future(router_address_ref.future())

                    if router_ip and router_port:
                        logger.info(f"[{request_id}] Router address: {router_ip}:{router_port}")
                    else:
                        logger.warning(f"[{request_id}] RolloutManager returned None for router address")
                except Exception as e:
                    logger.error(f"[{request_id}] Failed to get router address: {e}")
                    router_ip = None
                    router_port = None
            else:
                # SFT mode: No RolloutManager needed
                logger.info(f"[{request_id}] Skipping RolloutManager (SFT mode, debug_train_only=True)")

            # Store client info
            training_clients[model_id] = {
                "train_group": train_group,
                "rollout_manager": rollout_manager,
                "placement_group": pg,
                "training_run_id": training_run_id,
                "args": args,
                "hf_path": hf_path,
                "router_ip": router_ip,
                "router_port": router_port,
                "created_at": datetime.now().isoformat()
            }

        # Return result
        result = {
            "model_id": model_id,
            "base_model": base_model,
            "lora_config": lora_config,
            "status": "ready"
        }

        logger.info(f"[{request_id}] Model {model_id} created successfully")
        return result

    async def delete_model(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]],
        metadata_storage: MetadataStorage
    ) -> Dict[str, Any]:
        """
        Delete training client and release GPU resources.

        Args:
            model_id: Model identifier to delete
            training_clients: Global training clients dict
            metadata_storage: Metadata storage instance

        Returns:
            Dict with model_id, message, resources_freed list

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        train_group = client_info["train_group"]

        # Kill Ray actors
        resources_freed = []
        for actor in train_group._actor_handlers:
            ray.kill(actor, no_restart=True)
            resources_freed.append(f"actor_{actor}")

        # Kill rollout manager if exists
        if "rollout_manager" in client_info and client_info["rollout_manager"] is not None:
            ray.kill(client_info["rollout_manager"], no_restart=True)
            resources_freed.append("rollout_manager")

        # Remove placement group
        if "placement_group" in client_info:
            ray.util.remove_placement_group(client_info["placement_group"])
            resources_freed.append("placement_group")

        # Remove from training_clients
        del training_clients[model_id]

        # Update metadata
        if "training_run_id" in client_info:
            metadata_storage.update_training_run(
                client_info["training_run_id"],
                {"last_request_time": datetime.now().isoformat()}
            )

        logger.info(f"Deleted model {model_id}, freed {len(resources_freed)} resources")

        return {
            "model_id": model_id,
            "message": "Training client resources freed, metadata preserved for resume",
            "resources_freed": resources_freed
        }

    def get_model_info(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get model info for tokenizer initialization.

        Args:
            model_id: Model identifier
            training_clients: Global training clients dict

        Returns:
            Dict with model_id, model_data, is_lora, lora_rank, model_name

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        args = client_info["args"]

        # Extract model name and architecture
        model_name = extract_model_name(args)
        arch = detect_architecture(model_name)

        return {
            "model_id": model_id,
            "model_data": {
                "arch": arch,
                "model_name": model_name
            },
            "is_lora": args.lora_rank > 0,
            "lora_rank": args.lora_rank if args.lora_rank > 0 else None,
            "model_name": model_name
        }

    def get_tokenizer_info(
        self,
        model_id: str,
        training_clients: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get tokenizer information from HuggingFace model.

        Args:
            model_id: Model identifier
            training_clients: Global training clients dict

        Returns:
            Dict with tokenizer information

        Raises:
            KeyError: If model_id not found
            ValueError: If HuggingFace path not available
            Exception: If tokenizer loading fails
        """
        if model_id not in training_clients:
            raise KeyError(f"Model {model_id} not found")

        client_info = training_clients[model_id]
        hf_path = client_info.get("hf_path")

        if not hf_path:
            raise ValueError("HuggingFace path not available")

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

            # Build special tokens
            special_tokens = {
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "bos_token": tokenizer.bos_token,
                "unk_token": tokenizer.unk_token
            }

            return {
                "vocab_size": len(tokenizer),
                "model_max_length": tokenizer.model_max_length,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "special_tokens": special_tokens,
                "hf_checkpoint": hf_path
            }
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def get_training_run_metadata(
        self,
        model_id: str,
        metadata_storage: MetadataStorage
    ) -> Dict[str, Any]:
        """
        Load persistent training run metadata.

        Args:
            model_id: Model identifier
            metadata_storage: Metadata storage instance

        Returns:
            Dict with training run metadata

        Raises:
            KeyError: If training run not found
        """
        metadata = metadata_storage.load_training_run(model_id)

        if not metadata:
            raise KeyError(f"Training run {model_id} not found")

        # Update last access time
        metadata_storage.update_training_run(
            model_id,
            {"last_request_time": datetime.now().isoformat()}
        )

        return metadata
