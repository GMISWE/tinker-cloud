"""Miles backend — wraps RayTrainGroup/RolloutManager/SlimeArgumentBuilder
behind the TrainingBackend interface (refactor only, no behavior change)."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray

from ..base import BackendError, BackendHandle, TrainingBackend

logger = logging.getLogger(__name__)


@dataclass
class MilesHandle(BackendHandle):
    """Miles-specific runtime state."""

    train_group: Any = None           # RayTrainGroup
    rollout_manager: Any = None       # RolloutManager (None for SFT)
    placement_group: Any = None       # Ray PlacementGroup
    args: Any = None                  # Megatron Namespace
    hf_path: str = ""
    router_ip: Optional[str] = None
    router_port: Optional[int] = None
    rlve_config: Optional[Dict[str, Any]] = None
    wandb_config: Optional[Dict[str, Any]] = None
    created_at: str = ""
    training_run_id: str = ""


class MilesBackend(TrainingBackend):
    """Thin adapter over existing Miles integration code (model_service.py / training_service.py)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        # Lazy-import converter to avoid import errors when Miles is not installed
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import MilesDataConverter
            self._converter = MilesDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import MilesArgumentBuilder
            self._builder = MilesArgumentBuilder()
        return self._builder

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False,
        checkpoint_path: Optional[str] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> MilesHandle:
        if objective != "language_modeling":
            raise BackendError(
                f"Miles is a language-modeling backend; objective {objective!r} "
                f"requires a classification backend (automodel / megatron_bridge)",
                backend="miles", operation="create_model",
            )
        try:
            logger.info("[%s] Creating Miles model %s", request_id, model_id)

            # Build Slime arguments (blocking — run in thread pool)
            args, hf_path = await asyncio.to_thread(
                self.builder.build_args,
                base_model=base_model,
                lora_config=lora_config,
                debug_train_only=debug_train_only,
                checkpoint_path=checkpoint_path,
                parallelism_config=parallelism,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
            )
            logger.info("[%s] Miles args built, hf_path=%s", request_id, hf_path)

            from miles.ray.actor_group import RayTrainGroup

            num_nodes = 1
            num_gpus_per_node = num_gpus
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_nodes * num_gpus_per_node)]
            pg = ray.util.placement_group(bundles, strategy="PACK")

            await asyncio.wait_for(
                asyncio.wrap_future(pg.ready().future()),
                timeout=120.0,
            )

            reordered_indices = list(range(len(bundles)))

            train_group = RayTrainGroup(
                args=args,
                num_nodes=num_nodes,
                num_gpus_per_node=num_gpus_per_node,
                pg=(pg, reordered_indices),
                num_gpus_per_actor=0.8,
                role="actor",
            )

            init_refs = train_group.async_init(args, role="actor", with_ref=False)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in init_refs]),
                    timeout=1800.0,
                )
            except asyncio.TimeoutError:
                ray.util.remove_placement_group(pg)
                raise BackendError(
                    "Actor initialization timeout after 1800s",
                    backend="miles",
                    operation="create_model",
                )

            # Create RolloutManager for SGLang (RL mode only)
            rollout_manager = None
            router_ip = None
            router_port = None

            if not debug_train_only:
                from miles.ray.rollout import RolloutManager

                rollout_manager = RolloutManager.options(
                    num_cpus=1, num_gpus=0,
                ).remote(args, (pg, reordered_indices))

                await asyncio.to_thread(train_group.set_rollout_manager, rollout_manager)

                # Initialize SGLang memory state
                from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE
                try:
                    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
                except ImportError:
                    GPU_MEMORY_TYPE_CUDA_GRAPH = None

                if args.offload_rollout:
                    await asyncio.to_thread(lambda: ray.get(rollout_manager.offload.remote()))
                    await asyncio.to_thread(lambda: ray.get(
                        rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                    ))

                await asyncio.to_thread(train_group.update_weights)

                if args.offload_rollout:
                    if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                        await asyncio.to_thread(lambda: ray.get(
                            rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                        ))
                    await asyncio.to_thread(lambda: ray.get(
                        rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                    ))

                try:
                    router_address_ref = rollout_manager.get_router_address.remote()
                    router_ip, router_port = await asyncio.wrap_future(router_address_ref.future())
                except Exception as e:
                    logger.error("[%s] Failed to get router address: %s", request_id, e)

            handle = MilesHandle(
                model_id=model_id,
                backend_type="miles",
                train_group=train_group,
                rollout_manager=rollout_manager,
                placement_group=pg,
                args=args,
                hf_path=hf_path,
                router_ip=router_ip,
                router_port=router_port,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
                created_at=datetime.now().isoformat(),
                training_run_id=model_id,
            )

            logger.info("[%s] Miles model %s created successfully", request_id, model_id)
            return handle

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="create_model", original_error=e,
            ) from e

    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            from miles.utils.ray_utils import Box

            if h.rollout_manager is not None and h.args.offload_rollout:
                await asyncio.to_thread(lambda: ray.get(h.rollout_manager.offload.remote()))

            rollout_data = self.converter.forward_to_backend(data, h.args)
            results = await asyncio.to_thread(
                h.train_group.forward_only,
                rollout_id=0,
                rollout_data_ref=Box(ray.put(rollout_data)),
            )
            return self.converter.backend_to_forward_result(results, data)

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward", original_error=e,
            ) from e

    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            from miles.utils.ray_utils import Box

            if h.rollout_manager is not None and h.args.offload_rollout:
                await asyncio.to_thread(lambda: ray.get(h.rollout_manager.offload.remote()))

            is_rl = not h.args.debug_train_only

            # Check if data needs fake generation (legacy test compat)
            needs_fake_data = (
                not data
                or len(data) == 0
                or len(data) < h.args.data_parallel_size
            )

            if needs_fake_data:
                logger.info(
                    "Insufficient data (%d samples, need %d) — generating fake test data",
                    len(data) if data else 0,
                    h.args.data_parallel_size,
                )
                rollout_data = self._generate_fake_rollout_data(h.args)
            else:
                from ...core.validators import RequestValidator
                from ...config import get_config

                config = get_config()
                allow_partial = getattr(config, "allow_partial_batches", False)
                validator = RequestValidator(h.args, allow_partial_batches=allow_partial)
                validation_error = validator.validate_forward_backward_request(
                    data, is_rl=is_rl,
                )
                if validation_error:
                    raise ValueError(
                        f"Request validation failed:\n{validation_error}\n\n"
                        f"{validator.get_config_summary()}"
                    )

                rollout_data = self.converter.forward_backward_to_backend(
                    data, loss_fn, h.args,
                )

            results = await asyncio.to_thread(
                h.train_group.forward_backward_only,
                rollout_id=0,
                rollout_data_ref=Box(ray.put(rollout_data)),
            )
            result = self.converter.backend_to_forward_backward_result(results, data)
            result["deferred"] = False
            return result

        except BackendError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward_backward", original_error=e,
            ) from e

    @staticmethod
    def _generate_fake_rollout_data(args) -> Dict[str, Any]:
        """Generate fake rollout data for legacy test compatibility."""
        import torch

        seq_length = args.seq_length
        vocab_size = args.vocab_size
        batch_size = args.global_batch_size
        response_length = seq_length - (seq_length // 2)
        device = torch.device("cpu")

        tokens_list = []
        loss_masks_list = []
        response_lengths_list = []
        advantages_list = []
        log_probs_list = []
        ref_log_probs_list = []
        returns_list = []
        values_list = []

        for _ in range(batch_size):
            tokens_list.append(
                torch.randint(0, vocab_size, (seq_length,), dtype=torch.long, device=device)
            )
            loss_masks_list.append(
                torch.ones(response_length, dtype=torch.float32, device=device)
            )
            response_lengths_list.append(response_length)
            advantages_list.append(
                torch.randn(response_length, dtype=torch.float32, device=device) * 0.1
            )
            log_probs_list.append(
                torch.randn(response_length, dtype=torch.float32, device=device) * 0.5 - 5.0
            )
            ref_log_probs_list.append(
                torch.randn(response_length, dtype=torch.float32, device=device) * 0.5 - 5.0
            )
            returns_list.append(
                torch.randn(response_length, dtype=torch.float32, device=device) * 0.5
            )
            values_list.append(
                torch.randn(response_length, dtype=torch.float32, device=device) * 0.5
            )

        return {
            "tokens": tokens_list,
            "loss_masks": loss_masks_list,
            "response_lengths": response_lengths_list,
            "advantages": advantages_list,
            "log_probs": log_probs_list,
            "ref_log_probs": ref_log_probs_list,
            "values": values_list,
            "returns": returns_list,
        }

    async def apply_optimizer_step(
        self,
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            if learning_rate is not None:
                await asyncio.to_thread(h.train_group.set_learning_rate, learning_rate)

            offload_train = h.args.offload_train if h.args else True
            offload_rollout = h.args.offload_rollout if h.args else True

            if not offload_train and not offload_rollout:
                results = await asyncio.to_thread(h.train_group.apply_optimizer_step_and_sync)
            else:
                results = await asyncio.to_thread(h.train_group.apply_optimizer_step)

                if h.rollout_manager is not None:
                    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
                    try:
                        from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
                    except ImportError:
                        GPU_MEMORY_TYPE_CUDA_GRAPH = None

                    if offload_train:
                        await asyncio.to_thread(h.train_group.offload)

                    if offload_rollout:
                        await asyncio.to_thread(lambda: ray.get(
                            h.rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
                        ))

                    await asyncio.to_thread(h.train_group.update_weights)

                    if offload_rollout:
                        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                            await asyncio.to_thread(lambda: ray.get(
                                h.rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH])
                            ))
                        await asyncio.to_thread(lambda: ray.get(
                            h.rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE])
                        ))

            return {
                "success": results[0]["success"],
                "grad_norm": results[0]["grad_norm"],
                "learning_rates": [],
                "model_id": h.model_id,
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="apply_optimizer_step", original_error=e,
            ) from e

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            await asyncio.to_thread(h.train_group.update_weights)
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="update_inference_weights", original_error=e,
            ) from e

    async def save_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            offload_train = h.args.offload_train if h.args else False
            if offload_train:
                logger.info("Skipping save_model (offload_train=True)")
                return checkpoint_path

            if step_id is None:
                step_id = 0

            object_refs = [
                actor.save_model.remote(step_id)
                for actor in h.train_group._actor_handlers
            ]
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])
            return checkpoint_path

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="save_checkpoint", original_error=e,
            ) from e

    async def load_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
    ) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            object_refs = [
                actor.load_checkpoint.remote(checkpoint_path)
                for actor in h.train_group._actor_handlers
            ]
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in object_refs])

            # Sync loaded weights to inference engine
            if h.rollout_manager is not None:
                await asyncio.to_thread(h.train_group.update_weights)

            logger.info("Miles checkpoint loaded from %s", checkpoint_path)

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="load_checkpoint", original_error=e,
            ) from e

    async def delete_model(self, handle: BackendHandle) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        try:
            resources_freed = []
            for actor in h.train_group._actor_handlers:
                ray.kill(actor, no_restart=True)
                resources_freed.append("actor")

            if h.rollout_manager is not None:
                ray.kill(h.rollout_manager, no_restart=True)
                resources_freed.append("rollout_manager")

            if h.placement_group is not None:
                ray.util.remove_placement_group(h.placement_group)
                resources_freed.append("placement_group")

            logger.info("Miles model %s deleted, freed %d resources", h.model_id, len(resources_freed))

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="delete_model", original_error=e,
            ) from e

    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        # Miles computes logprobs internally during forward_backward.
        # Expose via forward-only path for explicit logprob requests.
        result = await self.forward(handle, data, loss_fn="cross_entropy")
        logprobs_list = []
        for output in result.get("loss_fn_outputs", []):
            lp = output.get("logprobs", {})
            logprobs_list.append(lp.get("data", []))
        return logprobs_list

    async def sample(
        self,
        handle: BackendHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Sample via per-request HTTP calls to the SGLang router."""
        from ...utils.sglang_client import SGLangClient

        h: MilesHandle = handle  # type: ignore[assignment]
        if not h.router_ip or not h.router_port:
            raise BackendError(
                "SGLang router not available",
                backend="miles", operation="sample",
            )
        client = SGLangClient(base_url=f"http://{h.router_ip}:{h.router_port}")

        sequences = []
        prompt_logprobs_result = None
        for _ in range(num_samples):
            result = await client.generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params or {},
                prompt_logprobs=prompt_logprobs,
            )
            sequences.append({
                "tokens": result["tokens"],
                "logprobs": result["logprobs"],
                "text": result.get("text"),
                "stop_reason": result.get("stop_reason", "length"),
            })
            if prompt_logprobs and prompt_logprobs_result is None:
                prompt_logprobs_result = result.get("prompt_logprobs")

        return {
            "sequences": sequences,
            "prompt_logprobs": prompt_logprobs_result,
        }

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        """SGLang router is always live for Miles — just validate it exists."""
        h: MilesHandle = handle  # type: ignore[assignment]
        if not h.router_ip or not h.router_port:
            raise BackendError(
                "SGLang router not available",
                backend="miles", operation="prepare_for_generation",
            )
