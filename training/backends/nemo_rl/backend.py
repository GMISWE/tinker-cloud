"""
NeMo RL backend implementation.

Implements TrainingBackend using NeMo RL's Policy push-mode API.

Key design: R9 Buffering Strategy
  - forward_backward() BUFFERS data (no GPU work yet)
  - apply_optimizer_step() calls policy.train() ONCE with all buffered data,
    then syncs weights to inference engine and clears the buffer.

This matches NeMo RL's training model where policy.train() always couples
forward + backward + optimizer.step() in a single call.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend

logger = logging.getLogger(__name__)


# Default maximum number of forward_backward() calls that can be buffered
# before apply_optimizer_step() must be called. Prevents unbounded memory growth.
DEFAULT_MAX_BUFFER_SIZE = 64


@dataclass
class NemoRLHandle(BackendHandle):
    """NeMo RL-specific runtime state."""

    policy: Any = None               # nemo_rl.models.policy.lm_policy.Policy
    policy_generation: Any = None    # SGLangGeneration or Policy (if colocated)
    cluster: Any = None              # RayVirtualCluster
    config: Dict = field(default_factory=dict)   # Full config dict
    tokenizer: Any = None            # HuggingFace tokenizer
    loss_fn: Any = None              # ClippedPGLossFn instance
    data_buffer: List = field(default_factory=list)  # R9 buffering
    max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE  # CHK006: bound buffer growth
    _buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # CHK018: thread safety
    hf_path: str = ""
    colocated_inference: bool = True
    rlve_config: Optional[Dict[str, Any]] = None
    wandb_config: Optional[Dict[str, Any]] = None
    created_at: str = ""
    training_run_id: str = ""
    debug_train_only: bool = False


class NemoRLBackend(TrainingBackend):
    """
    NeMo RL backend — uses Policy.train() push-mode API.

    Implements the R9 buffering strategy: forward_backward() buffers data,
    apply_optimizer_step() calls policy.train() once with all buffered data.
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import NemoRLDataConverter
            self._converter = NemoRLDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import NemoRLArgumentBuilder
            self._builder = NemoRLArgumentBuilder(overrides=self.overrides)
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
    ) -> NemoRLHandle:
        try:
            logger.info("[%s] Creating NeMo RL model %s", request_id, model_id)

            # Build NeMo RL config
            config_dict, hf_path = await asyncio.to_thread(
                self.builder.build_args,
                base_model=base_model,
                num_gpus=num_gpus,
                lora_config=lora_config,
                parallelism=parallelism,
                rl_config=rl_config,
                rollout_config=rollout_config,
                debug_train_only=debug_train_only,
                checkpoint_path=checkpoint_path,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
            )
            logger.info("[%s] NeMo RL config built, hf_path=%s", request_id, hf_path)

            # Initialize NeMo RL components (blocking — run in thread pool)
            policy, policy_generation, cluster, tokenizer, loss_fn = await asyncio.to_thread(
                _init_nemo_rl_components,
                config_dict=config_dict,
                checkpoint_path=checkpoint_path,
                debug_train_only=debug_train_only,
            )

            handle = NemoRLHandle(
                model_id=model_id,
                backend_type="nemo_rl",
                policy=policy,
                policy_generation=policy_generation,
                cluster=cluster,
                config=config_dict,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                hf_path=hf_path,
                colocated_inference=not debug_train_only,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
                created_at=datetime.now().isoformat(),
                training_run_id=model_id,
                debug_train_only=debug_train_only,
            )

            logger.info("[%s] NeMo RL model %s created successfully", request_id, model_id)
            return handle

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="create_model", original_error=e,
            ) from e

    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        """Forward-only pass — compute logprobs without gradients."""
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            # Convert data to NeMo RL format
            batched_data = self.converter.forward_to_backend(data, h.config)

            # Compute logprobs via policy.get_logprobs()
            result = await asyncio.to_thread(h.policy.get_logprobs, batched_data)

            return self.converter.backend_to_forward_result(result, data)

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="forward", original_error=e,
            ) from e

    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        """
        Buffer incoming data per R9 strategy. NO GPU training work yet.

        Data is converted and appended to handle.data_buffer.
        Actual training happens in apply_optimizer_step().

        Deferred contract: Returns {"metrics": {}, "deferred": True,
        "loss_fn_outputs": []}. Real metrics at optim_step time.

        Thread safety: Buffer access is protected by asyncio.Lock.
        Concurrent forward_backward calls are serialized at the buffer.

        Constraints:
        - Buffer bounded by max_buffer_size (default 64, raises on overflow)
        - All microbatches must have identical field keys
        - Microbatch ordering is preserved (FIFO)
        """
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            # Basic validation before buffering
            if not data:
                raise BackendError(
                    "Empty data list", backend="nemo_rl", operation="forward_backward",
                )

            # Convert data to NeMo RL BatchedDataDict (outside lock — CPU-bound)
            batched_data = self.converter.forward_backward_to_backend(
                data, loss_fn, h.config,
            )

            # CHK018: Acquire lock for buffer access (concurrent requests)
            async with h._buffer_lock:
                # CHK006: Enforce maximum buffer size to prevent unbounded memory growth
                if len(h.data_buffer) >= h.max_buffer_size:
                    raise BackendError(
                        f"Buffer full: {len(h.data_buffer)} microbatches buffered "
                        f"(max_buffer_size={h.max_buffer_size}). "
                        f"Call apply_optimizer_step() to flush the buffer.",
                        backend="nemo_rl",
                        operation="forward_backward",
                    )

                # CHK016: Validate field compatibility with existing buffer
                if h.data_buffer:
                    existing_keys = set(h.data_buffer[0].keys())
                    new_keys = set(batched_data.keys())
                    if existing_keys != new_keys:
                        raise BackendError(
                            f"Microbatch field mismatch: buffered has {sorted(existing_keys)}, "
                            f"new batch has {sorted(new_keys)}",
                            backend="nemo_rl",
                            operation="forward_backward",
                        )

                # Buffer the data — training deferred to apply_optimizer_step()
                h.data_buffer.append(batched_data)
                buffer_len = len(h.data_buffer)

            logger.info(
                "Buffered microbatch %d for model %s (%d samples)",
                buffer_len, h.model_id, len(data),
            )

            # Return deferred result — no training metrics available yet.
            # CHK010: Deferred contract: empty metrics, deferred=True,
            # empty loss_fn_outputs. Real metrics available at optim_step.
            return {
                "metrics": {},
                "deferred": True,
                "loss_fn_outputs": [],
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="forward_backward", original_error=e,
            ) from e

    async def apply_optimizer_step(
        self,
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute single policy.train() call with all buffered data.

        1. Acquire buffer lock and drain buffer (concatenate + clear)
        2. Pad partial batch if needed (dp_size alignment)
        3. Optionally set learning rate
        4. Call policy.train(all_data, loss_fn)
        5. Sync weights to inference engine (refit)
        6. Return real training metrics

        Buffer-batch relationship: NeMo RL processes ALL buffered data
        regardless of train_global_batch_size. Micro-batching is controlled
        by train_micro_batch_size. Ensure config matches expected total.

        Failure recovery: Buffer is cleared BEFORE policy.train() to
        prevent stale data re-processing. If policy.train() fails (OOM),
        the client must re-send all forward_backward() data.
        """
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            # CHK018: Acquire lock to drain buffer atomically.
            # Snapshot and clear under the lock; concatenate outside it.
            async with h._buffer_lock:
                if not h.data_buffer:
                    raise BackendError(
                        "No buffered data to optimize — call forward_backward() first",
                        backend="nemo_rl",
                        operation="apply_optimizer_step",
                    )

                # Atomic swap: take ownership of buffer, give handle a fresh one.
                # This ensures the buffer is cleared even if concatenation fails (CHK024).
                buffered_batches = h.data_buffer
                h.data_buffer = []
                num_buffered = len(buffered_batches)

            # Concatenate outside the lock (CPU-bound, no contention needed)
            all_data = await asyncio.to_thread(_concatenate_batches, buffered_batches)

            # CHK027: Warn if buffered sample count doesn't match train_global_batch_size
            # (check BEFORE padding so the warning reflects actual data volume)
            original_size = all_data.size
            gbs = h.config.get("policy", {}).get("train_global_batch_size", 0)
            if gbs > 0 and original_size != gbs:
                logger.warning(
                    "Buffered %d samples but train_global_batch_size=%d. "
                    "NeMo RL will process all %d samples. Verify this is intended.",
                    original_size, gbs, original_size,
                )

            # Pad partial batch if needed — policy.train() → shard_by_batch_size()
            # asserts batch_size % dp_size == 0, so a partial batch will crash.
            # Use NeMo RL's maybe_pad_last_batch to pad with sample_mask=0.
            dp_size = h.config.get("dp_size", 1)
            mbs = h.config.get("policy", {}).get("train_micro_batch_size", 1)
            all_data = await asyncio.to_thread(
                _maybe_pad_batch, all_data, dp_size, mbs,
            )
            if all_data.size != original_size:
                logger.info(
                    "Padded partial batch from %d to %d samples "
                    "(dp_size=%d, mbs=%d, padding samples have sample_mask=0)",
                    original_size, all_data.size, dp_size, mbs,
                )

            logger.info(
                "Executing policy.train() with %d buffered microbatches for %s",
                num_buffered, h.model_id,
            )

            # Set learning rate if provided
            if learning_rate is not None:
                _set_learning_rate(h.policy, learning_rate)

            # Execute training — policy.train() does forward + backward + optimizer.step()
            train_result = await asyncio.to_thread(
                h.policy.train,
                data=all_data,
                loss_fn=h.loss_fn,
                eval_mode=False,
            )

            # Sync weights to inference engine
            if h.policy_generation is not None and not h.debug_train_only:
                logger.info("Refitting policy generation for %s", h.model_id)
                await asyncio.to_thread(
                    _refit_policy_generation,
                    h.policy,
                    h.policy_generation,
                    h.colocated_inference,
                )

            # Normalize metrics to common schema
            result = self.converter.backend_to_forward_backward_result(train_result, [])

            # Override with optimizer step format
            return {
                "success": True,
                "grad_norm": result.get("grad_norm", 0.0),
                "learning_rates": [],
                "model_id": h.model_id,
                "metrics": result.get("metrics", {}),
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="apply_optimizer_step", original_error=e,
            ) from e

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        """Sync weights between training policy and inference engine."""
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            if h.policy_generation is not None:
                await asyncio.to_thread(
                    _refit_policy_generation,
                    h.policy,
                    h.policy_generation,
                    h.colocated_inference,
                )
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="update_inference_weights",
                original_error=e,
            ) from e

    async def save_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        """Save model checkpoint via policy.save_checkpoint()."""
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            weights_path = f"{checkpoint_path}/weights"
            optimizer_path = f"{checkpoint_path}/optimizer"

            await asyncio.to_thread(
                h.policy.save_checkpoint,
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )

            logger.info("NeMo RL checkpoint saved to %s", checkpoint_path)
            return checkpoint_path

        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="save_checkpoint", original_error=e,
            ) from e

    async def load_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
    ) -> None:
        """Load checkpoint weights into policy, then sync to inference engine."""
        import os

        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            weights_path = f"{checkpoint_path}/weights"
            optimizer_path = f"{checkpoint_path}/optimizer"
            # Only load optimizer state if it exists (checkpoint may be weights-only)
            if not os.path.exists(optimizer_path):
                optimizer_path = None

            await asyncio.to_thread(
                h.policy.load_checkpoint,
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )

            # Sync loaded weights to inference engine
            if h.policy_generation is not None and not h.debug_train_only:
                logger.info("Refitting policy generation after checkpoint load for %s", h.model_id)
                await asyncio.to_thread(
                    _refit_policy_generation,
                    h.policy,
                    h.policy_generation,
                    h.colocated_inference,
                )

            logger.info("NeMo RL checkpoint loaded from %s", checkpoint_path)

        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="load_checkpoint", original_error=e,
            ) from e

    async def delete_model(self, handle: BackendHandle) -> None:
        """Tear down NeMo RL Policy and release GPU resources."""
        h: NemoRLHandle = handle  # type: ignore[assignment]
        try:
            # CHK007: Warn and discard pending buffered data on deletion.
            # CHK018: Acquire lock to prevent race with concurrent forward_backward.
            async with h._buffer_lock:
                if h.data_buffer:
                    logger.warning(
                        "Deleting model %s with %d buffered microbatches "
                        "(data will be discarded without training)",
                        h.model_id, len(h.data_buffer),
                    )
                    h.data_buffer.clear()

            if h.policy is not None:
                await asyncio.to_thread(h.policy.shutdown)
                logger.info("NeMo RL policy shut down for %s", h.model_id)

            if h.policy_generation is not None and h.policy_generation is not h.policy:
                try:
                    await asyncio.to_thread(h.policy_generation.shutdown)
                    logger.info("NeMo RL generation shut down for %s", h.model_id)
                except Exception:
                    pass  # Generation may share resources with policy

            logger.info("NeMo RL model %s deleted", h.model_id)

        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="delete_model", original_error=e,
            ) from e

    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        """Compute logprobs via policy.get_logprobs()."""
        result = await self.forward(handle, data, loss_fn="cross_entropy")
        logprobs_list = []
        for output in result.get("loss_fn_outputs", []):
            lp = output.get("logprobs", {})
            logprobs_list.append(lp.get("data", []))
        return logprobs_list


# ---------------------------------------------------------------------------
# Internal helper functions (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _init_nemo_rl_components(
    config_dict: Dict[str, Any],
    checkpoint_path: Optional[str],
    debug_train_only: bool,
):
    """
    Initialize NeMo RL Policy, generation engine, cluster, tokenizer, and loss fn.

    This is a blocking function — must be called via asyncio.to_thread().
    """
    import ray
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.algorithms.loss_functions import ClippedPGLossFn
    from transformers import AutoTokenizer

    policy_config = config_dict["policy"]
    loss_fn_config = config_dict["loss_fn"]
    cluster_config = config_dict["cluster"]

    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create virtual cluster
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=cluster_config["bundle_ct_per_node_list"],
        num_gpus_per_node=cluster_config.get("num_gpus_per_node", 8),
        max_colocated_worker_groups=cluster_config.get("max_colocated_worker_groups", 2),
    )

    # Load tokenizer
    model_name = policy_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create Policy
    weights_path = checkpoint_path if checkpoint_path else None
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        name_prefix="tinker_policy",
        init_optimizer=True,
        weights_path=weights_path,
        init_reference_model=not debug_train_only,
    )

    # Create loss function
    loss_fn = ClippedPGLossFn(loss_fn_config)

    # For inference, use the policy's built-in generation if not debug_train_only
    policy_generation = None
    if not debug_train_only:
        # In colocated mode, the policy IS the generation engine
        policy_generation = policy

    logger.info(
        "NeMo RL components initialized: model=%s, debug_train_only=%s",
        model_name, debug_train_only,
    )

    return policy, policy_generation, cluster, tokenizer, loss_fn


def _concatenate_batches(data_buffer: List) -> Any:
    """Concatenate multiple BatchedDataDicts into one."""
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if len(data_buffer) == 1:
        return data_buffer[0]

    return BatchedDataDict.from_batches(data_buffer)


def _refit_policy_generation(policy, policy_generation, colocated_inference: bool):
    """
    Sync weights from training policy to inference engine.

    Follows the pattern from nemo_rl/algorithms/grpo.py:refit_policy_generation().
    """
    import ray

    if colocated_inference:
        policy.offload_before_refit()
        if hasattr(policy_generation, "prepare_for_generation"):
            policy_generation.prepare_for_generation(tags=["weights"])

    try:
        if colocated_inference:
            # IPC ZMQ path for colocated inference
            refit_buffer_size_gb = policy.cfg.get("refit_buffer_size_gb", None)
            if refit_buffer_size_gb is not None:
                buffer_size_bytes = int(refit_buffer_size_gb * (1024 ** 3))
            else:
                buffer_size_bytes = None

            futures_train = policy.stream_weights_via_ipc_zmq(
                buffer_size_bytes=buffer_size_bytes,
            )
            if hasattr(policy_generation, "update_weights_via_ipc_zmq"):
                futures_inference = policy_generation.update_weights_via_ipc_zmq()
                ray.get(futures_train)
                ray.get(futures_inference)
            else:
                ray.get(futures_train)
        else:
            # Collective (NCCL) path for non-colocated inference
            futures_train = policy.broadcast_weights_for_collective()
            if hasattr(policy_generation, "update_weights_from_collective"):
                futures_inference = policy_generation.update_weights_from_collective()
                ray.get(futures_train)
                ray.get(futures_inference)
            else:
                ray.get(futures_train)
    except Exception as e:
        logger.error("Weight sync failed during refit: %s", e)
        raise

    if colocated_inference:
        policy.offload_after_refit()
        if hasattr(policy_generation, "prepare_for_generation"):
            policy_generation.prepare_for_generation(tags=["kv_cache"])


def _set_learning_rate(policy, learning_rate: float):
    """Set learning rate on the policy's optimizer."""
    try:
        worker_group = policy.worker_group
        # Set LR on all workers' optimizers
        import ray
        futures = worker_group.run_all_workers_single_data(
            "set_learning_rate",
            learning_rate=learning_rate,
        )
        ray.get(futures)
    except Exception as e:
        raise BackendError(
            f"Failed to set learning rate to {learning_rate}",
            backend="nemo_rl", operation="set_learning_rate", original_error=e,
        ) from e


def _maybe_pad_batch(batch, dp_size: int, mbs: int):
    """Pad batch to next multiple of mbs * dp_size.

    NeMo RL's shard_by_batch_size() asserts batch_size % dp_size == 0.
    Padding entries get sample_mask=0 so they don't affect loss computation.

    Logic duplicated from nemo_rl.algorithms.utils.maybe_pad_last_batch
    to avoid coupling TinkerCloud to NeMo RL internals.
    """
    import math
    import torch

    min_padding = (math.ceil(batch.size / (mbs * dp_size)) * mbs * dp_size) - batch.size
    if min_padding <= 0:
        return batch

    batch["input_ids"] = torch.cat([
        batch["input_ids"],
        batch["input_ids"][-1].unsqueeze(0).repeat(min_padding, 1),
    ])
    batch["input_lengths"] = torch.cat([
        batch["input_lengths"],
        batch["input_lengths"][-1].unsqueeze(0).repeat(min_padding),
    ])
    if "token_mask" in batch:
        batch["token_mask"] = torch.cat([
            batch["token_mask"],
            batch["token_mask"][-1].unsqueeze(0).repeat(min_padding, 1),
        ])
    batch["sample_mask"] = torch.cat([
        batch["sample_mask"],
        torch.zeros_like(batch["sample_mask"][-1]).unsqueeze(0).repeat(min_padding),
    ])
    # Pad all remaining 2D tensor fields that NeMo RL may access during sharding
    for key in ("advantages", "prev_logprobs", "generation_logprobs", "reference_policy_logprobs"):
        if key in batch:
            batch[key] = torch.cat([
                batch[key],
                batch[key][-1].unsqueeze(0).repeat(min_padding, 1),
            ])

    return batch
