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
from pathlib import Path
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend
from ...models.requests import Datum
from .config import NemoRLConfig
from ...core.loss_registry import clip_thresholds
from ...checkpoints.interchange import export_hf_adapter, stage_hf_adapter

if TYPE_CHECKING:
    from .generation import NemoRLRequestSubmitter

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
    loss_fn: Any = None              # TinkerSumPGLoss instance (RL)
    data_buffer: List = field(default_factory=list)  # R9 buffering
    max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE  # CHK006: bound buffer growth
    _buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # CHK018: thread safety
    image_preprocessor: Any = None   # ImagePreprocessor (VLM only)
    colocated_inference: bool = True
    refit_memory_ratio: float = 0.3      # share of free GPU memory for the IPC refit buffer
    rlve_config: Optional[Dict[str, Any]] = None
    wandb_config: Optional[Dict[str, Any]] = None
    created_at: str = ""
    training_run_id: str = ""
    debug_train_only: bool = False
    loss_fn_name: str = ""               # String name from last forward_backward()
    loss_fn_config: Optional[Dict[str, float]] = None  # per-call hyperparameters of the buffered batch
    generation_state: str = "generation_ready"  # "generation_ready" | "training_ready"
    _generation_state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _training_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Serialize optim_step GPU lifecycle
    ref_logprob_accumulator: Any = None  # Lazy _RefLogprobAccumulator (BUG-015)
    staleness_k: int = 0                 # Declared max sampler staleness (A4, specs/012)
    generation_synced_version: int = 0   # ver(S): weight version the inference engine holds
    dyn_mb_base: int = 0                 # Base dynamic-batching token budget (A5, specs/013)
    adam_mismatch_warned: bool = False   # the P4 AdamParams-mismatch warning fired once
    training_resident: bool = False      # model+optimizer GPU-resident in train mode; when
                                         # True the per-step prepare_for_training (BUG-005,
                                         # ~0.35 s/call, Q3-R2 phase decomposition) is skipped.
                                         # Invalidated by every offload path (refit, ckpt load).


class NemoRLBackend(TrainingBackend[NemoRLHandle]):
    config: NemoRLConfig
    # TinkerSumCELoss / TinkerSumPGLoss; cispo and dro have no NeMo RL loss.
    SUPPORTED_LOSS_FNS = frozenset({"cross_entropy", "importance_sampling", "ppo"})
    """
    NeMo RL backend — uses Policy.train() push-mode API.

    Implements the R9 buffering strategy: forward_backward() buffers data,
    apply_optimizer_step() calls policy.train() once with all buffered data.
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        # known keys -> typed config; the rest are raw MasterConfig overrides
        self.config, self.overrides = NemoRLConfig.split_overrides(overrides)
        self._converter = None
        self._builder = None
        # PERF-002: per-model batch accumulators for sample()
        self._submitters: Dict[str, "NemoRLRequestSubmitter"] = {}

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
            self._builder = NemoRLArgumentBuilder(overrides=self.overrides, config=self.config)
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
        resume_from: Optional[Path] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        staleness_k: int = 0,
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> NemoRLHandle:
        if objective != "language_modeling":
            raise BackendError(
                f"NeMo RL is a language-modeling backend; objective {objective!r} "
                f"requires a classification backend (automodel / megatron_bridge)",
                backend="nemo_rl", operation="create_model",
            )
        try:
            logger.info("[%s] Creating NeMo RL model %s", request_id, model_id)

            config_dict, hf_path = await asyncio.to_thread(
                self.builder.build_args,
                base_model=base_model,
                num_gpus=num_gpus,
                lora_config=lora_config,
                parallelism=parallelism,
                rl_config=rl_config,
                rollout_config=rollout_config,
                debug_train_only=debug_train_only,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
            )
            logger.info("[%s] NeMo RL config built, hf_path=%s", request_id, hf_path)

            # Policy(weights_path=...) goes straight to the checkpoint
            # manager's load_checkpoint, so it needs the <root>/weights dir,
            # with a foreign adapter staged into the layout that loader sniffs.
            init_weights_path = _stage_foreign_adapter(str(resume_from)) if resume_from else None

            policy, policy_generation, cluster, tokenizer, loss_fn = await asyncio.to_thread(
                _init_nemo_rl_components,
                config_dict=config_dict,
                checkpoint_path=init_weights_path,
                debug_train_only=debug_train_only,
                refit_memory_ratio=self.config.refit_buffer_memory_ratio,
            )

            handle = NemoRLHandle(
                model_id=model_id,
                backend_type="nemo_rl",
                # vLLM max_model_len == policy max_total_sequence_length (GAP-003)
                context_length=(config_dict.get("policy") or {}).get("max_total_sequence_length"),
                policy=policy,
                policy_generation=policy_generation,
                cluster=cluster,
                config=config_dict,
                tokenizer=tokenizer,
                loss_fn=loss_fn,
                hf_path=hf_path,
                colocated_inference=not debug_train_only,
                refit_memory_ratio=self.config.refit_buffer_memory_ratio,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
                created_at=datetime.now().isoformat(),
                training_run_id=model_id,
                debug_train_only=debug_train_only,
                staleness_k=staleness_k,
            )
            if staleness_k > 0:
                logger.info(
                    "[%s] staleness_k=%d declared: inference-engine refit deferred "
                    "while latest - synced <= k; ver(S) certified per sample",
                    request_id, staleness_k,
                )


            # Detect VLM via config (cheap — only reads config.json)
            try:
                from transformers import AutoConfig
                cfg = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
                is_vlm = (
                    hasattr(cfg, "vision_config")
                    or "VL" in cfg.__class__.__name__
                    or "Vision" in cfg.__class__.__name__
                )
                if is_vlm:
                    from .image_utils import ImagePreprocessor
                    handle.image_preprocessor = ImagePreprocessor(hf_path)
                    logger.info("[%s] VLM detected: %s", request_id, cfg.__class__.__name__)
            except Exception as e:
                logger.debug("[%s] VLM detection skipped: %s", request_id, e)

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
        handle: NemoRLHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Forward-only pass — compute logprobs without gradients.

        After init+refit, training workers have offloaded weights to CPU.
        Must sleep vLLM, move model to GPU, compute, then refit to restore.
        """
        try:
            batched_data = self.converter.forward_to_backend(data, handle.config)

            # Pad for dp_size alignment (get_logprobs uses shard_by_batch_size)
            dp_size = handle.config["dp_size"]
            mbs = handle.config["policy"]["train_micro_batch_size"]
            batched_data = await asyncio.to_thread(
                _maybe_pad_batch, batched_data, dp_size, mbs, handle.image_preprocessor,
            )

            # Sleep vLLM to free GPU memory for training workers. Recorded as
            # training_ready so a later sample wakes the engine first (weights
            # are still in sync, so that wake needs no refit).
            if handle.policy_generation is not None and handle.colocated_inference:
                async with handle._generation_state_lock:
                    handle.generation_state = "training_ready"
                await asyncio.to_thread(handle.policy_generation.finish_generation)

            # Use prepare_for_training(), not prepare_for_lp_inference(): the latter
            # offloads the optimizer, deadlocking a concurrent apply_optimizer_step
            # in pipelined SFT. train vs eval mode gives identical logprobs (no dropout).
            if not handle.training_resident:
                await asyncio.to_thread(handle.policy.prepare_for_training)
                handle.training_resident = True

            _ensure_dyn_mb_budget(handle, batched_data)
            result = await asyncio.to_thread(handle.policy.get_logprobs, batched_data)

            # No refit after forward(): read-only pass, inference weights already in
            # sync. Refitting here crashes (stale DTensor sharding after CPU→CUDA →
            # NCCL error in full_tensor()); refit only follows apply_optimizer_step().

            return self.converter.backend_to_forward_result(
                result, data, loss_fn=loss_fn,
            )

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="forward", original_error=e,
            ) from e

    async def forward_backward(
        self,
        handle: NemoRLHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
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
        try:
            # Empty data is a normal GRPO condition (all advantages zero in batch).
            # Return a no-op deferred result — training skips this batch.
            if not data:
                logger.warning(
                    "forward_backward: empty data for %s — no-op (all advantages likely zero)",
                    handle.model_id,
                )
                return {
                    "loss_fn_output_type": loss_fn,
                    "metrics": {},
                    "deferred": True,
                    "loss_fn_outputs": [],
                }

            # Convert data to NeMo RL BatchedDataDict (outside lock — CPU-bound)
            _t_conv = time.time()
            batched_data = self.converter.forward_backward_to_backend(
                data, loss_fn, handle.config,
                image_preprocessor=handle.image_preprocessor,
            )
            _dt_conv = time.time() - _t_conv

            # CHK018: Acquire lock for buffer access (concurrent requests)
            async with handle._buffer_lock:
                # CHK006: Enforce maximum buffer size to prevent unbounded memory growth
                if len(handle.data_buffer) >= handle.max_buffer_size:
                    raise BackendError(
                        f"Buffer full: {len(handle.data_buffer)} microbatches buffered "
                        f"(max_buffer_size={handle.max_buffer_size}). "
                        f"Call apply_optimizer_step() to flush the buffer.",
                        backend="nemo_rl",
                        operation="forward_backward",
                    )

                # CHK016: Validate field compatibility with existing buffer
                if handle.data_buffer:
                    existing_keys = set(handle.data_buffer[0].keys())
                    new_keys = set(batched_data.keys())
                    if existing_keys != new_keys:
                        raise BackendError(
                            f"Microbatch field mismatch: buffered has {sorted(existing_keys)}, "
                            f"new batch has {sorted(new_keys)}",
                            backend="nemo_rl",
                            operation="forward_backward",
                        )

                # One loss configuration per optimizer step: the buffer trains as
                # a single policy.train() call, so a differing config cannot be honoured.
                if handle.data_buffer and (handle.loss_fn_name, handle.loss_fn_config or {}) != (loss_fn, loss_fn_config or {}):
                    raise BackendError(
                        f"loss mismatch within one optimizer step: buffered "
                        f"{handle.loss_fn_name!r} {handle.loss_fn_config or {}}, new {loss_fn!r} {loss_fn_config or {}}",
                        backend="nemo_rl", operation="forward_backward",
                    )
                handle.data_buffer.append(batched_data)
                buffer_len = len(handle.data_buffer)
                # Store loss_fn name + config for apply_optimizer_step()
                handle.loss_fn_name = loss_fn
                handle.loss_fn_config = dict(loss_fn_config) if loss_fn_config else None

            logger.info(
                "Buffered microbatch %d for model %s (%d samples)",
                buffer_len, handle.model_id, len(data),
            )

            # CHK010: deferred contract — empty metrics, real metrics at optim_step.
            #
            # loss_fn_outputs by path:
            # - cross_entropy (SFT + forward_backward_custom coefficients): zero
            #   placeholders, shape-matched to `weights`, so client-side strict
            #   zip(logprobs, weights) survives (supervised/train.py, SDK custom path).
            # - RL loss fns: EMPTY list. This is the published deferred contract
            #   tinker-cookbook rl/train.py (T062) branches on — "forward_backward
            #   returns empty logprobs" — to consume the real per-sample logprobs
            #   from the matching optim_step response instead. Per-datum empty
            #   entries here would make that branch unreachable (001-P3 gap).
            placeholder_outputs = []
            for datum in data if loss_fn == "cross_entropy" else []:
                weights = datum.loss_fn_inputs.get("weights")
                n = len(weights.data) if weights is not None else 0
                placeholder_outputs.append({
                    "logprobs": {"data": [0.0] * n, "shape": [n], "dtype": "float32"},
                })
            logger.info(
                "fb phases (s): {'convert': %.3f} samples=%d placeholder_zeros=%d",
                _dt_conv, len(data),
                sum(len(p["logprobs"]["data"]) for p in placeholder_outputs),
            )
            return {
                "loss_fn_output_type": loss_fn,
                "metrics": {},
                "deferred": True,
                "loss_fn_outputs": placeholder_outputs,
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="forward_backward", original_error=e,
            ) from e

    async def apply_optimizer_step(
        self,
        handle: NemoRLHandle,
        learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
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
        try:
            # CHK018: Acquire lock to drain buffer atomically.
            # Snapshot and clear under the lock; concatenate outside it.
            async with handle._buffer_lock:
                if not handle.data_buffer:
                    # Empty buffer is normal when forward_backward received
                    # empty data (GRPO all-zero advantages). No-op — no GPU work.
                    logger.warning(
                        "apply_optimizer_step: empty buffer for %s — no-op",
                        handle.model_id,
                    )
                    return {
                        "success": True,
                        "grad_norm": 0.0,
                        "learning_rates": [],
                        "model_id": handle.model_id,
                        "metrics": {},
                        "loss_fn_outputs": [],
                    }

                # Atomic swap: take ownership of buffer, give handle a fresh one.
                # This ensures the buffer is cleared even if concatenation fails (CHK024).
                buffered_batches = handle.data_buffer
                handle.data_buffer = []
                num_buffered = len(buffered_batches)
                # The config that governed this buffer; a pipelined fb(N+1) may
                # overwrite h.loss_fn_config before we reach policy.train().
                step_loss_config = handle.loss_fn_config
                handle.loss_fn_config = None

            # Concatenate outside the lock (CPU-bound, no contention needed)
            phases: Dict[str, float] = {}  # Q3-R2 per-step cost decomposition
            _t = time.time()
            all_data = await asyncio.to_thread(_concatenate_batches, buffered_batches)
            phases["concat"] = time.time() - _t

            # CHK027: Warn if buffered sample count doesn't match train_global_batch_size
            # (check BEFORE padding so the warning reflects actual data volume)
            original_size = all_data.size
            gbs = handle.config["policy"]["train_global_batch_size"]
            if gbs > 0 and original_size != gbs:
                logger.warning(
                    "Buffered %d samples but train_global_batch_size=%d. "
                    "NeMo RL will process all %d samples. Verify this is intended.",
                    original_size, gbs, original_size,
                )

            # Pad partial batch if needed — policy.train() → shard_by_batch_size()
            # asserts batch_size % dp_size == 0, so a partial batch will crash.
            # Use NeMo RL's maybe_pad_last_batch to pad with sample_mask=0.
            dp_size = handle.config["dp_size"]
            mbs = handle.config["policy"]["train_micro_batch_size"]
            all_data = await asyncio.to_thread(
                _maybe_pad_batch, all_data, dp_size, mbs, handle.image_preprocessor,
            )
            if all_data.size != original_size:
                logger.info(
                    "Padded partial batch from %d to %d samples "
                    "(dp_size=%d, mbs=%d, padding samples have sample_mask=0)",
                    original_size, all_data.size, dp_size, mbs,
                )

            logger.info(
                "Executing policy.train() with %d buffered microbatches for %s",
                num_buffered, handle.model_id,
            )

            if learning_rate is not None:
                _set_learning_rate(handle.policy, learning_rate)

            # P4: betas/eps are fixed at creation (builder defaults match the
            # Tinker AdamParams contract: 0.9/0.95/1e-8); the worker exposes no
            # setter, so warn loudly if a client requests different values.
            if adam_params:
                _warn_on_adam_mismatch(handle, adam_params)

            # Serialize GPU lifecycle: sleep vLLM → train → refit.
            # Pipelined SFT loops fire multiple optim_step calls concurrently;
            # without this lock they race on the same Ray actors/GPU memory,
            # causing "No backend type associated with device type cpu" during refit.
            # The lock only gates GPU work — buffer drain + padding above run freely.
            async with handle._training_lock:
                # Transition to training state — generation requests after this
                # point will hit the safety net path in _ensure_generation_ready().
                async with handle._generation_state_lock:
                    handle.generation_state = "training_ready"

                try:
                    # BUG-006 fix: In colocated mode, vLLM generation engine holds
                    # model weights + KV cache on GPU after sampling. Must call
                    # finish_generation() to sleep vLLM workers and free GPU memory
                    # before loading the training model. Native NeMo RL does this
                    # explicitly in grpo.py after generation completes (line 1352).
                    if handle.policy_generation is not None and handle.colocated_inference:
                        logger.info("Finishing generation (sleep vLLM to free GPU memory)")
                        await asyncio.to_thread(handle.policy_generation.finish_generation)

                    # BUG-005 fix: NeMo RL requires prepare_for_training() before
                    # policy.train() to move model+optimizer from CPU→CUDA and set
                    # model.train(). Without this, the model stays on CPU after
                    # generation's offload_after_refit() and loss.backward() fails
                    # with "element 0 of tensors does not require grad".
                    if not handle.training_resident:
                        logger.info("Preparing policy for training (CPU→CUDA + train mode)")
                        _t = time.time()
                        await asyncio.to_thread(handle.policy.prepare_for_training)
                        phases["prepare"] = time.time() - _t
                        handle.training_resident = True
                    else:
                        # Q3-R2 adoption fix: nothing offloaded since the last train
                        # call, so the BUG-005 invariant already holds.
                        phases["prepare"] = 0.0

                    # The ratio's denominator is the client's sampling logprobs
                    # (Tinker contract). The earlier BUG-011 recompute against the
                    # current weights forced ratio == 1 and dropped the
                    # train/sample mismatch; the ~0.003 ratio it masked was the
                    # BUG-012 alignment defect, fixed in the converter.

                    active_loss_fn = _step_loss_fn(handle, step_loss_config)

                    # Pass gbs=actual size so NeMo RL shards correctly instead of
                    # defaulting to config train_global_batch_size.
                    _ensure_dyn_mb_budget(handle, all_data)
                    _t = time.time()
                    train_result = await asyncio.to_thread(
                        handle.policy.train,
                        data=all_data,
                        loss_fn=active_loss_fn,
                        eval_mode=False,
                        gbs=all_data.size,
                    )
                    phases["train"] = time.time() - _t

                    logger.info(
                        "policy.train() result: loss=%s, grad_norm=%s, keys=%s, mb_metrics=%s",
                        train_result.get("loss"), train_result.get("grad_norm"),
                        list(train_result.keys()),
                        {k: v for k, v in train_result.get("all_mb_metrics", {}).items()},
                    )

                    # BUG-015: weights advanced — bump version so pinned samplers
                    # (e.g. DPO's frozen reference) stop matching the live engine.
                    handle.weight_version += 1

                    if handle.policy_generation is not None and not handle.debug_train_only:
                        # A4 staleness-k: refit only when the engine would exceed
                        # the declared bound; otherwise wake it with the (stale)
                        # weights its level-1 sleep backed up. k=0 == old behavior.
                        staleness = handle.weight_version - handle.generation_synced_version
                        if staleness > handle.staleness_k:
                            logger.info(
                                "Refitting policy generation for %s (engine v%d -> v%d)",
                                handle.model_id, handle.generation_synced_version, handle.weight_version,
                            )
                            await asyncio.to_thread(
                                _refit_policy_generation,
                                handle.policy,
                                handle.policy_generation,
                                handle.colocated_inference,
                                handle.refit_memory_ratio,
                            )
                            handle.generation_synced_version = handle.weight_version
                            handle.training_resident = False  # refit offloaded the policy
                        else:
                            logger.info(
                                "ver(S): refit deferred for %s — engine stays at v%d, "
                                "latest v%d (staleness %d <= k=%d)",
                                handle.model_id, handle.generation_synced_version,
                                handle.weight_version, staleness, handle.staleness_k,
                            )
                            await asyncio.to_thread(
                                _wake_generation_stale,
                                handle.policy,
                                handle.policy_generation,
                                handle.colocated_inference,
                            )

                    # Transition back to generation_ready — sampling can now
                    # call generate() directly without offload/refit overhead.
                    async with handle._generation_state_lock:
                        handle.generation_state = "generation_ready"
                finally:
                    # If training failed before refit, reset state so sampling
                    # doesn't get permanently stuck in training_ready.
                    async with handle._generation_state_lock:
                        if handle.generation_state == "training_ready":
                            logger.warning(
                                "apply_optimizer_step failed before refit for %s "
                                "— resetting generation_state to generation_ready",
                                handle.model_id,
                            )
                            handle.generation_state = "generation_ready"

            # The worker records each microbatch loss after `loss *= dp*cp` (which
            # only cancels FSDP's mean-reduce for the backward) and all-reduces by
            # sum, so its global_loss is dp*cp times the batch loss. Report the
            # batch loss, which the API defines as the sum over the datums.
            _policy = handle.config["policy"]
            _cp = (_policy["megatron_cfg"]["context_parallel_size"]
                   if _policy["megatron_cfg"]["enabled"]
                   else _policy["dtensor_cfg"]["context_parallel_size"])
            _scale = handle.config["dp_size"] * _cp
            if isinstance(train_result, dict) and train_result.get("loss") is not None and _scale > 1:
                train_result["loss"] = train_result["loss"] / _scale
            result = self.converter.backend_to_forward_backward_result(
                train_result, [], loss_fn=handle.loss_fn_name,
            )

            # Extract per-sample training logprobs from curr_logprobs [B, S-1].
            # curr_logprobs are computed on pre-optimizer weights (correct KL semantics).
            # Use token_mask to extract response-only logprobs matching Miles format.
            _t = time.time()
            loss_fn_outputs = _extract_loss_fn_outputs(
                train_result.get("curr_logprobs"), all_data, original_size,
            )
            phases["extract"] = time.time() - _t
            logger.info(
                "optim_step phases (s): %s",
                {k: round(v, 3) for k, v in phases.items()},
            )

            metrics = dict(result.get("metrics", {}))
            if handle.loss_fn_name == "ppo":
                low, high = clip_thresholds(step_loss_config)
                metrics["clip_low_threshold"] = low
                metrics["clip_high_threshold"] = high

            return {
                "success": True,
                "grad_norm": result.get("grad_norm", 0.0),
                "learning_rates": [],
                "model_id": handle.model_id,
                "metrics": metrics,
                "loss_fn_outputs": loss_fn_outputs,
                # A4 ver(S): post-step version state for driver-side accounting
                "weight_version": handle.weight_version,
                "generation_synced_version": handle.generation_synced_version,
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="apply_optimizer_step", original_error=e,
            ) from e

    async def update_inference_weights(self, handle: NemoRLHandle) -> None:
        """Sync weights between training policy and inference engine."""
        try:
            if handle.policy_generation is not None:
                await asyncio.to_thread(
                    _refit_policy_generation,
                    handle.policy,
                    handle.policy_generation,
                    handle.colocated_inference,
                    handle.refit_memory_ratio,
                )
                handle.training_resident = False  # refit offloaded the policy
                async with handle._generation_state_lock:
                    handle.generation_state = "generation_ready"
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="update_inference_weights",
                original_error=e,
            ) from e

    async def save_checkpoint(
        self,
        handle: NemoRLHandle,
        root: Path,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        """Save model checkpoint via policy.save_checkpoint()."""
        if not persist:
            return  # ephemeral sampler save: the refit already delivered the weights
        try:
            local_path = str(root)

            weights_path = f"{local_path}/weights"
            checkpointing_cfg = handle.config["checkpointing"]

            # Optimizer state rides beside the weights so load_weights(optimizer=true)
            # can restore it; LoRA moments are small.
            await asyncio.to_thread(
                handle.policy.save_checkpoint,
                weights_path=weights_path,
                optimizer_path=f"{local_path}/optimizer",
                checkpointing_cfg=checkpointing_cfg,
            )

            # LoRA runs write the PEFT pair under <weights>/model; publish it
            # as the cross-backend interchange artifact (specs/007 §2.1).
            await asyncio.to_thread(
                export_hf_adapter, f"{weights_path}/model", local_path,
            )

            logger.info("NeMo RL checkpoint saved to %s (step %s)", local_path, step)

        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="save_checkpoint", original_error=e,
            ) from e

    async def load_checkpoint(
        self,
        handle: NemoRLHandle,
        root: Path,
        optimizer: bool = False,
    ) -> None:
        """Load checkpoint weights (and, on request, optimizer state) into the
        policy workers, then sync to the inference engine."""
        import os

        try:
            local_path = str(root)
            weights_path = _stage_foreign_adapter(local_path)
            optimizer_path = None
            if optimizer:
                optimizer_path = f"{local_path}/optimizer"
                if not os.path.isdir(optimizer_path):
                    raise BackendError(
                        f"Checkpoint {root} carries no optimizer state",
                        backend="nemo_rl", operation="load_checkpoint",
                    )

            await asyncio.to_thread(
                _policy_load_checkpoint, handle.policy, weights_path, optimizer_path,
            )

            if handle.policy_generation is not None and not handle.debug_train_only:
                # The policy was offloaded after the last refit, so the loaded
                # parameters sit on CPU; the refit streams DTensors and needs
                # them on GPU (all_gather has no CPU backend). Same sequence as
                # after an optimizer step: prepare_for_training, refit, offload.
                if not handle.training_resident:
                    await asyncio.to_thread(handle.policy.prepare_for_training)
                    handle.training_resident = True
                logger.info("Refitting policy generation after checkpoint load for %s", handle.model_id)
                await asyncio.to_thread(
                    _refit_policy_generation,
                    handle.policy,
                    handle.policy_generation,
                    handle.colocated_inference,
                    handle.refit_memory_ratio,
                )
                handle.training_resident = False  # the refit offloaded the policy
            else:
                handle.training_resident = False  # loaded state may not be GPU-resident
                async with handle._generation_state_lock:
                    handle.generation_state = "generation_ready"

            logger.info("NeMo RL checkpoint loaded from %s", root)

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="load_checkpoint", original_error=e,
            ) from e

    async def delete_model(self, handle: NemoRLHandle) -> None:
        """Tear down NeMo RL Policy and release GPU resources."""
        try:
            # CHK007: Warn and discard pending buffered data on deletion.
            # CHK018: Acquire lock to prevent race with concurrent forward_backward.
            async with handle._buffer_lock:
                if handle.data_buffer:
                    logger.warning(
                        "Deleting model %s with %d buffered microbatches "
                        "(data will be discarded without training)",
                        handle.model_id, len(handle.data_buffer),
                    )
                    handle.data_buffer.clear()

            if handle.policy is not None:
                await asyncio.to_thread(handle.policy.shutdown)
                logger.info("NeMo RL policy shut down for %s", handle.model_id)

            if handle.policy_generation is not None and handle.policy_generation is not handle.policy:
                try:
                    await asyncio.to_thread(handle.policy_generation.shutdown)
                    logger.info("NeMo RL generation shut down for %s", handle.model_id)
                except Exception:
                    # The generation worker group can share actors with the policy,
                    # which was shut down just above; log rather than hide it.
                    logger.warning("delete_model: generation shutdown failed for %s", handle.model_id, exc_info=True)

            self._submitters.pop(handle.model_id, None)

            logger.info("NeMo RL model %s deleted", handle.model_id)

        except Exception as e:
            raise BackendError(
                str(e), backend="nemo_rl", operation="delete_model", original_error=e,
            ) from e

    async def get_logprobs(
        self,
        handle: NemoRLHandle,
        data: List[Datum],
    ) -> List[Any]:
        """Compute logprobs via policy.get_logprobs()."""
        result = await self.forward(handle, data, loss_fn="cross_entropy")
        logprobs_list = []
        for output in result.get("loss_fn_outputs", []):
            lp = output.get("logprobs", {})
            logprobs_list.append(lp.get("data", []))
        return logprobs_list

    async def sample(
        self,
        handle: NemoRLHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample via generate_async() on the async vLLM Ray workers.

        specs/019: every row is its own engine request, so rows complete
        independently, a cancelled sample aborts its request, and an engine
        error fails only its own sample.

        BUG-015: `pinned_version` is the weight version a snapshot sampler was
        created at. The live vLLM engine is refit to the current policy every
        optim step, so a pinned sampler's compute_logprobs must NOT be served
        from it once versions diverge. v0-pinned samplers (DPO's frozen
        reference) are served from NeMo RL's built-in frozen reference model.
        """
        from .generation import NemoRLRequestSubmitter

        # BUG-015 routing: version-pinned logprob reads
        if prompt_logprobs and pinned_version is not None and pinned_version != handle.weight_version:
            if pinned_version == 0:
                return await self._reference_prompt_logprobs(handle, prompt_tokens)
            logger.warning(
                "[%s] sampler pinned at v%s but live weights at v%s — serving from "
                "LIVE weights (known-wrong; needs version-pinned sampling, 003)",
                request_id, pinned_version, handle.weight_version,
            )

        if handle.policy_generation is None:
            raise BackendError(
                "generation engine not initialized (debug_train_only mode?)",
                backend="nemo_rl", operation="sample",
            )
        # Wake/refit an engine a forward-only pass left asleep (fast path: one lock).
        await _ensure_generation_ready(handle)
        if handle.model_id not in self._submitters:
            self._submitters[handle.model_id] = NemoRLRequestSubmitter(
                max_in_flight=self.config.sample_max_in_flight_per_worker * handle.config["dp_size"],
            )
        result = await self._submitters[handle.model_id].submit(
            handle=handle,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            sampling_params=sampling_params or {},
            prompt_logprobs=prompt_logprobs,
        )
        # ver(S) monitor (A4): certify the version actually served against the
        # declared bound, and stamp it into the response. Versions are read
        # post-serve; sampling is not part of the model's ordered training
        # program (services.ordering), so a concurrent optim_step can bump
        # weight_version between serve and here — the stamp is best-effort.
        served_v = handle.generation_synced_version
        latest_v = handle.weight_version
        if latest_v - served_v > handle.staleness_k:
            raise BackendError(
                f"ver(S) certificate violation: served v{served_v}, latest "
                f"v{latest_v}, staleness {latest_v - served_v} > declared "
                f"k={handle.staleness_k}",
                backend="nemo_rl", operation="sample",
            )
        result["weight_version"] = served_v
        result["latest_weight_version"] = latest_v
        return result

    async def _reference_prompt_logprobs(
        self, h: "NemoRLHandle", prompt_tokens: List[int],
    ) -> Dict[str, Any]:
        """Serve compute_logprobs from the frozen reference model (W0).

        BUG-015 fix: DPO's reference sampler is a t=0 snapshot; the policy
        workers already hold frozen W0 (init_reference_model=True), so we call
        NeMo RL's public get_reference_policy_logprobs — no NeMo RL changes.
        Requests are batch-accumulated (DPO fires ~2x batch_size concurrent
        calls per step) and flushed as one call.
        """
        if h.ref_logprob_accumulator is None:
            h.ref_logprob_accumulator = _RefLogprobAccumulator()
        lp = await h.ref_logprob_accumulator.submit(self, h, prompt_tokens)
        # SDK reads prompt_logprobs only; one dummy sequence satisfies the schema.
        return {
            "sequences": [
                {"stop_reason": "length", "tokens": [], "logprobs": [], "text": None}
            ],
            "prompt_logprobs": lp,
        }

    async def _run_reference_logprobs(
        self, h: "NemoRLHandle", prompts: List[List[int]],
    ) -> List[List[Optional[float]]]:
        """One batched frozen-reference logprob pass. GPU lifecycle mirrors
        forward(): sleep vLLM -> prepare_for_training -> compute, NO refit
        (read-only; next generate() re-wakes via the safety net)."""
        import torch
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        batch_size = len(prompts)
        max_len = max(len(p) for p in prompts)
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        token_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
        for i, p in enumerate(prompts):
            input_ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
            token_mask[i, : len(p)] = 1.0
        data = BatchedDataDict({
            "input_ids": input_ids,
            "input_lengths": torch.tensor([len(p) for p in prompts], dtype=torch.long),
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size, dtype=torch.float32),
        })
        dp_size = h.config["dp_size"]
        mbs = h.config["policy"]["train_micro_batch_size"]
        data = await asyncio.to_thread(_maybe_pad_batch, data, dp_size, mbs, None)

        async with h._training_lock:
            async with h._generation_state_lock:
                h.generation_state = "training_ready"
            if h.policy_generation is not None and h.colocated_inference:
                await asyncio.to_thread(h.policy_generation.finish_generation)
            await asyncio.to_thread(h.policy.prepare_for_training)
            _ensure_dyn_mb_budget(h, data)
            out = await asyncio.to_thread(h.policy.get_reference_policy_logprobs, data)

        ref = out["reference_logprobs"]  # [B_padded, S]; row t = logprob(token_t | <t)
        results: List[List[Optional[float]]] = []
        for i, p in enumerate(prompts):
            row = ref[i]
            vals = row[1: len(p)].tolist()
            # BUG-013 convention: position 0 has no logprob
            results.append([None] + [float(v) for v in vals])
        return results

    async def prepare_for_generation(self, handle: NemoRLHandle) -> None:
        """Safety-net refit + wake if the engine was left in training state."""
        if handle.policy_generation is None:
            raise BackendError(
                "generation engine not initialized (debug_train_only mode?)",
                backend="nemo_rl", operation="prepare_for_generation",
            )
        await _ensure_generation_ready(handle)


def _step_loss_fn(h: NemoRLHandle, step_loss_config: Optional[Dict[str, float]]):
    """The loss object for this optimizer step. ppo always rebuilds the pure-sum
    PG loss with the call's clip range (contract defaults 0.8/1.2 when the call
    sends none); h.loss_fn is the unclipped importance_sampling loss."""
    if h.loss_fn_name == "cross_entropy":
        # Pure-sum CE (Tinker contract), not NeMo RL's mean-normalized NLLLoss
        # (BUG-015). Ships to Ray workers by reference (training is pip
        # install -e'd, so importable in the shared venv on server + workers).
        from .losses import TinkerSumCELoss
        return TinkerSumCELoss()
    if h.loss_fn_name == "ppo":
        from .losses import TinkerSumPGLoss
        low, high = clip_thresholds(step_loss_config)
        return TinkerSumPGLoss({
            **h.config["loss_fn"],
            "ratio_clip_min": 1.0 - low,
            "ratio_clip_max": high - 1.0,
        })
    return h.loss_fn  # TinkerSumPGLoss (importance_sampling, bounds that cannot bind)


async def _ensure_generation_ready(handle: NemoRLHandle) -> None:
    """Ensure the handle is ready for generation.

    Fast path (normal case): generation_state is already "generation_ready"
    after model init or after apply_optimizer_step refit — returns immediately.

    Safety net: if "training_ready" (e.g. pipelined SFT eval while prior
    optim_step is still running), wait for the training lock before refitting.
    Attempting refit while policy.train() is active deadlocks because both
    compete for the same Ray GPU workers.
    """
    async with handle._generation_state_lock:
        if handle.generation_state == "generation_ready":
            return

    # Must wait for any in-progress training to complete before refitting.
    # The _training_lock gates the GPU lifecycle: sleep vLLM → train → refit.
    # Without this, refit calls ray.get() on GPU workers busy with policy.train(),
    # causing a deadlock (BUG-010: pipelined SFT hang at eval boundary).
    logger.info(
        "Generation requested in training_ready state for %s — "
        "waiting for in-progress training to complete",
        handle.model_id,
    )
    async with handle._training_lock:
        # Re-check: training may have completed and already set generation_ready
        async with handle._generation_state_lock:
            if handle.generation_state == "generation_ready":
                return
            in_sync = handle.generation_synced_version == handle.weight_version
            logger.warning(
                "Generation requested with the engine asleep for %s — %s",
                handle.model_id,
                "waking it (weights already in sync)" if in_sync else "performing refit",
            )

        # Training has released GPU resources. A forward-only pass left the
        # engine asleep with current weights: wake it (a refit here trips
        # stale DTensor sharding). After an un-refit optimizer step: refit.
        if in_sync:
            await asyncio.to_thread(
                _wake_generation_stale,
                handle.policy,
                handle.policy_generation,
                handle.colocated_inference,
            )
        else:
            await asyncio.to_thread(
                _refit_policy_generation,
                handle.policy,
                handle.policy_generation,
                handle.colocated_inference,
                handle.refit_memory_ratio,
            )
            handle.generation_synced_version = handle.weight_version
        handle.training_resident = False  # the policy was offloaded either way

        async with handle._generation_state_lock:
            handle.generation_state = "generation_ready"


# ---------------------------------------------------------------------------
# Internal helper functions (called via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _policy_load_checkpoint(policy, weights_path: str, optimizer_path: Optional[str]) -> None:
    """Policy exposes save_checkpoint but no load; fan the workers' load_checkpoint
    out the same way (optimizer_path None = weights only)."""
    import ray
    ray.get(policy.worker_group.run_all_workers_single_data(
        "load_checkpoint", weights_path=weights_path, optimizer_path=optimizer_path,
    ))

def _stage_foreign_adapter(local_path: str) -> str:
    """Return the weights_path to load, materializing an interchange adapter
    written by ANOTHER backend into the layout automodel's loader sniffs
    (<weights>/model/adapter_model.safetensors). No-op for our own saves."""
    weights_path = f"{local_path}/weights"
    stage_hf_adapter(local_path, f"{weights_path}/model")
    return weights_path


def _init_nemo_rl_components(
    config_dict: Dict[str, Any],
    checkpoint_path: Optional[str],
    debug_train_only: bool,
    refit_memory_ratio: float = 0.3,
):
    """
    Initialize NeMo RL Policy, VllmGeneration, cluster, tokenizer, and loss fn.

    In colocated mode (default), VllmGeneration must initialize FIRST to allocate
    GPU memory for the vLLM engine, then Policy uses the remaining memory.

    This is a blocking function — must be called via asyncio.to_thread().
    """
    import ray
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.algorithms.utils import get_tokenizer
    from .losses import TinkerSumPGLoss

    policy_config = config_dict["policy"]
    loss_fn_config = config_dict["loss_fn"]
    cluster_config = config_dict["cluster"]

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=cluster_config["bundle_ct_per_node_list"],
        num_gpus_per_node=cluster_config.get("num_gpus_per_node", 8),
        max_colocated_worker_groups=cluster_config.get("max_colocated_worker_groups", 2),
    )

    # Load tokenizer (use NeMo RL's utility which sets pad_token_id if absent)
    model_name = policy_config["model_name"]
    tokenizer_config = {"name": model_name}
    tokenizer = get_tokenizer(tokenizer_config)

    loss_fn = TinkerSumPGLoss(loss_fn_config)

    policy_generation = None
    if not debug_train_only:
        generation_config = policy_config.get("generation")
        if generation_config and generation_config.get("backend") == "vllm":
            from nemo_rl.models.generation import configure_generation_config
            from nemo_rl.models.generation.vllm import VllmGeneration

            generation_config["model_name"] = model_name

            # Set stop_strings BEFORE configure_generation_config() so it sets
            # skip_tokenizer_init=False (vLLM needs the tokenizer for text matching).
            # Use stop_strings, not stop_token_ids: vLLM keeps stop strings in the
            # output (renderers need them for parse_response) but strips stop ids.
            stop_strs = set()
            for token_str in ["<|im_end|>", "<|eot_id|>", "<|end▁of▁sentence|>"]:
                if len(tokenizer.encode(token_str, add_special_tokens=False)) == 1:
                    stop_strs.add(token_str)
            if stop_strs:
                generation_config["stop_strings"] = list(stop_strs)
                logger.info("Set stop_strings from tokenizer: %s", list(stop_strs))

            # configure_generation_config derives skip_tokenizer_init from
            # stop_strings — do NOT override it manually.
            generation_config = configure_generation_config(generation_config, tokenizer)

            logger.info("Initializing VllmGeneration (colocated mode)...")
            policy_generation = VllmGeneration(
                cluster=cluster,
                config=generation_config,
                name_prefix="tinker_vllm",
            )
            policy_generation.finish_generation()
            logger.info("VllmGeneration initialized successfully")

    # Create Policy (after vLLM in colocated mode — uses remaining GPU memory)
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

    # Prepare refit info — needed for weight sync between policy and generation
    if policy_generation is not None:
        state_dict_info = policy.prepare_refit_info()
        policy_generation.prepare_refit_info(state_dict_info)
        logger.info("Refit info prepared for weight sync")

        # Do initial weight sync so generation has the correct weights
        logger.info("Performing initial weight sync (refit)...")
        _refit_policy_generation(policy, policy_generation, colocated_inference=True,
                                 memory_ratio=refit_memory_ratio)
        logger.info("Initial weight sync complete")

    logger.info(
        "NeMo RL components initialized: model=%s, debug_train_only=%s, "
        "generation=%s",
        model_name, debug_train_only,
        type(policy_generation).__name__ if policy_generation else "None",
    )

    return policy, policy_generation, cluster, tokenizer, loss_fn


def _concatenate_batches(data_buffer: List) -> Any:
    """Concatenate multiple BatchedDataDicts into one."""
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    if len(data_buffer) == 1:
        return data_buffer[0]

    return BatchedDataDict.from_batches(data_buffer)


def _refit_policy_generation(policy, policy_generation, colocated_inference: bool,
                             memory_ratio: float = 0.3):
    """
    Sync weights from training policy to inference engine (VllmGeneration).

    Follows the pattern from nemo_rl/algorithms/grpo.py:refit_policy_generation().
    """
    import ray

    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    try:
        if colocated_inference:
            # IPC ZMQ path for colocated inference
            buffer_size_bytes = int(policy.get_free_memory_bytes() * memory_ratio)

            futures_train = policy.stream_weights_via_ipc_zmq(
                buffer_size_bytes=buffer_size_bytes,
            )
            futures_inference = policy_generation.update_weights_via_ipc_zmq()
            ray.get(futures_train)
            ray.get(futures_inference)
        else:
            # Collective (NCCL) path for non-colocated inference
            futures_train = policy.broadcast_weights_for_collective()
            futures_inference = policy_generation.update_weights_from_collective()
            ray.get(futures_train)
            ray.get(futures_inference)
    except Exception as e:
        logger.error("Weight sync failed during refit: %s", e)
        raise

    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])


def _wake_generation_stale(policy, policy_generation, colocated_inference: bool):
    """Restore generation WITHOUT streaming new weights: vLLM's level-1 sleep
    backed its weights up to CPU, so a plain wake restores what the engine held
    before it slept. Two callers: the A4 staleness-k skip after an optimizer
    step (stale by a declared bound), and generation after a forward-only pass
    (weights unchanged, so exactly current)."""
    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation()


def _applied_grad_clip(max_grad_norm) -> float:
    """The clip the API sees: 0.0 (off) when the worker bound cannot bind."""
    from .loss_config import NO_GRAD_CLIP
    if max_grad_norm is None or float(max_grad_norm) >= NO_GRAD_CLIP:
        return 0.0
    return float(max_grad_norm)


def _warn_on_adam_mismatch(h: "NemoRLHandle", adam_params: Dict[str, Any]) -> None:
    """P4: compare client-requested Adam params against the applied config.

    Betas/eps/weight_decay/clip are baked into the optimizer at model creation;
    NeMo RL's worker only exposes set_learning_rate, so differing requests
    cannot be honored per-step without an upstream setter. Warn once per handle.
    """
    if h.adam_mismatch_warned:
        return
    kwargs = h.config["policy"]["optimizer"]["kwargs"]
    applied = {
        "beta1": kwargs["betas"][0],
        "beta2": kwargs["betas"][1],
        "eps": kwargs["eps"],
        "weight_decay": kwargs["weight_decay"],
        "grad_clip_norm": _applied_grad_clip(h.config["policy"]["max_grad_norm"]),
    }
    mismatches = {
        k: (v, applied[k]) for k, v in adam_params.items()
        if k in applied and v is not None and abs(float(v) - float(applied[k])) > 1e-12
    }
    if mismatches:
        logger.warning(
            "[P4] client AdamParams differ from applied optimizer config and "
            "CANNOT be applied per-step (requested vs applied): %s — fix at "
            "model creation (builder) or add an upstream NeMo RL setter.",
            mismatches,
        )
    h.adam_mismatch_warned = True


def _set_learning_rate(policy, learning_rate: float):
    """Set learning rate on the policy's optimizer via worker RPC."""
    import ray

    try:
        futures = policy.worker_group.run_all_workers_single_data(
            "set_learning_rate",
            learning_rate=learning_rate,
        )
        ray.get(futures)
        logger.info("Set learning rate to %s", learning_rate)
    except Exception as e:
        raise BackendError(
            f"could not set learning rate {learning_rate} on the workers: {e}",
            backend="nemo_rl", operation="apply_optimizer_step", original_error=e,
        ) from e


class _RefLogprobAccumulator:
    """Batch-accumulate frozen-reference logprob requests (BUG-015).

    Flush-window batching (PERF-002): concurrent compute_logprobs calls
    within a flush window are served by ONE get_reference_policy_logprobs
    pass. Single event loop => the
    drain-then-exit check under the lock is race-free.
    """

    def __init__(self, flush_interval_s: float = 0.05):
        self._flush_interval = flush_interval_s
        self._pending: List[Any] = []  # (prompt_tokens, future)
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    async def submit(self, backend, handle, prompt_tokens: List[int]):
        fut = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._pending.append((prompt_tokens, fut))
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(
                    self._flush_loop(backend, handle)
                )
        return await fut

    async def _flush_loop(self, backend, handle):
        while True:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                pending, self._pending = self._pending, []
            if not pending:
                return
            try:
                results = await backend._run_reference_logprobs(
                    handle, [p for p, _ in pending]
                )
                for (_, fut), lp in zip(pending, results):
                    if not fut.done():
                        fut.set_result(lp)
            except Exception as e:  # propagate to all waiters
                for _, fut in pending:
                    if not fut.done():
                        fut.set_exception(
                            BackendError(
                                str(e), backend="nemo_rl",
                                operation="reference_logprobs", original_error=e,
                            )
                        )
            async with self._lock:
                if not self._pending:
                    return


def _ensure_dyn_mb_budget(h: "NemoRLHandle", data) -> None:
    """Raise the dynamic-batching token budgets to admit this batch (A5).

    The builder sets a memory-sized base budget (default 8192 tokens);
    NeMo RL asserts every sample fits its micro-batch budget after
    sequence_length_round padding. Setting max(base, longest sample)
    before each train/logprob call keeps short-sequence packing bounded
    while a long sample degrades to its own micro-batch — the old MBS=1
    worst case — instead of a hard assert. Non-ratcheting: a long batch
    does not inflate the budget for later short batches.
    """
    policy = h.policy
    dyn = getattr(policy, "cfg", {}).get("dynamic_batching") if policy else None
    if not dyn or not dyn.get("enabled"):
        return
    lengths = data.get("input_lengths") if hasattr(data, "get") else None
    if lengths is None or len(lengths) == 0:
        return
    if h.dyn_mb_base <= 0:
        h.dyn_mb_base = int(dyn.get("train_mb_tokens", 0))
    rounded = -(-int(lengths.max()) // 64) * 64  # sequence_length_round
    budget = max(h.dyn_mb_base, rounded)
    for key in ("train_mb_tokens", "logprob_mb_tokens"):
        if dyn.get(key) != budget:
            logger.info(
                "dynamic batching %s: %s -> %d (longest sample %d, base %d)",
                key, dyn.get(key), budget, int(lengths.max()), h.dyn_mb_base,
            )
            dyn[key] = budget


def _maybe_pad_batch(batch, dp_size: int, mbs: int, image_preprocessor=None):
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

    # Most fields pad by repeating the last sample (1D or 2D)
    for key in ("input_ids", "input_lengths", "token_mask",
                "advantages", "prev_logprobs", "generation_logprobs", "reference_policy_logprobs"):
        if key in batch:
            t = batch[key]
            batch[key] = torch.cat([t, t[-1:].expand(min_padding, *t.shape[1:])])

    # sample_mask is zero-padded so padded samples don't contribute to loss
    sm = batch["sample_mask"]
    batch["sample_mask"] = torch.cat([sm, torch.zeros(min_padding, dtype=sm.dtype, device=sm.device)])

    # Pad PackedTensor multimodal fields with None entries (VLM only)
    if image_preprocessor is not None:
        from nemo_rl.data.multimodal_utils import PackedTensor
        for key in image_preprocessor.multimodal_keys:
            if key in batch.data:
                batch[key] = PackedTensor(
                    batch[key].tensors + [None] * min_padding,
                    batch[key].dim_to_pack,
                )
    return batch


def _extract_loss_fn_outputs(curr_logprobs, batched_data, original_size: int):
    """Convert curr_logprobs [B, S-1] to per-sample loss_fn_outputs.

    Returns datum-aligned FULL-LENGTH logprobs (Miles parity): position p holds
    the logprob of target token x_p, length input_lengths[i]-1 == the wire
    tensor length (mask/logprobs/advantages). Do NOT mask-compress — the
    cookbook indexes these with a full-length action mask
    (rl/metrics.py compute_kl_sample_train).
    Only produces outputs for the first ``original_size`` samples (skips padding).

    Returns:
        List of dicts: [{"logprobs": {"data": [...], "shape": [N], "dtype": "float32"}}, ...]
    """
    if curr_logprobs is None:
        return []

    input_lengths = batched_data.get("input_lengths") if hasattr(batched_data, "get") else None
    loss_fn_outputs = []

    for i in range(original_size):
        lp_i = curr_logprobs[i]  # [S_max-1], right-padded; real values in [:seq_len-1]

        if input_lengths is not None:
            lp_i = lp_i[: int(input_lengths[i]) - 1]
        response_lp = lp_i.cpu().tolist()

        loss_fn_outputs.append({
            "logprobs": {
                "data": response_lp,
                "shape": [len(response_lp)],
                "dtype": "float32",
            }
        })

    return loss_fn_outputs
