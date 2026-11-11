"""
veRL backend — binds veRL's upstream Tinker split-training primitives
(verl.workers.engine_workers_tinker) behind the TrainingBackend ABC.

Execution model: EAGER (Miles-shaped, NOT R9-buffered). veRL upstream ships
workers with decoupled forward_backward (grad-accumulating,
returns per-token logprobs) + optimizer_step (accepts lr/betas/eps/
weight_decay) + optimizer_zero_grad. forward_backward here does real GPU
work and returns real logprobs; apply_optimizer_step applies AdamParams and
steps once.

Rollout: every model is a TinkerActorRolloutRefWorker group (actor_rollout
role). Its vLLM server replicas (LLMServerManager) are built at the model's
first sample, in hybrid timeshare: replicas sleep during training ops and
wake with a naive weight sync (CheckpointEngineManager) for sampling. A model
that never samples (a forward-only teacher) never builds one.
update_inference_weights is lazy: versions are tracked on the handle and the
sync happens on the next sample/prepare_for_generation, so pure-training runs
never pay for it.

GPUs: the server's models share one placement group of whole GPUs
(_SharedGpus), verl's own colocation; a worker takes 1/models_per_gpu of its
GPU, so several models are resident on the same devices.
"""
import asyncio
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import torch

from ..base import BackendError, BackendHandle, SampleRequestError, TrainingBackend, UnsupportedFeatureError
from ...utils.model_config import read_raw_hf_config
from ..objectives import classification_spec
from .config import VerlConfig
from .losses import padded_logprobs
from ...models.requests import AdamParams, Datum, OptimParams
from ...checkpoints.interchange import (
    HF_ADAPTER_DIRNAME,
    find_hf_adapter,
    read_adapter_config,
)

logger = logging.getLogger(__name__)

# vLLM's max_logprobs: the top-k it computes per position (handle.max_topk_logprobs).
VLLM_MAX_LOGPROBS = 20


def _set_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


@dataclass
class VerlHandle(BackendHandle):
    """veRL-specific runtime state."""

    worker_group: Any = None          # RayWorkerGroup over TinkerActorRolloutRefWorker
    config: Dict = field(default_factory=dict)
    # the composed verl trainer config the workers and the sampler read
    trainer_config: Any = field(kw_only=True)
    dp_size: int = 1
    loss_fn_name: str = ""            # currently installed loss fn on workers
    created_at: str = ""
    # rollout state (hybrid timeshare); the three managers are None until the first sample
    llm_manager: Any = None           # LLMServerManager (replicas + balancer)
    llm_client: Any = None            # LLMServerClient (token-in-token-out)
    ckpt_manager: Any = None          # CheckpointEngineManager (sleep/sync)
    rollout_awake: bool = False
    rollout_synced_version: int = -1  # weight_version last synced to vLLM
    lora_rank: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # FIFO per handle
    # Sample concurrency: generates run OUTSIDE _lock so vLLM batches them;
    # training ops quiesce active samplers first (_quiesce_samplers).
    _active_samplers: int = 0
    _samplers_drained: asyncio.Event = field(default_factory=_set_event)


class _SharedGpus:
    """The placement group every model's workers live in: one bundle per GPU,
    up to `models_per_gpu` workers per bundle. Built by the first model,
    removed with the last. Both methods block on Ray: call them off the event loop."""

    def __init__(self, models_per_gpu: int):
        self.models_per_gpu = models_per_gpu
        self.pool: Any = None            # RayResourcePool, None while no model exists
        self.model_ids: set = set()
        self._lock = threading.Lock()

    def acquire(self, model_id: str, num_gpus: int):
        """The pool for one more model of `num_gpus` workers, its placement group ready.

        Raises:
            BackendError: the GPUs hold models_per_gpu models already, or the
                model's GPU count is not the pool's.
        """
        from verl.single_controller.ray import RayResourcePool

        with self._lock:
            if len(self.model_ids) >= self.models_per_gpu:
                raise BackendError(
                    f"the GPUs hold {len(self.model_ids)} models, the limit TINKERCLOUD_VERL_MODELS_PER_GPU sets "
                    f"({self.models_per_gpu}): {', '.join(sorted(self.model_ids))}",
                    backend="verl", operation="create_model",
                )
            if self.pool is None:   # no model resident: the first one builds the pool, the last one removes it
                pool = RayResourcePool(
                    process_on_nodes=[num_gpus], max_colocate_count=self.models_per_gpu, name_prefix=SHARED_POOL_NAME_PREFIX,
                )
                pool.get_placement_groups()
                self.pool = pool
            elif self.pool.store[0] != num_gpus:
                raise BackendError(
                    f"model {model_id} asks for {num_gpus} GPUs; this server's models share {self.pool.store[0]}",
                    backend="verl", operation="create_model",
                )
            self.model_ids.add(model_id)
            return self.pool

    def release(self, model_id: str) -> None:
        """Give up `model_id`'s share; the placement group goes with the last model."""
        import ray

        with self._lock:
            if self.model_ids == {model_id}:
                for pg in self.pool.pgs:
                    ray.util.remove_placement_group(pg)
                self.pool = None
            # last, so that a release that failed above repeats the same steps
            self.model_ids.remove(model_id)


class VerlBackend(TrainingBackend[VerlHandle]):
    """Full or LoRA models on verl's FSDP engine. Models share the server's GPUs
    (`models_per_gpu`); a model builds its vLLM sampler at its first sample, and
    one model per server holds a sampler."""
    config: VerlConfig

    SUPPORTED_LOSS_FNS = frozenset({"cross_entropy", "importance_sampling", "ppo"})  # losses.LOSS_FNS

    SUPPORTED_OPTIMIZERS = frozenset({"adamw"})   # no Muon on this path
    NK_TARGET_LOSS_FNS = frozenset({"cross_entropy"})

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self.config = VerlConfig.from_env()
        self._converter = None
        self._builder = None
        self._gpus = _SharedGpus(self.config.models_per_gpu)
        # vLLM server actors are Ray-named by replica rank alone, so one model per Ray
        # cluster can own a sampler: the first to sample or to be prepared for it, until deleted.
        self._sampler_model_id: Optional[str] = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import VerlDataConverter
            self._converter = VerlDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import VerlArgumentBuilder
            self._builder = VerlArgumentBuilder(overrides=self.overrides)
        return self._builder

    def model_repositories(self, base_model: str) -> List[str]:
        """The model directory or hub id the builder loads (VerlArgumentBuilder.resolve_hf_path)."""
        from .builder import VerlArgumentBuilder
        return [VerlArgumentBuilder.resolve_hf_path(base_model)]

    # ---------------------------------------------------------------- create

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
        native_root: Optional[Path] = None,
        *,
        optimizer: str,
    ) -> VerlHandle:
        classification = classification_spec(await asyncio.to_thread(read_raw_hf_config, base_model))
        if classification is not None:
            raise BackendError(
                f"verl is a language-modeling backend; {base_model!r} declares a "
                f"{classification.objective.value} head",
                backend="verl", operation="create_model",
            )
        try:
            cfg = self.builder.build_args(
                base_model=base_model, num_gpus=num_gpus, lora_config=lora_config,
                parallelism=parallelism, rl_config=rl_config,
                rollout_config=rollout_config, max_seq_len=self.config.max_seq_len,
            )
            dp_size = _dp_size(num_gpus, cfg["engine"]["ulysses_sequence_parallel_size"])   # before any actor exists

            # Interchange adapters attach at model build (verl's PEFT path wraps
            # the module before FSDP), and nothing is loaded at create any more
            # (D9): a foreign adapter resumes only through load_checkpoint's
            # native path. verl is UNGATED; see specs/025.
            adapter_dir = None

            trainer_config = await asyncio.to_thread(_compose_actor_rollout_config, cfg)
            worker_group = await asyncio.to_thread(self._boot, trainer_config, model_id, num_gpus)
            handle = VerlHandle(
                model_id=model_id,
                backend_type="verl",
                # vLLM max_model_len == rollout prompt_length + response_length
                context_length=int(cfg["rollout"]["prompt_length"]) + int(cfg["rollout"]["response_length"]),
                max_topk_logprobs=VLLM_MAX_LOGPROBS,
                worker_group=worker_group,
                config=cfg,
                trainer_config=trainer_config,
                hf_path=cfg["hf_path"],
                dp_size=dp_size,
                created_at=datetime.now().isoformat(),
                lora_rank=int(cfg["model"]["lora_rank"] or 0),
            )
            logger.info(
                "[%s] verl model %s created (dp=%d, adapter=%s)",
                request_id, model_id, handle.dp_size, adapter_dir,
            )
            return handle
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="create_model", original_error=e) from e

    # ------------------------------------------------------------- train ops

    async def forward_backward(
        self,
        handle: VerlHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                await asyncio.to_thread(self._ensure_training_ready, handle)
                await asyncio.to_thread(self._ensure_loss_fn, handle, loss_fn)
                td = await asyncio.to_thread(self._to_tensordict, handle, data, loss_fn)
                out = await asyncio.to_thread(_wg_call, handle.worker_group.forward_backward, td)
                logprobs = await asyncio.to_thread(padded_logprobs, out, td)
                loss_fn_outputs = self.converter.extract_logprobs(logprobs, data)
                metrics = _scalar_metrics(out)
                return {
                    "loss_fn_output_type": loss_fn,
                    "loss_fn_outputs": loss_fn_outputs,
                    "metrics": metrics,
                    "deferred": False,
                }
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="forward_backward", original_error=e) from e

    async def apply_optimizer_step(
        self,
        handle: VerlHandle,
        optim_params: OptimParams,
    ) -> Dict[str, Any]:
        adam = cast(AdamParams, optim_params)   # the only family this backend declares
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                await asyncio.to_thread(self._ensure_training_ready, handle)
                params: Dict[str, Any] = {
                    "lr": float(adam.learning_rate),
                    "betas": (float(adam.beta1), float(adam.beta2)),
                    "eps": float(adam.eps),
                    "weight_decay": float(adam.weight_decay),
                }
                if adam.grad_clip_norm > 0:
                    logger.warning("verl optim_step: grad_clip_norm is set at engine build, not per step")

                metrics = await asyncio.to_thread(
                    _wg_scalar_call, handle.worker_group.optimizer_step, params,
                )
                handle.weight_version += 1
                grad_norm = metrics.get("grad_norm")
                return {"success": True, "grad_norm": grad_norm, "metrics": metrics}
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="apply_optimizer_step", original_error=e) from e

    async def forward(
        self,
        handle: VerlHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        logprobs = await self.get_logprobs(handle, data)
        return {"loss_fn_output_type": loss_fn, "loss_fn_outputs": logprobs, "metrics": {}}

    async def get_logprobs(
        self,
        handle: VerlHandle,
        data: List[Datum],
    ) -> List[Any]:
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                await asyncio.to_thread(self._ensure_training_ready, handle)
                # the infer path runs the loss too: the engine's logits hook, which
                # [N, K] targets need, is the installed loss function
                await asyncio.to_thread(self._ensure_loss_fn, handle, "cross_entropy")
                td = await asyncio.to_thread(self._to_tensordict, handle, data, "cross_entropy", True)
                out = await asyncio.to_thread(_wg_call, handle.worker_group.compute_log_prob, td)
                logprobs = await asyncio.to_thread(padded_logprobs, out, td)
                return self.converter.extract_logprobs(logprobs, data)
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="get_logprobs", original_error=e) from e

    # ------------------------------------------------------------ lifecycle

    async def update_inference_weights(self, handle: VerlHandle) -> None:
        # Lazy: weight_version already advanced at optim_step; the actual
        # vLLM sync happens on the next sample/prepare_for_generation
        # (version-gated), so pure-training runs never pay wake/sync cost.
        logger.debug(
            "verl update_inference_weights deferred (v%d, synced v%d)",
            handle.weight_version, handle.rollout_synced_version,
        )
        return None

    async def save_checkpoint(
        self,
        handle: VerlHandle,
        root: Path,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        if not persist:
            return  # ephemeral sampler save: the engines already hold the weights
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                local_path = str(root)
                await asyncio.to_thread(
                    handle.worker_group.save_checkpoint, local_path, None, step or handle.weight_version,
                )
                if handle.lora_rank > 0:
                    # verl writes FSDP-sharded .pt only; no PEFT export path
                    # yet (specs/007 §3.5) — say so rather than leaving a
                    # silently unmigratable checkpoint behind.
                    logger.warning(
                        "verl checkpoint %s has no %s/ (verl PEFT export unimplemented): "
                        "usable for verl resume, not for cross-backend migration",
                        local_path, HF_ADAPTER_DIRNAME,
                    )
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="save_checkpoint", original_error=e) from e

    async def load_checkpoint(self, handle: VerlHandle, root: Path,
                              optimizer: bool = False) -> None:
        if optimizer:
            raise BackendError("verl restores weights only; optimizer state is not restored",
                               backend="verl", operation="load_checkpoint")
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                local_path = str(root)
                if find_hf_adapter(local_path):
                    # PeftModel.from_pretrained wraps pre-FSDP: only create_model
                    # can attach an interchange adapter.
                    raise BackendError(
                        "verl imports an interchange adapter at create_model "
                        "(lora_adapter_path), not via load_checkpoint",
                        backend="verl", operation="load_checkpoint",
                    )
                await asyncio.to_thread(handle.worker_group.load_checkpoint, local_path)
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="load_checkpoint", original_error=e) from e

    async def delete_model(self, handle: VerlHandle) -> None:
        # under the lock: a first sample may be building the sampler, and its actors must exist to be killed
        async with handle._lock:
            try:
                await self._quiesce_samplers(handle)
                if self._sampler_model_id == handle.model_id:
                    # rollout server actors first (they hold engine refs into workers)
                    await asyncio.to_thread(_kill_named_actors, VLLM_SERVER_NAME_PREFIX)
                    self._sampler_model_id = None
                handle.llm_manager = None
                handle.llm_client = None
                handle.ckpt_manager = None
                handle.rollout_awake = False
                await asyncio.to_thread(_kill_named_actors, worker_name_prefix(handle.model_id))
                handle.worker_group = None
                await asyncio.to_thread(self._gpus.release, handle.model_id)
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="delete_model", original_error=e) from e

    async def sample(
        self,
        handle: VerlHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
        topk_sample_logprobs: int = 0,
        topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        if topk_prompt_logprobs > 0:  # the verl sample path scores sampled positions only
            raise SampleRequestError("topk_prompt_logprobs is not supported on the verl backend", backend="verl")
        vllm_sp = _to_vllm_sampling_params(sampling_params)
        if topk_sample_logprobs > 0:
            if vllm_sp["temperature"] == 0:  # verl refuses it: workers/rollout/utils.py, "rollout.topk_log_probs requires stochastic sampling"
                raise SampleRequestError(
                    "topk_sample_logprobs needs temperature > 0 on the verl backend", backend="verl")
            vllm_sp["topk_log_probs"] = topk_sample_logprobs
        # Hybrid timeshare: generation must not overlap this model's training
        # ops. Register as an active sampler under the FIFO lock (which
        # serializes the wake/weight-sync), then generate WITHOUT the lock so
        # concurrent sample requests batch inside vLLM; the model's training ops
        # quiesce samplers via _quiesce_samplers first. Another model's passes
        # (a teacher's forward) run beside the awake sampler on the same GPUs.
        async with handle._lock:
            try:
                await self._ready_sampler(handle)

                if pinned_version is not None and pinned_version != handle.weight_version:
                    # vLLM serves the live (base+adapter) route; a pinned
                    # nonzero version is not honorable — BUG-015 aliasing
                    # class, logged loudly rather than silently aliased.
                    logger.warning(
                        "[%s] pinned_version=%s not honored for %s (verl serves "
                        "the live engine, v%d) — BUG-015 aliasing risk",
                        request_id, pinned_version, handle.model_id, handle.weight_version,
                    )
                if prompt_logprobs:
                    logger.warning(
                        "[%s] prompt_logprobs unsupported on verl sample path "
                        "(use forward/get_logprobs); returning None", request_id,
                    )
                handle._active_samplers += 1
                handle._samplers_drained.clear()
            except (BackendError, UnsupportedFeatureError):
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="sample", original_error=e) from e
        try:
            try:
                max_tokens = vllm_sp.get("max_tokens")

                async def _one(i: int):
                    sp_i = dict(vllm_sp)
                    if sp_i.get("seed") is not None:
                        # vLLM seeds per request; a fixed seed across the
                        # num_samples fanout would return identical samples.
                        # Derive per-sample seeds (deterministic across reruns).
                        sp_i["seed"] = int(sp_i["seed"]) + i
                    out = await handle.llm_client.generate(
                        f"{request_id}_{i}",
                        prompt_ids=list(prompt_tokens),
                        sampling_params=sp_i,
                    )
                    return out

                outs = await asyncio.gather(*[_one(i) for i in range(num_samples)])
            finally:
                handle._active_samplers -= 1
                if handle._active_samplers == 0:
                    handle._samplers_drained.set()
            sequences = []
            for out in outs:
                if out.stop_reason == "aborted":
                    raise BackendError(
                        "rollout request aborted", backend="verl", operation="sample",
                    )
                n_tok = len(out.token_ids)
                stop_reason = "length" if (max_tokens is not None and n_tok >= max_tokens) else "stop"
                sequence = {
                    "tokens": list(out.token_ids),
                    "logprobs": list(out.log_probs) if out.log_probs is not None else None,
                    "text": None,
                    "stop_reason": stop_reason,
                }
                if topk_sample_logprobs > 0:
                    # verl's server returns the head best first, [T, k] ids and logprobs
                    ids = out.extra_fields["response_topk_ids"].tolist()
                    lps = out.extra_fields["response_topk_log_probs"].tolist()
                    sequence["topk_sample_logprobs"] = [list(zip(i, lp)) for i, lp in zip(ids, lps)]
                sequences.append(sequence)
            return {"sequences": sequences, "prompt_logprobs": None}
        except (BackendError, UnsupportedFeatureError):
            raise
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="sample", original_error=e) from e

    async def prepare_for_generation(self, handle: VerlHandle) -> None:
        async with handle._lock:
            try:
                await self._ready_sampler(handle)
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(
                    str(e), backend="verl", operation="prepare_for_generation", original_error=e,
                ) from e

    # -------------------------------------------------------------- helpers

    def _boot(self, trainer_config, model_id: str, num_gpus: int):
        """One model's TinkerActorRolloutRefWorker group in the shared GPUs, its engines
        initialized. Blocking. A failed boot leaves no actor and no share behind."""
        import ray
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

        pool = self._gpus.acquire(model_id, num_gpus)
        try:
            ray_cls = RayClassWithInitArgs(
                cls=ray.remote(TinkerActorRolloutRefWorker),
                config=trainer_config.actor_rollout_ref, role="actor_rollout",
            )
            worker_group = RayWorkerGroup(
                resource_pool=pool, ray_cls_with_init=ray_cls, name_prefix=worker_name_prefix(model_id))
            worker_group.init_model()
        except BaseException:   # a cancelled create too: the actors and the share go, then it re-raises
            try:
                _kill_named_actors(worker_name_prefix(model_id))
            finally:
                self._gpus.release(model_id)
            raise
        return worker_group

    @staticmethod
    async def _quiesce_samplers(h: VerlHandle) -> None:
        """Drain in-flight generates. Call with h._lock HELD: new samplers
        queue behind the lock; active ones finish without needing it."""
        while h._active_samplers:
            await h._samplers_drained.wait()

    async def _ready_sampler(self, h: VerlHandle) -> None:
        """Wake `h`'s sampler at its current weights. Call with h._lock HELD.

        Raises:
            SampleRequestError: another model holds the server's one sampler
                (it does until it is deleted); nothing is built for `h`.
        """
        # no await between the check and the claim
        if self._sampler_model_id not in (None, h.model_id):
            raise SampleRequestError(
                f"the verl backend serves one sampling model per server and {self._sampler_model_id} holds "
                f"the sampler; {h.model_id} serves forward and training only", backend="verl")
        self._sampler_model_id = h.model_id
        build = asyncio.ensure_future(asyncio.to_thread(self._ensure_rollout_ready, h))
        try:
            await asyncio.shield(build)
        except BaseException:   # a cancelled sample too: re-raised below
            # cancellation does not stop the build thread: judge the sampler after it returns
            await asyncio.wait([build])
            if h.llm_manager is None:
                # a failed build can leave server actors behind (verl names them, then launches);
                # the claim goes only after they do
                await asyncio.to_thread(_kill_named_actors, VLLM_SERVER_NAME_PREFIX)
                self._sampler_model_id = None
            raise

    @staticmethod
    def _ensure_training_ready(h: VerlHandle) -> None:
        """Hybrid timeshare: sleep vLLM replicas before training ops."""
        if h.rollout_awake:
            h.ckpt_manager.sleep_replicas()
            h.rollout_awake = False

    @staticmethod
    def _ensure_rollout_ready(h: VerlHandle) -> None:
        """Wake vLLM replicas with current weights, building them at the model's first sample.

        update_weights(naive) resumes replica weights, streams the trainer's
        params (LoRA: base once, then adapter), and resumes kv_cache — correct
        both for a version-stale sync and for a plain wake after sleep
        (full-FT sleep discards weights, so wake-without-sync is unsafe).
        """
        if h.rollout_awake and h.rollout_synced_version == h.weight_version:
            return
        if h.llm_manager is None:
            from verl.checkpoint_engine import CheckpointEngineManager
            from verl.utils import omega_conf_to_dataclass
            from verl.workers.rollout.llm_server import LLMServerManager

            full = h.trainer_config
            llm_manager = LLMServerManager.create(config=full, worker_group=h.worker_group)
            ckpt_manager = CheckpointEngineManager(
                config=omega_conf_to_dataclass(full.actor_rollout_ref.rollout.checkpoint_engine),
                actor_wg=h.worker_group,
                replicas=llm_manager.get_replicas(),
            )
            # the replicas load dummy weights (load_format) and must take the trainer's before
            # they serve; verl's update_weights wants them asleep first (its docstring)
            ckpt_manager.sleep_replicas()
            # all three at once: a half-built sampler is rebuilt from nothing
            h.llm_manager, h.llm_client, h.ckpt_manager = llm_manager, llm_manager.get_client(), ckpt_manager
        h.ckpt_manager.update_weights(global_steps=h.weight_version)
        h.rollout_awake = True
        h.rollout_synced_version = h.weight_version

    def _ensure_loss_fn(self, h: VerlHandle, loss_fn: str) -> None:
        if h.loss_fn_name == loss_fn:
            return
        from .losses import get_loss_fn
        from functools import partial
        h.worker_group.set_loss_fn(partial(_loss_dispatch, loss_name=loss_fn))
        _ = get_loss_fn(loss_fn)  # validate name eagerly
        h.loss_fn_name = loss_fn

    def _to_tensordict(self, h: VerlHandle, data: List[Datum], loss_fn: str, infer: bool = False):
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        padded = self.converter.forward_backward_to_backend(data, loss_fn, h.config)

        # Divisibility: engine needs per-DP-rank count % micro_batch_size == 0.
        # Pad with zero-weight clones (pure-sum => zero grad, E0-safe)
        # instead of rejecting or trimming. The infer path (compute_log_prob)
        # partitions by infer_micro_batch_size_per_gpu, not the train one.
        mb_key = "infer_micro_batch_size_per_gpu" if infer else "micro_batch_size_per_gpu"
        micro_bs = int(h.config.get("engine", {}).get(mb_key, 1) or 1)
        multiple = h.dp_size * micro_bs
        b = padded["input_ids"].shape[0]
        remainder = b % multiple
        if remainder:
            pad_n = multiple - remainder
            for key, t in padded.items():
                clone = t[:1].repeat(pad_n, *([1] * (t.dim() - 1))).clone()
                if key in ("weights", "advantages"):
                    clone.zero_()
                padded[key] = torch.cat([t, clone], dim=0)

        global_token_num = padded["attention_mask"].sum(dim=-1).tolist()
        td = TensorDict(padded, batch_size=[padded["input_ids"].shape[0]])
        td = left_right_2_no_padding(td)
        # temperature: engine divides logits by it (logprob provenance);
        # training-side logprobs are untempered in the Tinker contract
        tu.assign_non_tensor(td, global_token_num=global_token_num, temperature=1.0)
        if "target_tokens" in padded:
            if h.config["engine"]["ulysses_sequence_parallel_size"] > 1:
                # each rank's logits then cover a slice of the sequence; the hook's targets do not follow it
                raise UnsupportedFeatureError("[N, K] target_tokens with sequence parallelism", backend="verl")
            # the engine hands the full logits to the loss function (losses.target_logprob_columns)
            # and computes no single-target logprobs of its own
            tu.assign_non_tensor(td, distillation_use_topk=True, distillation_only=True)
        return td


def _loss_dispatch(config=None, model_output=None, data=None, dp_group=None, student_logits=None,
                   *, loss_name: str):
    """Module-level picklable shim: resolves the loss by name inside the worker.
    The engine also calls it as its logits hook, with `student_logits` and `data` only."""
    from . import losses
    if student_logits is not None:
        return losses.target_logprob_columns(student_logits, data)
    return losses.get_loss_fn(loss_name)(config, model_output, data, dp_group)


def _wg_call(method, td):
    """Blocking worker-group data call: resolve DataProtoFuture if non-blocking.

    Blocking dispatches (e.g. compute_log_prob) return the TensorDict directly;
    TensorDict also has .get(key), so type-check instead of duck-typing.
    """
    from tensordict import TensorDictBase

    out = method(td)
    if not isinstance(out, TensorDictBase) and hasattr(out, "get"):
        out = out.get()
    return out


def _wg_scalar_call(method, *args):
    out = method(*args)
    if isinstance(out, list):
        # ONE_TO_ALL returns per-worker dicts; grad_norm identical post-clip
        merged: Dict[str, Any] = {}
        for d in out:
            if isinstance(d, dict):
                merged.update(d)
        return merged
    if hasattr(out, "get"):
        out = out.get()
    return out or {}


def _scalar_metrics(out) -> Dict[str, Any]:
    from verl.utils import tensordict_utils as tu
    metrics = tu.get(out, "metrics") or {}
    out_metrics = {}
    for k, v in dict(metrics).items():
        if k.startswith("perf/"):
            continue
        if hasattr(v, "data"):
            v = v.data
        if isinstance(v, (int, float)):
            # SDK metric keys are "name:reduction" (chunked_fwdbwd_helpers)
            key = k.replace(":", "_").replace("/", "_")
            out_metrics[f"{key}:sum"] = float(v)
    return out_metrics


def _to_vllm_sampling_params(sampling_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Tinker SamplingParams dict -> vLLM server generate sampling_params."""
    sp = dict(sampling_params or {})
    temperature = sp.get("temperature")
    top_p = sp.get("top_p")
    out: Dict[str, Any] = {
        "temperature": float(1.0 if temperature is None else temperature),
        "top_p": float(1.0 if top_p is None else top_p),
        "logprobs": True,
    }
    top_k = sp.get("top_k")
    # explicit -1 (disabled) when unset: never inherit engine/generation-config
    # truncation the client didn't ask for
    out["top_k"] = -1 if top_k in (None, 0, -1) else int(top_k)
    max_tokens = sp.get("max_tokens") or sp.get("max_new_tokens")
    if max_tokens:
        out["max_tokens"] = int(max_tokens)
    if sp.get("stop"):
        out["stop"] = sp["stop"]
    if sp.get("stop_token_ids"):
        out["stop_token_ids"] = [int(t) for t in sp["stop_token_ids"]]
    if sp.get("seed") is not None:
        out["seed"] = int(sp["seed"])
    return out


def _adopt_adapter_shape(checkpoint_root: str, model_cfg: Dict[str, Any]) -> None:
    """Adopt the interchange adapter's r/alpha over the request's.

    PeftModel.from_pretrained honors adapter_config.json, but verl's rollout
    LoRA plumbing reads model.lora_rank/lora_alpha — leaving them disagreeing
    changes the effective alpha/r scaling on one side only (silent rescale at
    the migration seam; specs/007 §1.3).
    """
    cfg = read_adapter_config(checkpoint_root)
    if not cfg:
        logger.warning(
            "Interchange adapter at %s has no adapter_config.json; keeping requested "
            "rank=%s alpha=%s", checkpoint_root, model_cfg.get("lora_rank"), model_cfg.get("lora_alpha"),
        )
        return
    for key, cfg_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha")):
        want = cfg.get(cfg_key)
        if want is None:
            continue
        have = model_cfg.get(key)
        if have != want:
            logger.warning(
                "Interchange adapter %s=%s overrides requested %s=%s", cfg_key, want, key, have,
            )
            model_cfg[key] = want


def _compose_actor_rollout_config(cfg: Dict[str, Any]):
    """Assemble a trainer-shaped OmegaConf for ActorRolloutRefWorker +
    LLMServerManager from verl's packaged ppo_trainer defaults."""
    import os as _os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import verl.trainer.config as _cfgpkg

    model = cfg["model"]
    engine = cfg["engine"]
    rollout = cfg["rollout"]
    micro_bs = int(engine["micro_batch_size_per_gpu"])
    num_gpus = int(cfg["num_gpus"])

    tm = model["target_modules"]
    tm_override = "[" + ",".join(tm) + "]" if isinstance(tm, (list, tuple)) else str(tm)

    overrides = [
        f"actor_rollout_ref.model.path={model['path']}",
        f"actor_rollout_ref.model.use_remove_padding={model['use_remove_padding']}",
        f"actor_rollout_ref.model.lora_rank={model['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={model['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={tm_override}",
        *(
            [f"actor_rollout_ref.model.lora_adapter_path={model['lora_adapter_path']}"]
            if model.get("lora_adapter_path") else []
        ),
        f"actor_rollout_ref.actor.strategy={engine['strategy']}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={engine['ulysses_sequence_parallel_size']}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={engine['use_dynamic_bsz']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_bs}",
        # unused by the Tinker split path (fb bypasses the mini-batch loop)
        # but ActorConfig validates divisibility
        f"actor_rollout_ref.actor.ppo_mini_batch_size={micro_bs * num_gpus}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={engine['max_token_len_per_gpu']}",
        f"actor_rollout_ref.actor.fsdp_config.fsdp_size={engine['fsdp_size']}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={engine['param_offload']}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={engine['optimizer_offload']}",
        f"actor_rollout_ref.actor.optim.lr={cfg['optim']['lr']}",
        f"actor_rollout_ref.rollout.name={rollout['name']}",
        f"actor_rollout_ref.rollout.mode={rollout['mode']}",
        f"actor_rollout_ref.rollout.prompt_length={rollout['prompt_length']}",
        f"actor_rollout_ref.rollout.response_length={rollout['response_length']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout['tensor_model_parallel_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.enforce_eager={rollout['enforce_eager']}",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        # Tinker logprob provenance: sampler must report RAW model logprobs
        # (training engine computes untempered); verl's default
        # processed_logprobs returns post-processing renormalized values
        # (measured ~1.3 nats off HF fp32 on Qwen2.5-0.5B, 2026-08-02)
        "actor_rollout_ref.rollout.logprobs_mode=raw_logprobs",
        f"actor_rollout_ref.rollout.log_prob_use_dynamic_bsz={engine['use_dynamic_bsz']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={engine['infer_micro_batch_size_per_gpu']}",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={engine['infer_max_token_len_per_gpu']}",
        f"+actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs={VLLM_MAX_LOGPROBS}",
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_os.path.dirname(_cfgpkg.__file__), version_base=None):
        full = compose(config_name="ppo_trainer", overrides=overrides)
    return full


# verl names its vLLM server actors `vllm_server_<replica>_<node>`; one model owns them all (VerlBackend._sampler_model_id)
VLLM_SERVER_NAME_PREFIX = "vllm_server_"
WORKER_NAME_PREFIX = "verl_"
SHARED_POOL_NAME_PREFIX = "verl_shared_"   # the placement group every model's workers sit in


def worker_name_prefix(model_id: str) -> str:
    """The name prefix of a model's worker actors."""
    return f"{WORKER_NAME_PREFIX}{model_id}_"


def _dp_size(num_gpus: int, sp_size: int) -> int:
    """Data-parallel ranks: the GPUs divided among sequence-parallel groups.

    Raises:
        BackendError: `sp_size` does not divide `num_gpus`.
    """
    if sp_size < 1 or num_gpus % sp_size:
        raise BackendError(f"ulysses_sequence_parallel_size {sp_size} does not divide {num_gpus} GPUs",
                           backend="verl", operation="create_model")
    return num_gpus // sp_size


def _kill_named_actors(prefix: str) -> None:
    """Kill every live named actor whose name starts with `prefix`. By name, not by
    handle: a failed build raises before it returns the handles of what it started."""
    import ray

    for name in ray.util.list_named_actors():
        if name.startswith(prefix):
            ray.kill(ray.get_actor(name))
