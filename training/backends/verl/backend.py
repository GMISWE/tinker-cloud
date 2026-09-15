"""
veRL backend — binds veRL's upstream Tinker split-training primitives
(verl.workers.engine_workers_tinker) behind the TrainingBackend ABC.

Execution model: EAGER (Miles-shaped, NOT R9-buffered). veRL upstream ships
TinkerTrainingWorker with decoupled forward_backward (grad-accumulating,
returns per-token logprobs) + optimizer_step (accepts lr/betas/eps/
weight_decay) + optimizer_zero_grad. forward_backward here does real GPU
work and returns real logprobs; apply_optimizer_step applies AdamParams and
steps once.

M2 adds rollout: TinkerActorRolloutRefWorker (actor_rollout role) boots
colocated vLLM server replicas (LLMServerManager) in hybrid timeshare —
replicas sleep during training ops and wake with a naive weight sync
(CheckpointEngineManager) for sampling. update_inference_weights is lazy:
versions are tracked on the handle and the sync happens on the next
sample/prepare_for_generation, so pure-training runs never pay for it.
Narrative: specs/006-verl-backend/design.md.
"""
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from ..base import BackendError, BackendHandle, TrainingBackend, UnsupportedFeatureError
from ...checkpoints.interchange import (
    HF_ADAPTER_DIRNAME,
    find_hf_adapter,
    read_adapter_config,
)

logger = logging.getLogger(__name__)


def _set_event() -> asyncio.Event:
    e = asyncio.Event()
    e.set()
    return e


@dataclass
class VerlHandle(BackendHandle):
    """veRL-specific runtime state."""

    worker_group: Any = None          # RayWorkerGroup over Tinker*Worker
    resource_pool: Any = None         # RayResourcePool
    config: Dict = field(default_factory=dict)
    hf_path: str = ""
    dp_size: int = 1
    loss_fn_name: str = ""            # currently installed loss fn on workers
    created_at: str = ""
    # M2 rollout state (hybrid timeshare)
    has_rollout: bool = False
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


class VerlBackend(TrainingBackend):
    SUPPORTED_LOSS_FNS = frozenset({"cross_entropy", "importance_sampling", "ppo"})  # losses.LOSS_FNS
    """veRL backend using upstream TinkerTrainingWorker split primitives."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None
        self._builder = None

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
    ) -> VerlHandle:
        if staleness_k > 0:
            # verl's sync is already lazy but version-gated to staleness 0 at
            # sample time; the declaration is accepted but unexploited.
            logger.info(
                "[%s] staleness_k=%d declared; verl serves staleness 0 "
                "(lazy sync, version-gated) — declaration unexploited",
                request_id, staleness_k,
            )
        if objective != "language_modeling":
            raise BackendError(
                f"verl is a language-modeling backend; objective {objective!r} unsupported",
                backend="verl", operation="create_model",
            )
        try:
            cfg = self.builder.build_args(
                base_model=base_model, num_gpus=num_gpus, lora_config=lora_config,
                parallelism=parallelism, rl_config=rl_config,
                rollout_config=rollout_config, max_seq_len=max_seq_len,
            )

            # A checkpoint carrying an interchange adapter (written by another
            # backend, or by us) is attached at model build time — verl's PEFT
            # path wraps the module BEFORE FSDP, so it cannot be a post-boot
            # load_checkpoint. Native verl shards still resume that way.
            adapter_dir = None
            if resume_from:
                ckpt_root = str(resume_from)
                adapter_dir = find_hf_adapter(ckpt_root)
                if adapter_dir:
                    cfg["model"]["lora_adapter_path"] = adapter_dir
                    _adopt_adapter_shape(ckpt_root, cfg["model"])

            enable_rollout = not debug_train_only
            boot = await asyncio.to_thread(
                _boot_worker_group, cfg, model_id, enable_rollout,
            )
            handle = VerlHandle(
                model_id=model_id,
                backend_type="verl",
                # vLLM max_model_len == rollout prompt_length + response_length
                context_length=int(cfg["rollout"]["prompt_length"]) + int(cfg["rollout"]["response_length"]),
                worker_group=boot["worker_group"],
                resource_pool=boot["resource_pool"],
                config=cfg,
                hf_path=cfg["hf_path"],
                dp_size=boot["dp_size"],
                created_at=datetime.now().isoformat(),
                has_rollout=enable_rollout,
                llm_manager=boot.get("llm_manager"),
                llm_client=boot.get("llm_client"),
                ckpt_manager=boot.get("ckpt_manager"),
                # boot does one naive sync so vLLM never serves dummy
                # weights, then sleeps the replicas for training
                rollout_synced_version=0 if enable_rollout else -1,
                lora_rank=int(cfg["model"]["lora_rank"] or 0),
            )
            if resume_from and adapter_dir is None:
                await self.load_checkpoint(handle, resume_from)
            logger.info(
                "[%s] verl model %s created (dp=%d, rollout=%s, adapter=%s)",
                request_id, model_id, handle.dp_size, enable_rollout, adapter_dir,
            )
            return handle
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="create_model", original_error=e) from e

    # ------------------------------------------------------------- train ops

    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await self._quiesce_samplers(h)
                await asyncio.to_thread(self._ensure_training_ready, h)
                await asyncio.to_thread(self._ensure_loss_fn, h, loss_fn)
                td = await asyncio.to_thread(self._to_tensordict, h, data, loss_fn)
                out = await asyncio.to_thread(_wg_call, h.worker_group.forward_backward, td)
                logprobs = await asyncio.to_thread(self._model_output_logprobs, out, td)
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
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await self._quiesce_samplers(h)
                await asyncio.to_thread(self._ensure_training_ready, h)
                params: Dict[str, Any] = {}
                if learning_rate is not None:
                    params["lr"] = float(learning_rate)
                ap = adam_params or {}
                if ap.get("beta1") is not None or ap.get("beta2") is not None:
                    params["betas"] = (float(ap.get("beta1", 0.9)), float(ap.get("beta2", 0.95)))
                if ap.get("eps") is not None:
                    params["eps"] = float(ap["eps"])
                if ap.get("weight_decay") is not None:
                    params["weight_decay"] = float(ap["weight_decay"])
                unsupported = {k: v for k, v in ap.items()
                               if k not in ("beta1", "beta2", "eps", "weight_decay", "grad_clip_norm") and v is not None}
                if unsupported:
                    logger.warning("verl optim_step ignoring AdamParams %s", sorted(unsupported))
                if ap.get("grad_clip_norm") is not None:
                    logger.warning("verl optim_step: grad_clip_norm set at engine build, not per step")

                metrics = await asyncio.to_thread(
                    _wg_scalar_call, h.worker_group.optimizer_step, params or None,
                )
                h.weight_version += 1
                grad_norm = metrics.get("grad_norm")
                return {"success": True, "grad_norm": grad_norm, "metrics": metrics}
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="apply_optimizer_step", original_error=e) from e

    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        logprobs = await self.get_logprobs(handle, data)
        return {"loss_fn_output_type": loss_fn, "loss_fn_outputs": logprobs, "metrics": {}}

    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await self._quiesce_samplers(h)
                await asyncio.to_thread(self._ensure_training_ready, h)
                td = await asyncio.to_thread(self._to_tensordict, h, data, "cross_entropy", True)
                from verl.utils import tensordict_utils as tu
                tu.assign_non_tensor(td, compute_loss=False)
                # rollout worker exposes the actor's infer path as compute_log_prob
                infer = getattr(h.worker_group, "infer_batch", None) or h.worker_group.compute_log_prob
                out = await asyncio.to_thread(_wg_call, infer, td)
                logprobs = await asyncio.to_thread(self._model_output_logprobs, out, td)
                return self.converter.extract_logprobs(logprobs, data)
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="get_logprobs", original_error=e) from e

    # ------------------------------------------------------------ lifecycle

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        # Lazy: weight_version already advanced at optim_step; the actual
        # vLLM sync happens on the next sample/prepare_for_generation
        # (version-gated), so pure-training runs never pay wake/sync cost.
        h: VerlHandle = handle  # type: ignore[assignment]
        if h.has_rollout:
            logger.debug(
                "verl update_inference_weights deferred (v%d, synced v%d)",
                h.weight_version, h.rollout_synced_version,
            )
        return None

    async def save_checkpoint(
        self,
        handle: BackendHandle,
        root: Path,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        if not persist:
            return  # ephemeral sampler save: the engines already hold the weights
        async with h._lock:
            try:
                await self._quiesce_samplers(h)
                local_path = str(root)
                await asyncio.to_thread(
                    h.worker_group.save_checkpoint, local_path, None, step or h.weight_version,
                )
                if h.lora_rank > 0:
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

    async def load_checkpoint(self, handle: BackendHandle, root: Path,
                              optimizer: bool = False) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        if optimizer:
            raise BackendError("verl restores weights only; optimizer state is not restored",
                               backend="verl", operation="load_checkpoint")
        async with h._lock:
            try:
                await self._quiesce_samplers(h)
                local_path = str(root)
                if find_hf_adapter(local_path):
                    # PeftModel.from_pretrained wraps pre-FSDP: only create_model
                    # can attach an interchange adapter.
                    raise BackendError(
                        "verl imports an interchange adapter at create_model "
                        "(lora_adapter_path), not via load_checkpoint",
                        backend="verl", operation="load_checkpoint",
                    )
                await asyncio.to_thread(h.worker_group.load_checkpoint, local_path)
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="load_checkpoint", original_error=e) from e

    async def delete_model(self, handle: BackendHandle) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        try:
            import ray
            # rollout server actors first (they hold engine refs into workers)
            for replica in (h.llm_manager.get_replicas() if h.llm_manager else []):
                for server in getattr(replica, "servers", []) or []:
                    try:
                        ray.kill(server)
                    except Exception:
                        pass
            h.llm_manager = None
            h.llm_client = None
            h.ckpt_manager = None
            for w in getattr(h.worker_group, "workers", []) or []:
                try:
                    ray.kill(w)
                except Exception:
                    pass
            if h.resource_pool is not None:
                pgs = getattr(h.resource_pool, "pgs", None) or []
                for pg in pgs:
                    try:
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
            h.worker_group = None
            h.resource_pool = None
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="delete_model", original_error=e) from e

    async def sample(
        self,
        handle: BackendHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        if not h.has_rollout:
            raise UnsupportedFeatureError(
                "sample", backend="verl",
                suggestion="model was created debug_train_only; recreate with rollout enabled",
            )
        # Hybrid timeshare: generation must not overlap training ops on the
        # same GPUs. Register as an active sampler under the FIFO lock (which
        # serializes the wake/weight-sync), then generate WITHOUT the lock so
        # concurrent sample requests batch inside vLLM; training ops quiesce
        # samplers via _quiesce_samplers before touching the GPUs.
        async with h._lock:
            try:
                await asyncio.to_thread(self._ensure_rollout_ready, h)

                if pinned_version is not None and pinned_version != h.weight_version:
                    # vLLM serves the live (base+adapter) route; a pinned
                    # nonzero version is not honorable — BUG-015 aliasing
                    # class, logged loudly rather than silently aliased.
                    logger.warning(
                        "[%s] pinned_version=%s not honored for %s (verl serves "
                        "the live engine, v%d) — BUG-015 aliasing risk",
                        request_id, pinned_version, h.model_id, h.weight_version,
                    )
                if prompt_logprobs:
                    logger.warning(
                        "[%s] prompt_logprobs unsupported on verl sample path "
                        "(use forward/get_logprobs); returning None", request_id,
                    )
                h._active_samplers += 1
                h._samplers_drained.clear()
            except (BackendError, UnsupportedFeatureError):
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="sample", original_error=e) from e
        try:
            try:
                vllm_sp = _to_vllm_sampling_params(sampling_params)
                max_tokens = vllm_sp.get("max_tokens")

                async def _one(i: int):
                    sp_i = dict(vllm_sp)
                    if sp_i.get("seed") is not None:
                        # vLLM seeds per request; a fixed seed across the
                        # num_samples fanout would return identical samples.
                        # Derive per-sample seeds (deterministic across reruns).
                        sp_i["seed"] = int(sp_i["seed"]) + i
                    out = await h.llm_client.generate(
                        f"{request_id}_{i}",
                        prompt_ids=list(prompt_tokens),
                        sampling_params=sp_i,
                    )
                    return out

                outs = await asyncio.gather(*[_one(i) for i in range(num_samples)])
            finally:
                h._active_samplers -= 1
                if h._active_samplers == 0:
                    h._samplers_drained.set()
            sequences = []
            for out in outs:
                if out.stop_reason == "aborted":
                    raise BackendError(
                        "rollout request aborted", backend="verl", operation="sample",
                    )
                n_tok = len(out.token_ids)
                stop_reason = "length" if (max_tokens is not None and n_tok >= max_tokens) else "stop"
                sequences.append({
                    "tokens": list(out.token_ids),
                    "logprobs": list(out.log_probs) if out.log_probs is not None else None,
                    "text": None,
                    "stop_reason": stop_reason,
                })
            return {"sequences": sequences, "prompt_logprobs": None}
        except (BackendError, UnsupportedFeatureError):
            raise
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="sample", original_error=e) from e

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        if not h.has_rollout:
            raise UnsupportedFeatureError(
                "prepare_for_generation", backend="verl",
                suggestion="model was created debug_train_only; recreate with rollout enabled",
            )
        async with h._lock:
            try:
                await asyncio.to_thread(self._ensure_rollout_ready, h)
            except Exception as e:
                raise BackendError(
                    str(e), backend="verl", operation="prepare_for_generation", original_error=e,
                ) from e

    # -------------------------------------------------------------- helpers

    @staticmethod
    async def _quiesce_samplers(h: VerlHandle) -> None:
        """Drain in-flight generates. Call with h._lock HELD: new samplers
        queue behind the lock; active ones finish without needing it."""
        while h._active_samplers:
            await h._samplers_drained.wait()

    @staticmethod
    def _ensure_training_ready(h: VerlHandle) -> None:
        """Hybrid timeshare: sleep vLLM replicas before training ops."""
        if h.has_rollout and h.rollout_awake:
            h.ckpt_manager.sleep_replicas()
            h.rollout_awake = False

    @staticmethod
    def _ensure_rollout_ready(h: VerlHandle) -> None:
        """Wake vLLM replicas with current weights.

        update_weights(naive) resumes replica weights, streams the trainer's
        params (LoRA: base once, then adapter), and resumes kv_cache — correct
        both for a version-stale sync and for a plain wake after sleep
        (full-FT sleep discards weights, so wake-without-sync is unsafe).
        """
        if h.rollout_awake and h.rollout_synced_version == h.weight_version:
            return
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

    def _to_tensordict(self, h: VerlHandle, data: List[Dict], loss_fn: str, infer: bool = False):
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
        return td

    @staticmethod
    def _model_output_logprobs(out, td) -> torch.Tensor:
        """Gathered worker output -> padded [B, Rmax] logprobs."""
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import no_padding_2_padding

        lp = tu.get(out, "log_probs")
        if lp is None:
            raise BackendError("worker returned no log_probs", backend="verl", operation="forward_backward")
        return no_padding_2_padding(lp, td)


def _loss_dispatch(config=None, model_output=None, data=None, dp_group=None, loss_name: str = "cross_entropy"):
    """Module-level picklable shim: resolves the loss by name inside the worker."""
    from .losses import get_loss_fn
    return get_loss_fn(loss_name)(config, model_output, data, dp_group)


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
    try:
        metrics = tu.get(out, "metrics") or {}
    except Exception:
        metrics = {}
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
    out: Dict[str, Any] = {
        "temperature": float(sp.get("temperature") if sp.get("temperature") is not None else 1.0),
        "top_p": float(sp.get("top_p") if sp.get("top_p") is not None else 1.0),
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
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_os.path.dirname(_cfgpkg.__file__), version_base=None):
        full = compose(config_name="ppo_trainer", overrides=overrides)
    return full


def _boot_worker_group(cfg: Dict[str, Any], model_id: str, enable_rollout: bool = False):
    """Boot the verl worker group.

    enable_rollout=False: TinkerTrainingWorker (M1 training-only path).
    enable_rollout=True: TinkerActorRolloutRefWorker + colocated vLLM server
    replicas; one naive weight sync at boot (load_format=dummy must never be
    served), then replicas sleep until the first sample.
    """
    import ray
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    if enable_rollout:
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.utils import omega_conf_to_dataclass
        from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
        from verl.workers.rollout.llm_server import LLMServerManager

        full = _compose_actor_rollout_config(cfg)
        ray_cls = RayClassWithInitArgs(
            cls=ray.remote(TinkerActorRolloutRefWorker),
            config=full.actor_rollout_ref, role="actor_rollout",
        )
        resource_pool = RayResourcePool(
            process_on_nodes=[cfg["num_gpus"]], name_prefix=f"verl_{model_id}",
        )
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
        try:
            wg.init_model()

            # KNOWN LIMIT: vLLM server actors are Ray-named per replica rank
            # (manager doesn't thread name_suffix), so at most ONE rollout-enabled
            # verl model per Ray cluster; a second create_model fails loudly on
            # the actor name. delete_model frees the names.
            llm_manager = LLMServerManager.create(config=full, worker_group=wg)
            ckpt_manager = CheckpointEngineManager(
                config=omega_conf_to_dataclass(full.actor_rollout_ref.rollout.checkpoint_engine),
                actor_wg=wg,
                replicas=llm_manager.get_replicas(),
            )
            # canonical boot order (agent-loop harness): sleep -> sync -> sleep;
            # vLLM never serves dummy weights, training starts with GPUs free
            ckpt_manager.sleep_replicas()
            ckpt_manager.update_weights(global_steps=0)
            ckpt_manager.sleep_replicas()
        except Exception:
            # boot failure must not leak live actors/PG: the next create_model
            # would hang forever on placement-group resources (runbook footgun 2)
            for w in getattr(wg, "workers", []) or []:
                try:
                    ray.kill(w)
                except Exception:
                    pass
            for pg in getattr(resource_pool, "pgs", None) or []:
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
            raise

        sp = cfg["engine"]["ulysses_sequence_parallel_size"]
        dp_size = max(cfg["num_gpus"] // max(sp, 1), 1)
        return {
            "worker_group": wg,
            "resource_pool": resource_pool,
            "dp_size": dp_size,
            "llm_manager": llm_manager,
            "llm_client": llm_manager.get_client(),
            "ckpt_manager": ckpt_manager,
        }

    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine_workers import TrainingWorkerConfig
    from verl.workers.engine_workers_tinker import TinkerTrainingWorker

    model = cfg["model"]
    engine = cfg["engine"]
    model_config = HFModelConfig(
        path=model["path"],
        use_remove_padding=model["use_remove_padding"],
        lora_rank=model["lora_rank"],
        lora_alpha=model["lora_alpha"],
        target_modules=model["target_modules"],
        lora_adapter_path=model.get("lora_adapter_path"),
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy=engine["strategy"],
        fsdp_size=engine["fsdp_size"],
        ulysses_sequence_parallel_size=engine["ulysses_sequence_parallel_size"],
        use_remove_padding=engine["use_remove_padding"],
        use_dynamic_bsz=engine["use_dynamic_bsz"],
        micro_batch_size_per_gpu=engine["micro_batch_size_per_gpu"],
        max_token_len_per_gpu=engine["max_token_len_per_gpu"],
        infer_micro_batch_size_per_gpu=engine["infer_micro_batch_size_per_gpu"],
        infer_max_token_len_per_gpu=engine["infer_max_token_len_per_gpu"],
        param_offload=engine["param_offload"],
        optimizer_offload=engine["optimizer_offload"],
        grad_offload=engine["grad_offload"],
    )
    optimizer_config = FSDPOptimizerConfig(
        lr=cfg["optim"]["lr"], lr_scheduler_type=cfg["optim"]["lr_scheduler_type"],
    )
    worker_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )

    ray_cls = RayClassWithInitArgs(cls=ray.remote(TinkerTrainingWorker), config=worker_config)
    resource_pool = RayResourcePool(process_on_nodes=[cfg["num_gpus"]], name_prefix=f"verl_{model_id}")
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
    wg.reset()

    sp = engine["ulysses_sequence_parallel_size"]
    dp_size = max(cfg["num_gpus"] // max(sp, 1), 1)
    return {"worker_group": wg, "resource_pool": resource_pool, "dp_size": dp_size}
