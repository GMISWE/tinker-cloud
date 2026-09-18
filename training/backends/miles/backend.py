"""Miles backend — wraps TinkerTrainGroup/RolloutManager/MilesArgumentBuilder
behind the TrainingBackend interface.

Targets the miles `tinker-seam` branch (upstream-based): async
TinkerTrainGroup fanout, decoupled
forward_backward_only / apply_optimizer_step, pure-sum loss via rollout keys
set in the converter."""
import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

from ..base import BackendError, BackendHandle, TrainingBackend, UnsupportedFeatureError
from ...models.requests import Datum
from .config import NO_CLIP_EPS_HIGH, MilesConfig
from ...core import routing
from ...core.loss_registry import clip_thresholds
from ...checkpoints.interchange import export_hf_adapter
from .model_setup import record_native_checkpoint, resolve_native_checkpoint
from .sglang_client import SGLangClient, SGLangClientPool

logger = logging.getLogger(__name__)

def _dp_size(args: Any) -> int:
    """Data-parallel width the actors will split a batch across."""
    return int(args.data_parallel_size)


def _router_url(router_ip: Optional[str], router_port: Optional[int]) -> Optional[str]:
    """SGLang router base URL, or None when the model booted without one (SFT)."""
    if not router_ip or not router_port:
        return None
    return f"http://{router_ip}:{router_port}"


def _adapter_save_dir(native_root: Optional[Path], adapter_name: str) -> Path:
    """Per-tenant dir miles writes adapter checkpoints into, under the model's
    own native area. Without it (config.save=None) miles skips per-adapter
    checkpoints entirely.

    Path, not str: TinkerAdapterConfig.save is annotated `str | Path | None`
    but miles only ever does `config.save / "checkpoints"` — a str explodes
    at adapter registration (declared type wider than the honored one).
    """
    if native_root is None:
        raise BackendError("create_model needs native_root for the adapter save dir",
                           backend="miles", operation="create_model")
    return Path(native_root) / "adapters" / adapter_name


def _publish_adapter(adapter_save_dir: Path, root: str, adapter_name: Optional[str]) -> None:
    """Copy the newest per-adapter HF PEFT pair miles wrote into the
    cross-backend interchange dir under the checkpoint root (specs/007 §2.1)."""
    ckpt_root = os.path.join(adapter_save_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        logger.warning("Adapter %s: no checkpoints under %s to publish", adapter_name, ckpt_root)
        return
    steps = [d for d in os.listdir(ckpt_root) if d.startswith("step_") and d[5:].isdigit()]
    if not steps:
        logger.warning("Adapter %s: no step_* checkpoint in %s", adapter_name, ckpt_root)
        return
    latest = max(steps, key=lambda d: int(d[5:]))
    export_hf_adapter(os.path.join(ckpt_root, latest), root)


def _publish_native_adapter(args, root: str) -> None:
    """Single-tenant publish: miles' save_model writes the PEFT pair only into
    its native `<args.save>/iter_*/adapter` dir (as `adapter_model.bin` — the
    fused-QKV lora_A aliasing makes safetensors refuse), and nothing exported
    the interchange copy since the publish hook moved onto the pool-mode
    `adapter_save_dir`. Pick the adapter dir the save just wrote (newest
    mtime; the numeric iter suffix is miles' own counter, monotone per save)
    and export it. Runs under the handle's lock right after save_model, so
    newest-mtime is the one this save produced."""
    save_root = args.save
    if not save_root or not os.path.isdir(save_root):
        logger.warning("No miles save root at %r; skipping interchange publish", save_root)
        return
    candidates = [
        os.path.join(save_root, d, "adapter")
        for d in os.listdir(save_root)
        if d.startswith("iter_")
    ]
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        logger.warning("No iter_*/adapter under %s; nothing to publish", save_root)
        return
    latest = max(candidates, key=os.path.getmtime)
    export_hf_adapter(latest, root)
    # the iter_* dir this adapter came from is what a resume must hand Megatron
    record_native_checkpoint(root, os.path.dirname(latest))


def _model_input_lens(data: List[Datum]) -> List[int]:
    """Per-datum model_input token lengths (the observation-contract unit:
    fb logprobs are datum-aligned to these, NOT to rollout tokens which
    append the final target)."""
    from .rollout_data import TinkerDataConverter

    return [len(TinkerDataConverter.extract_tokens_from_model_input(d.model_input)) for d in data]


def _engine_context_length(hf_path: str) -> Optional[int]:
    """SGLang is booted without --context-length, so its window is the model's
    max_position_embeddings; None if the config cannot be read."""
    try:
        from ...utils.model_config import load_model_config
        return int(load_model_config(hf_path)["max_position_embeddings"])
    except Exception as e:  # noqa: BLE001 - unknown context disables the check, never blocks boot
        logger.warning("context_length unknown for %s: %s", hf_path, e)
        return None


@dataclass
class MilesHandle(BackendHandle):
    """Miles-specific runtime state."""

    train_group: Any = None           # RayTrainGroup
    rollout_manager: Any = None       # RolloutManager (None for SFT)
    placement_group: Any = None       # Ray PlacementGroup
    args: Any = None                  # Megatron Namespace
    router_ip: Optional[str] = None
    router_port: Optional[int] = None
    rlve_config: Optional[Dict[str, Any]] = None
    wandb_config: Optional[Dict[str, Any]] = None
    created_at: str = ""
    training_run_id: str = ""
    # Multi-LoRA pool mode (TINKERCLOUD_MILES_MULTILORA_SLOTS): this model is
    # an adapter slot on shared rails rather than a dedicated full model.
    controller: Any = None            # MultiLoRAController (named Ray actor)
    adapter_name: Optional[str] = None
    adapter_slot: Optional[int] = None
    seed_warned: bool = False         # the ignored-sampling-seed warning fired once
    # Where miles writes this adapter's per-step checkpoints (Megatron shard +
    # HF PEFT pair); source of the cross-backend interchange export.
    adapter_save_dir: Optional[Path] = None
    # Weight version = successful optim steps applied. Sampler registration
    # snapshots it (routers/checkpoints.py) so a sampler saved before any
    # step pins v0. v0 == base model only for fresh-init LoRA (B=0 at init),
    # hence created_from_checkpoint gates the v0->base sampling route.
    created_from_checkpoint: bool = False
    # Serializes GPU-bound ops per model. The task manager runs request
    # handlers as concurrent asyncio tasks; without this, pipelined
    # fb/optim_step broadcasts interleave inconsistently across the DP actors
    # (mispaired collectives -> scrambled outputs; optim_step consuming a
    # later fb's grads). asyncio.Lock wakes waiters FIFO, so execution
    # follows submission order. Pool-mode handles share the pool's lock:
    # every broadcast rides the same actor group, so cross-tenant ops must
    # serialize too (M2: N tenants, serialized train calls).
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _PoolOp:
    """One queued pool operation. kind 'fb' ops are mergeable; 'other' ops
    run their closure; 'stop' ends the dispatcher."""

    kind: str                                   # "fb" | "other" | "stop"
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    run: Any = None                             # "other": async closure
    tenant: Optional[str] = None                # model_id; None = barrier op
    # fb fields:
    handle: Any = None
    rollout_data: Any = None
    loss_fn: Optional[str] = None
    num_samples: int = 0
    input_lens: Optional[List[int]] = None      # per-datum model_input lengths


@dataclass
class MilesPool:
    """Shared multi-LoRA rails (M2/M3): one boot serves N tenant adapters.

    First create_model boots the pool; later creates register into it;
    delete deregisters, and the last tenant out tears the pool down.

    All GPU-bound ops flow through `queue`, consumed by ONE dispatcher task
    in strict FIFO — the pool's serializer (was an asyncio.Lock in M2; a
    queue makes the M3 co-batch merge window explicit). Consecutive queued
    fb ops with the same loss_fn merge into one mixed-slot train call, up
    to `cobatch_max_samples` (0 = merging off; batch of 1 == M2 behavior)."""

    train_group: Any
    rollout_manager: Any
    placement_group: Any            # create_placement_groups() dict
    controller: Any                 # MultiLoRAController (named Ray actor)
    args: Any                       # pool-boot Megatron Namespace (governs all tenants)
    hf_path: str
    base_model: str
    router_ip: Optional[str] = None
    router_port: Optional[int] = None
    tenants: Dict[str, int] = field(default_factory=dict)   # model_id -> slot
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: "asyncio.Queue[_PoolOp]" = field(default_factory=asyncio.Queue)
    dispatcher: Optional[asyncio.Task] = None
    cobatch_max_samples: int = 0
    # Reorder drain: the merge window may defer OTHER tenants' non-fb ops
    # (cross-tenant order is not contractual; per-tenant FIFO is preserved
    # via the blocked-tenant rule). Off by default.
    cobatch_reorder: bool = False
    # E0 admission guard: refuse a merge that would move the call's per-rank
    # token total across this boundary. 0 disables. See
    # converter.cobatch_preserves_token_bucket.
    cobatch_e0_tokens: int = 512


def _boot_clip_range(args) -> tuple:
    return 1.0 - args.eps_clip, 1.0 + args.eps_clip_high


def _check_miles_clip_config(h: "MilesHandle", loss_fn: str, loss_fn_config: Optional[Dict[str, float]]) -> None:
    """eps_clip is read once at actor boot and both RL losses share the clamp:
    importance_sampling needs a range that cannot bind, ppo exactly the call's range."""
    if loss_fn not in ("importance_sampling", "ppo"):
        return
    boot_low, boot_high = _boot_clip_range(h.args)
    if loss_fn == "importance_sampling":
        if boot_low <= 0.0 and boot_high >= 1.0 + NO_CLIP_EPS_HIGH:
            return
        raise UnsupportedFeatureError(
            "importance_sampling on a clipped actor group", backend="miles",
            suggestion=f"booted with clip ({boot_low:g}, {boot_high:g}); unset SLIME_EPS_CLIP / "
                       "SLIME_EPS_CLIP_HIGH before create_model (default boot is unclipped)",
        )
    low, high = clip_thresholds(loss_fn_config)
    if abs(low - boot_low) > 1e-9 or abs(high - boot_high) > 1e-9:
        raise UnsupportedFeatureError(
            f"ppo clip thresholds ({low}, {high})", backend="miles",
            suggestion=f"this actor group was booted with ({boot_low:g}, {boot_high:g}); "
                       f"set SLIME_EPS_CLIP={1 - low:g} SLIME_EPS_CLIP_HIGH={high - 1:g} before create_model",
        )


class MilesBackend(TrainingBackend[MilesHandle]):
    # sft_loss / policy_loss; PPO clip range is a boot-time Megatron arg (see forward_backward).
    SUPPORTED_LOSS_FNS = frozenset({"cross_entropy", "importance_sampling", "ppo"})
    """Thin adapter over existing Miles integration code (model_service.py / training_service.py)."""
    config: MilesConfig

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self.config = MilesConfig.from_env(self.overrides)
        # One persistent connection pool per SGLang router this process samples from.
        self._sglang_pool = SGLangClientPool(max_connections=self.config.sglang_max_connections)
        # Lazy-import converter to avoid import errors when Miles is not installed
        self._converter = None
        self._builder = None
        # Multi-LoRA pool (M2). _pool_admin serializes boot/join/teardown;
        # lock order is always admin -> pool.lock.
        self._pool: Optional[MilesPool] = None
        self._pool_admin = asyncio.Lock()

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
            self._builder = MilesArgumentBuilder(config=self.config)
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
        objective: str = "language_modeling",
        staleness_k: int = 0,
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> MilesHandle:
        if staleness_k > 0:
            # A staleness declaration is a permission (served staleness <= k), so
            # eager refit-every-step satisfies it trivially; carried, unexploited.
            logger.info(
                "[%s] staleness_k=%d declared; miles refits eagerly — "
                "declaration accepted but unexploited (served staleness always 0)",
                request_id, staleness_k,
            )
        boot_kwargs: Dict[str, Any] = dict(
            model_id=model_id, request_id=request_id, base_model=base_model,
            num_gpus=num_gpus, lora_config=lora_config, parallelism=parallelism,
            rl_config=rl_config, rollout_config=rollout_config,
            debug_train_only=debug_train_only, resume_from=resume_from,
            max_batch_size=max_batch_size, max_seq_len=max_seq_len,
            rlve_config=rlve_config, wandb_config=wandb_config,
            objective=objective, num_labels=num_labels, head_config=head_config,
            native_root=native_root,
        )
        # Mirror the builder's pool gate (configured slots + LoRA rank).
        slots = self.config.multilora_slots
        if slots <= 0 or lora_config is None or lora_config.get("rank", 0) <= 0:
            return await self._boot_model(**boot_kwargs)
        async with self._pool_admin:
            if self._pool is not None:
                return await self._join_pool(
                    pool=self._pool, model_id=model_id, request_id=request_id,
                    base_model=base_model, lora_config=lora_config,
                    debug_train_only=debug_train_only,
                    resume_from=resume_from, rlve_config=rlve_config,
                    native_root=native_root,
                    objective=objective,
                )
            return await self._boot_model(**boot_kwargs)

    async def _join_pool(
        self,
        pool: MilesPool,
        model_id: str,
        request_id: str,
        base_model: str,
        lora_config: Dict[str, Any],
        debug_train_only: bool,
        resume_from: Optional[Path],
        rlve_config: Optional[Dict[str, Any]],
        objective: str,
        native_root: Optional[Path] = None,
    ) -> MilesHandle:
        """Register a new tenant adapter into the live pool (caller holds
        _pool_admin). The pool's boot args govern parallelism/batch shape;
        only the tenant's LoRA rank/alpha are per-adapter."""
        if objective != "language_modeling":
            raise BackendError(
                f"Miles is a language-modeling backend; objective {objective!r} "
                f"requires a classification backend (automodel / megatron_bridge)",
                backend="miles", operation="create_model",
            )
        if base_model != pool.base_model:
            raise BackendError(
                f"Multi-LoRA pool serves base model {pool.base_model!r}; "
                f"cannot create {base_model!r} on it (one base per pool)",
                backend="miles", operation="create_model",
            )
        if debug_train_only or rlve_config:
            raise BackendError(
                "Multi-LoRA pool mode supports neither debug_train_only nor RLVE",
                backend="miles", operation="create_model",
            )
        if resume_from:
            raise BackendError(
                "Resuming from a checkpoint into a multi-LoRA pool is not "
                "supported yet (adapter-scoped resume unimplemented)",
                backend="miles", operation="create_model",
            )

        from miles.utils.adapter_config import TinkerAdapterConfig

        # Same rank/alpha derivation as the builder (MilesArgumentBuilder LoRA args).
        rank = int(lora_config.get("rank", 0))
        alpha = int(lora_config.get("alpha") or rank)
        adapter_name = re.sub(r"[^A-Za-z0-9._-]", "-", model_id)
        adapter_save_dir = _adapter_save_dir(native_root, adapter_name)
        try:
            registration = await pool.controller.register_adapter.remote(
                adapter_name,
                TinkerAdapterConfig(rank=rank, alpha=alpha, save=adapter_save_dir),
            )
        except Exception as e:
            # Registry errors (slots full, name colliding/cleaning-up,
            # rank > allocated max) surface here.
            raise BackendError(
                str(e), backend="miles", operation="create_model", original_error=e,
            ) from e
        adapter_slot = registration["slot"]

        async def _load_and_push() -> None:
            # Actors load the PENDING adapter into its slot and mark it
            # for push; update_weights upserts exactly the pending set
            # (LoRA B=0 => zero-delta) and promotes it to ACTIVE.
            await pool.train_group.reconcile_adapters()
            await pool.train_group.update_weights()

        async def _rollback() -> None:
            await pool.controller.deregister_adapter.remote(adapter_name)
            await pool.train_group.reconcile_adapters()

        try:
            await self._pool_run(pool, _load_and_push)
        except Exception as e:
            try:
                await self._pool_run(pool, _rollback)
            except Exception:
                logger.warning(
                    "[%s] Pool join rollback failed for adapter %s",
                    request_id, adapter_name, exc_info=True,
                )
            raise BackendError(
                str(e), backend="miles", operation="create_model", original_error=e,
            ) from e

        pool.tenants[model_id] = adapter_slot
        logger.info(
            "[%s] Multi-LoRA pool join: %s -> slot %d (%d tenants)",
            request_id, adapter_name, adapter_slot, len(pool.tenants),
        )
        return MilesHandle(
            model_id=model_id,
            backend_type="miles",
            context_length=_engine_context_length(pool.hf_path),
            train_group=pool.train_group,
            rollout_manager=pool.rollout_manager,
            placement_group=None,   # pool-owned; freed only at pool teardown
            args=pool.args,
            hf_path=pool.hf_path,
            inference_endpoint=_router_url(pool.router_ip, pool.router_port),
            router_ip=pool.router_ip,
            router_port=pool.router_port,
            created_at=datetime.now().isoformat(),
            training_run_id=model_id,
            controller=pool.controller,
            adapter_name=adapter_name,
            adapter_slot=adapter_slot,
            adapter_save_dir=adapter_save_dir,
            lock=pool.lock,
        )

    async def _boot_model(
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
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> MilesHandle:
        if objective != "language_modeling":
            raise BackendError(
                f"Miles is a language-modeling backend; objective {objective!r} "
                f"requires a classification backend (automodel / megatron_bridge)",
                backend="miles", operation="create_model",
            )
        _cleanup: Dict[str, Any] = {}
        try:
            logger.info("[%s] Creating Miles model %s", request_id, model_id)

            # The actors take Megatron directories: --load is the iter_* dir
            # recorded when the checkpoint was published; --save is this
            # model's own native area, so two models' iter_* never collide.
            load_dir = resolve_native_checkpoint(str(resume_from)) if resume_from else None
            if native_root is None:
                raise BackendError("create_model needs native_root for Megatron --save",
                                   backend="miles", operation="create_model")
            # Build Slime arguments (blocking — run in thread pool)
            args, hf_path = await asyncio.to_thread(
                self.builder.build_args,
                base_model=base_model,
                lora_config=lora_config,
                debug_train_only=debug_train_only,
                load_dir=load_dir,
                save_dir=str(native_root),
                parallelism_config=parallelism,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
            )
            logger.info("[%s] Miles args built, hf_path=%s", request_id, hf_path)

            # Reuse upstream's own wiring (miles tinker-seam branch): placement
            # groups + RolloutManager from the factories train.py uses, and the
            # TinkerTrainGroup fanout for the decoupled train-step seam.
            from miles.ray.placement_group import create_placement_groups, create_rollout_manager
            from miles.ray.tinker_group import TinkerTrainGroup

            # Sync ray calls (pg.ready waits, actor allocation) — keep them off
            # the event loop or /retrieve_future polls stall and clients time out.
            pgs = await asyncio.to_thread(create_placement_groups, args)
            # Failure past PG creation must not orphan GPU reservations
            # (orphaned PGs starve every later create_model until a rotation).
            _cleanup["pgs"] = pgs

            multi_lora = bool(args.multi_lora)
            if multi_lora and debug_train_only:
                raise BackendError(
                    "Multi-LoRA pool mode requires rollout engines (adapter "
                    "weight push targets SGLang); debug_train_only unsupported",
                    backend="miles", operation="create_model",
                )

            rollout_manager: Any = None   # RolloutManager actor; None only for debug_train_only
            router_ip = None
            router_port = None
            if not debug_train_only:
                rollout_manager, _ = await asyncio.to_thread(
                    create_rollout_manager, args, pgs["rollout"]
                )
                _cleanup["rollout_manager"] = rollout_manager

            controller = None
            adapter_name = None
            adapter_slot = None
            adapter_save_dir = None
            if multi_lora:
                # Mirror the upstream driver boot: router -> controller
                # (named actor; reconcile on the actors resolves it by name).
                from miles.ray.multi_lora.controller import create_multilora_controller
                from miles.utils.adapter_config import TinkerAdapterConfig

                router_ip, router_port = await rollout_manager.get_router_address.remote()
                args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
                controller = create_multilora_controller(
                    args, f"http://{router_ip}:{router_port}"
                )
                _cleanup["controller"] = controller
                await controller.start.remote()

                adapter_name = re.sub(r"[^A-Za-z0-9._-]", "-", model_id)
                adapter_save_dir = _adapter_save_dir(native_root, adapter_name)
                registration = await controller.register_adapter.remote(
                    adapter_name,
                    TinkerAdapterConfig(
                        rank=args.lora_rank, alpha=args.lora_alpha, save=adapter_save_dir,
                    ),
                )
                adapter_slot = registration["slot"]
                logger.info(
                    "[%s] Multi-LoRA pool: adapter %s -> slot %d",
                    request_id, adapter_name, adapter_slot,
                )

            train_group = await asyncio.to_thread(lambda: TinkerTrainGroup(
                args=args,
                num_nodes=args.actor_num_nodes,
                num_gpus_per_node=args.actor_num_gpus_per_node,
                pg=pgs["actor"],
                num_gpus_per_actor=0.4,
                role="actor",
                with_ref=False,
                rollout_manager=rollout_manager,
            ))
            _cleanup["train_group"] = train_group

            try:
                await asyncio.wait_for(train_group.init(), timeout=1800.0)
            except asyncio.TimeoutError:
                raise BackendError(
                    "Actor initialization timeout after 1800s",
                    backend="miles",
                    operation="create_model",
                )

            if rollout_manager is not None:
                await train_group.set_rollout_manager()

                # Pool mode: load the registered adapter into its slot before
                # the initial push (LoRA B=0 => zero-delta; PENDING->ACTIVE).
                if multi_lora:
                    await train_group.reconcile_adapters()

                # Mirror upstream train.py startup: load weights into SGLang
                # before anything samples, honoring rollout offload state.
                if args.offload_rollout:
                    await rollout_manager.onload_weights.remote()
                await train_group.update_weights()
                if args.offload_rollout:
                    await rollout_manager.onload_kv.remote()

                router_ip = getattr(args, "sglang_router_ip", None)
                router_port = getattr(args, "sglang_router_port", None)
                if not router_ip:
                    # Upstream create_rollout_manager does not publish the
                    # router address into args on the single-tenant path
                    # (pool mode fetches it explicitly); ask the manager.
                    try:
                        router_ip, router_port = (
                            await rollout_manager.get_router_address.remote()
                        )
                    except Exception:
                        logger.error(
                            "[%s] SGLang router address missing from args",
                            request_id,
                        )

            handle = MilesHandle(
                model_id=model_id,
                backend_type="miles",
                context_length=_engine_context_length(hf_path),
                train_group=train_group,
                rollout_manager=rollout_manager,
                placement_group=pgs,
                args=args,
                hf_path=hf_path,
                inference_endpoint=_router_url(router_ip, router_port),
                router_ip=router_ip,
                router_port=router_port,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
                created_at=datetime.now().isoformat(),
                training_run_id=model_id,
                controller=controller,
                adapter_name=adapter_name,
                adapter_slot=adapter_slot,
                adapter_save_dir=adapter_save_dir,
                created_from_checkpoint=bool(resume_from),
            )

            if adapter_slot is not None:  # multi_lora: registered above
                # First tenant boots the pool; later creates join it (M2).
                pool = MilesPool(
                    train_group=train_group,
                    rollout_manager=rollout_manager,
                    placement_group=pgs,
                    controller=controller,
                    args=args,
                    hf_path=hf_path,
                    base_model=base_model,
                    router_ip=router_ip,
                    router_port=router_port,
                )
                pool.tenants[model_id] = adapter_slot
                handle.lock = pool.lock
                pool.cobatch_max_samples = self.config.cobatch_max_samples
                pool.cobatch_reorder = self.config.cobatch_reorder
                if pool.cobatch_max_samples > 0:
                    # The guard threshold is a measured per-(model x parallel
                    # config) constant, resolved from the calibration registry
                    # (a configured threshold wins; an uncalibrated config gets
                    # no merging at all). See e0_registry.
                    from .e0_registry import resolve_e0
                    res = resolve_e0(
                        base_model,
                        tp=int(args.tensor_model_parallel_size),
                        dp=_dp_size(args),
                        override=self.config.cobatch_e0_tokens,
                    )
                    pool.cobatch_e0_tokens = res.threshold
                    if res.disable_cobatch:
                        pool.cobatch_max_samples = 0
                        logger.warning(
                            "Pool co-batching requested but DISABLED: %s",
                            "calibration measured no safe region for this config"
                            if res.source == "registry"
                            else "E0 calibration missing for this config",
                        )
                pool.dispatcher = asyncio.create_task(self._pool_dispatcher_loop(pool))
                if pool.cobatch_max_samples > 0:
                    logger.info(
                        "Pool co-batching ON: merging consecutive fb up to %d samples"
                        " (reorder drain %s)",
                        pool.cobatch_max_samples,
                        "ON" if pool.cobatch_reorder else "off",
                    )
                    logger.info(
                        "Pool co-batch E0 guard: %s",
                        f"refuse merges crossing {pool.cobatch_e0_tokens} tokens/rank"
                        if pool.cobatch_e0_tokens > 0
                        else "open (no exactness boundary in the calibrated range)",
                    )
                self._pool = pool

            logger.info("[%s] Miles model %s created successfully", request_id, model_id)
            return handle

        except BackendError:
            await asyncio.to_thread(self._teardown_partial, _cleanup)
            raise
        except Exception as e:
            await asyncio.to_thread(self._teardown_partial, _cleanup)
            raise BackendError(
                str(e), backend="miles", operation="create_model", original_error=e,
            ) from e

    @staticmethod
    def _teardown_partial(cleanup: Dict[str, Any]) -> None:
        """Best-effort release of partially-booted resources (create_model
        failure path). PG removal also reaps actors placed in them."""
        train_group = cleanup.get("train_group")
        if train_group is not None:
            for actor in getattr(train_group, "_actor_handles", None) or []:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    logger.warning("teardown: train actor %r not killed", actor, exc_info=True)
        for key in ("rollout_manager", "controller"):
            actor = cleanup.get(key)
            if actor is not None:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    logger.warning("teardown: %s not killed", key, exc_info=True)
        seen = set()
        for pg_tuple in (cleanup.get("pgs") or {}).values():
            pg_obj = pg_tuple[0] if isinstance(pg_tuple, tuple) else pg_tuple
            if pg_obj is not None and id(pg_obj) not in seen:
                seen.add(id(pg_obj))
                try:
                    ray.util.remove_placement_group(pg_obj)
                except Exception:
                    logger.warning("teardown: placement group %r not removed", pg_obj, exc_info=True)
        if cleanup:
            logger.info("create_model failure teardown: released %s", sorted(cleanup))

    # ---- Pool dispatcher: the pool's serializer + M3 co-batch window ----

    async def _pool_submit(self, pool: MilesPool, op: _PoolOp) -> Any:
        if pool.dispatcher is None or pool.dispatcher.done():
            raise BackendError(
                "pool dispatcher not running", backend="miles", operation="pool",
            )
        pool.queue.put_nowait(op)
        return await op.future

    async def _pool_run(self, pool: MilesPool, run, tenant: Optional[str] = None) -> Any:
        """Serialize an async closure through the pool queue (strict FIFO).
        `tenant` marks ownership for the reorder drain; None = barrier."""
        return await self._pool_submit(pool, _PoolOp(kind="other", run=run, tenant=tenant))

    async def _pool_dispatcher_loop(self, pool: MilesPool) -> None:
        """Single consumer of pool.queue. FIFO order is the execution order;
        the only transformation is merging same-loss_fn fb ops (any tenants)
        into one mixed-slot train call.

        Conservative drain (default): only CONSECUTIVE fb ops merge — an op
        of any other kind ends the drain, so a step never overtakes or
        absorbs a later fb. Pipelined recipe traffic (fb+step back-to-back
        per tenant) therefore never merges: the tenant's own step closes
        the window microseconds after its fb.

        Reorder drain (cobatch_reorder): only PER-TENANT submission order is
        contractual; cross-tenant order is scheduler freedom (isolation
        invariant, gate G4). The drain may defer other ops past the merge
        window and replay them, in order, right after the merged call.
        Per-tenant FIFO is exact: once ANY op of tenant t is deferred, t is
        blocked — its later fb cannot join the batch (would overtake the
        deferred op) and its later ops all defer (keep relative order).
        Barrier ops (tenant None: stop / pool admin) still end the drain."""
        carry: Optional[_PoolOp] = None
        pending: List[_PoolOp] = []       # reorder drain: deferred ops
        while True:
            if carry is not None:
                # carry left the queue after everything now in pending —
                # append keeps replay order == queue arrival order.
                pending.append(carry)
                carry = None
            if pending:
                op = pending.pop(0)
            else:
                op = await pool.queue.get()
            if op.kind == "stop":
                for p in pending:         # drain deferred before stopping
                    if not p.future.done():
                        p.future.set_exception(
                            BackendError("pool stopped", backend="miles", operation="pool")
                        )
                if not op.future.done():
                    op.future.set_result(None)
                return
            if op.kind == "other":
                try:
                    result = await op.run()
                    if not op.future.done():
                        op.future.set_result(result)
                except Exception as e:  # noqa: BLE001 — surfaced via the future
                    if not op.future.done():
                        op.future.set_exception(e)
                continue
            batch = [op]
            total = op.num_samples
            blocked = {p.tenant for p in pending}
            deferred_now = 0
            while pool.cobatch_max_samples > 0:
                try:
                    nxt = pool.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if (
                    nxt.kind == "fb"
                    and nxt.loss_fn == op.loss_fn
                    and nxt.tenant not in blocked
                    and total + nxt.num_samples <= pool.cobatch_max_samples
                ):
                    if self.converter.cobatch_preserves_token_bucket(
                        [o.rollout_data for o in batch] + [nxt.rollout_data],
                        _dp_size(pool.args),
                        pool.cobatch_e0_tokens,
                    ):
                        batch.append(nxt)
                        total += nxt.num_samples
                    else:
                        # E0 guard refusal ENDS the merge window: deferring a
                        # refused fb would convoy it behind blocked-tenant
                        # bookkeeping and replay for no gain (the next window
                        # re-evaluates it as head).
                        carry = nxt
                        break
                elif (
                    pool.cobatch_reorder
                    and nxt.kind != "stop"
                    and nxt.tenant is not None
                ):
                    pending.append(nxt)   # defer; replayed in order after the call
                    blocked.add(nxt.tenant)
                    deferred_now += 1
                else:
                    carry = nxt   # FIFO: this op becomes the next head
                    break
            if deferred_now:
                logger.info(
                    "Reorder drain: deferred %d op(s) past the merge window",
                    deferred_now,
                )
            try:
                per_request = await self._execute_fb_batch(pool, batch)
                for o, r in zip(batch, per_request):
                    if not o.future.done():
                        o.future.set_result(r)
            except Exception as e:  # noqa: BLE001 — surfaced via the futures
                for o in batch:
                    if not o.future.done():
                        o.future.set_exception(e)

    async def _execute_fb_batch(self, pool: MilesPool, batch: List[_PoolOp]) -> List[Dict[str, Any]]:
        """One train call for 1..k merged fb requests; split results back."""
        from miles.utils.ray_utils import Box
        from miles.ray.tinker_group import merge_dp_sample_outputs

        if pool.rollout_manager is not None and pool.args.offload_rollout:
            await pool.rollout_manager.offload.remote()

        merged = self.converter.merge_forward_backward_batches(
            [o.rollout_data for o in batch]
        )
        # Pad AFTER merging, never per request: pads must sit at the tail of
        # what is actually dispatched, or the per-request offsets below would
        # slice across them. See converter.pad_rollout_data_to_dp.
        n_pad = self.converter.pad_rollout_data_to_dp(merged, _dp_size(pool.args))
        if n_pad:
            logger.info(
                "Padded co-batched fb with %d inert sample(s) to align with dp=%d",
                n_pad, _dp_size(pool.args),
            )
        if len(batch) > 1:
            logger.info(
                "Co-batched fb: %d requests / %d samples / tenants %s",
                len(batch), sum(o.num_samples for o in batch),
                sorted({o.handle.model_id for o in batch}),
            )

        results = await pool.train_group.forward_backward_only(0, Box(ray.put(merged)))

        # Batch-global metrics (per-tenant scalar loss is not separable in a
        # mixed batch; per-datum outputs below are exact per tenant).
        summed: Dict[str, float] = {}
        reporting = 0
        for r in results or []:
            loss_dict = (r or {}).get("loss") or {}
            if loss_dict:
                reporting += 1
                for k, v in loss_dict.items():
                    summed[k] = summed.get(k, 0.0) + float(v)
        averaged = {k: v / reporting for k, v in summed.items()} if reporting else {}
        metrics = {f"{k}:mean": v for k, v in averaged.items()}
        if len(batch) > 1:
            metrics["co_batched_fb:max"] = float(len(batch))

        logprobs_list = merge_dp_sample_outputs(results or [], key="log_probs")
        expected = sum(o.num_samples for o in batch)
        # Drop the alignment pads before anything client-visible: a pad has no
        # observation key, so it must not reach a caller. Pads are at the tail
        # by construction, so this is a slice, and the equality check below
        # still guards the real per-request split.
        if n_pad and len(logprobs_list) == expected + n_pad:
            logprobs_list = logprobs_list[:expected]
        if len(logprobs_list) != expected:
            raise BackendError(
                f"co-batched fb returned {len(logprobs_list)} per-sample "
                f"outputs for {expected} samples (n_pad={n_pad})",
                backend="miles", operation="forward_backward",
            )

        per_request = []
        offset = 0
        for o in batch:
            lps = logprobs_list[offset:offset + o.num_samples]
            offset += o.num_samples
            # Observation contract: per-datum logprobs are datum-aligned —
            # exactly one per model_input token, entry k = logprob of target k
            # (the converter appends the final target so the response covers
            # all T targets). Any other length is a layout error: refuse
            # rather than guess (reassemble-or-refuse).
            in_lens = o.input_lens or []
            outs = []
            for j, lp in enumerate(lps):
                lp_list = lp.tolist()
                full = in_lens[j] if j < len(in_lens) and in_lens[j] > 0 else len(lp_list)
                if len(lp_list) != full:
                    raise BackendError(
                        f"sample {j}: {len(lp_list)} logprobs for "
                        f"{full}-token input",
                        backend="miles", operation="forward_backward",
                    )
                outs.append({"logprobs": {
                    "data": lp_list, "shape": [len(lp_list)], "dtype": "float32",
                }})
            per_request.append({
                "loss_fn_output_type": o.loss_fn,
                "loss": averaged.get("loss"),
                "metrics": dict(metrics),
                "loss_fn_outputs": outs,
                "deferred": False,
            })
        return per_request

    async def forward(
        self,
        handle: MilesHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        async def _run() -> Dict[str, Any]:
            from miles.utils.ray_utils import Box

            if handle.rollout_manager is not None and handle.args.offload_rollout:
                await handle.rollout_manager.offload.remote()

            rollout_data = self.converter.forward_to_backend(data, handle.args, adapter_slot=handle.adapter_slot)
            # Same DP alignment as fb: fewer samples than DP ranks gives an
            # actor an empty local batch (get_data_iterator divides by zero).
            n_pad = self.converter.pad_rollout_data_to_dp(rollout_data, _dp_size(handle.args))
            if n_pad:
                logger.info("Padded forward with %d inert sample(s) to align with dp=%d",
                            n_pad, _dp_size(handle.args))
            # TinkerTrainGroup returns per-sample logprob tensors already
            # merged into the client's submission order.
            logprobs = await handle.train_group.forward_logprobs(0, Box(ray.put(rollout_data)))
            if n_pad:
                if len(logprobs) != len(data) + n_pad:
                    raise BackendError(
                        f"forward returned {len(logprobs)} per-sample outputs for "
                        f"{len(data)} samples + {n_pad} pads",
                        backend="miles", operation="forward",
                    )
                logprobs = logprobs[:len(data)]

            loss_fn_outputs = [
                {"logprobs": {"data": lp.tolist(), "shape": [len(lp)], "dtype": "float32"}}
                for lp in logprobs
            ]
            return {
                "type": "forward",
                "loss_fn_output_type": loss_fn,
                "loss_fn_outputs": loss_fn_outputs,
                "metrics": {},
            }

        try:
            # Same sample-count gate as fb: a request smaller than the DP
            # width crashes get_data_iterator on the actors (num_local_gbs=0).
            self._validate_fb(handle, data)
            pool = self._pool
            if handle.adapter_slot is not None and pool is not None:
                return await self._pool_run(pool, _run, tenant=handle.model_id)
            await handle.lock.acquire()
            try:
                return await _run()
            finally:
                handle.lock.release()
        except (BackendError, ValueError):
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward", original_error=e,
            ) from e

    @staticmethod
    def _validate_fb(h: MilesHandle, data: List[Datum]) -> None:
        from ...core.validators import RequestValidator
        from ...config import get_config

        is_rl = not h.args.debug_train_only
        config = get_config()
        allow_partial = config.allow_partial_batches
        validator = RequestValidator(h.args, allow_partial_batches=allow_partial)
        # Validate what will actually be DISPATCHED, not what arrived: the
        # converter pads the batch up to a multiple of dp before it reaches the
        # actors (pad_rollout_data_to_dp), so a client sending an indivisible
        # count is legal and must not be refused. The validator's own
        # divisibility rule stays strict as a backstop for any path that
        # dispatches without padding.
        dp = _dp_size(h.args)
        padded = -(-len(data) // dp) * dp if dp > 1 else len(data)
        validation_error = validator.validate_sample_count(padded, is_rl=is_rl)
        if validation_error:
            raise ValueError(
                f"Request validation failed:\n{validation_error}\n\n"
                f"{validator.get_config_summary()}"
            )

    async def forward_backward(
        self,
        handle: MilesHandle,
        data: List[Datum],
        loss_fn: str,
        loss_fn_config: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        _check_miles_clip_config(handle, loss_fn, loss_fn_config)
        pool = self._pool
        if handle.adapter_slot is not None and pool is not None:
            # Pool path: validate + convert here (CPU), then queue the GPU
            # work — the dispatcher serializes and may co-batch it (M3).
            try:
                self._validate_fb(handle, data)
                rollout_data = self.converter.forward_backward_to_backend(
                    data, loss_fn, handle.args, adapter_slot=handle.adapter_slot,
                )
                return await self._pool_submit(pool, _PoolOp(
                    kind="fb", handle=handle, rollout_data=rollout_data,
                    loss_fn=loss_fn, num_samples=len(data), tenant=handle.model_id,
                    input_lens=_model_input_lens(data),
                ))
            except (BackendError, ValueError):
                raise
            except Exception as e:
                raise BackendError(
                    str(e), backend="miles", operation="forward_backward", original_error=e,
                ) from e
        await handle.lock.acquire()
        try:
            from miles.utils.ray_utils import Box

            if handle.rollout_manager is not None and handle.args.offload_rollout:
                await handle.rollout_manager.offload.remote()

            self._validate_fb(handle, data)

            rollout_data = self.converter.forward_backward_to_backend(
                data, loss_fn, handle.args, adapter_slot=handle.adapter_slot,
            )
            n_pad = self.converter.pad_rollout_data_to_dp(rollout_data, _dp_size(handle.args))
            if n_pad:
                logger.info(
                    "Padded fb with %d inert sample(s) to align with dp=%d",
                    n_pad, _dp_size(handle.args),
                )

            results = await handle.train_group.forward_backward_only(0, Box(ray.put(rollout_data)))

            # Only pipeline-last-stage actors return metrics; average across
            # the DP ranks that did. Per-sample logprobs are not emitted by the
            # seam's fb pass itself (they ride a separate forward).
            summed: Dict[str, float] = {}
            reporting = 0
            for r in results or []:
                loss_dict = (r or {}).get("loss") or {}
                if loss_dict:
                    reporting += 1
                    for k, v in loss_dict.items():
                        summed[k] = summed.get(k, 0.0) + float(v)
            averaged = {k: v / reporting for k, v in summed.items()} if reporting else {}
            # SDK metric keys carry their cross-chunk reduction as ":<type>"
            # (chunked_fwdbwd_helpers._metrics_reduction splits on ":").
            metrics = {f"{k}:mean": v for k, v in averaged.items()}

            # Per-datum response logprobs in client order (the SDK weights its
            # metric reduction by len(loss_fn_outputs), and the cookbook
            # computes NLL from these).
            from miles.ray.tinker_group import merge_dp_sample_outputs
            logprobs_list = merge_dp_sample_outputs(results or [], key="log_probs")
            # Alignment pads carry no observation key — drop them before the
            # client sees anything (they are at the tail by construction).
            if n_pad:
                if len(logprobs_list) != len(data) + n_pad:
                    raise BackendError(
                        f"fb returned {len(logprobs_list)} per-sample outputs for "
                        f"{len(data)} samples + {n_pad} pads",
                        backend="miles", operation="forward_backward",
                    )
                logprobs_list = logprobs_list[:len(data)]
            loss_fn_outputs = [
                {"logprobs": {"data": lp.tolist(), "shape": [len(lp)], "dtype": "float32"}}
                for lp in logprobs_list
            ]

            return {
                "loss_fn_output_type": loss_fn,
                "loss": averaged.get("loss"),
                "metrics": metrics,
                "loss_fn_outputs": loss_fn_outputs,
                "deferred": False,
            }

        except BackendError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward_backward", original_error=e,
            ) from e
        finally:
            handle.lock.release()

    async def apply_optimizer_step(
        self,
        handle: MilesHandle,
        learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # adam_params accepted for contract uniformity (P4); Miles applies lr
        # only — betas/eps are Megatron args fixed at creation.

        async def _run() -> Dict[str, Any]:
            # TinkerTrainGroup: apply_optimizer_step(learning_rate) fans out to
            # the actors; _and_sync additionally pushes weights to SGLang.
            offload_train = handle.args.offload_train if handle.args else True
            offload_rollout = handle.args.offload_rollout if handle.args else True

            step_kwargs = {"adapter_slot": handle.adapter_slot, "adapter_name": handle.adapter_name}
            if handle.rollout_manager is None:
                results = await handle.train_group.apply_optimizer_step(learning_rate, **step_kwargs)
            elif not offload_train and not offload_rollout:
                # Pool mode rides this arm: the sync pushes exactly the stepped
                # adapter (per-adapter upsert via the pending set).
                results = await handle.train_group.apply_optimizer_step_and_sync(learning_rate, **step_kwargs)
            else:
                results = await handle.train_group.apply_optimizer_step(learning_rate, **step_kwargs)

                # Mirror upstream train.py's offload dance around weight sync.
                if offload_train:
                    await handle.train_group.offload()
                if offload_rollout:
                    await handle.rollout_manager.onload_weights.remote()
                await handle.train_group.update_weights()
                if offload_rollout:
                    await handle.rollout_manager.onload_kv.remote()

            if results[0]["success"]:
                handle.weight_version += 1

            return {
                "success": results[0]["success"],
                "grad_norm": results[0]["grad_norm"],
                "learning_rates": [],
                "model_id": handle.model_id,
            }

        try:
            pool = self._pool
            if handle.adapter_slot is not None and pool is not None:
                return await self._pool_run(pool, _run, tenant=handle.model_id)
            await handle.lock.acquire()
            try:
                return await _run()
            finally:
                handle.lock.release()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="apply_optimizer_step", original_error=e,
            ) from e

    async def update_inference_weights(self, handle: MilesHandle) -> None:

        async def _run() -> None:
            await handle.train_group.update_weights()

        try:
            pool = self._pool
            if handle.adapter_slot is not None and pool is not None:
                await self._pool_run(pool, _run, tenant=handle.model_id)
                return
            await handle.lock.acquire()
            try:
                await _run()
            finally:
                handle.lock.release()
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="update_inference_weights", original_error=e,
            ) from e

    async def save_checkpoint(
        self,
        handle: MilesHandle,
        root: Path,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> None:
        if not persist:
            return  # ephemeral sampler save: update_weights already delivered them to SGLang

        async def _run() -> None:
            offload_train = handle.args.offload_train if handle.args else False
            if offload_train:
                # Never return a path nothing was written to.
                raise UnsupportedFeatureError(
                    "save_checkpoint with offload_train", backend="miles",
                    suggestion="boot the actor group without --offload-train",
                )

            # Pool mode: save_due_adapter_checkpoints only writes adapters with
            # registry step > 0 at a save-interval multiple (interval is 1 at
            # pool boot, builder.py). The registry step is miles' own
            # training-loop counter and nothing advances it here — the CLIENT
            # owns the loop — so publish our weight version (= applied optimizer
            # steps) as the step this checkpoint represents.
            if handle.adapter_slot is not None and handle.controller is not None:
                if handle.weight_version <= 0:
                    logger.info(
                        "Adapter %s has taken no optimizer step; nothing to checkpoint",
                        handle.adapter_name,
                    )
                    return
                await handle.controller.set_adapter_step.remote(handle.adapter_name, handle.weight_version)

            await handle.train_group.save_model(step if step is not None else 0)

            if handle.adapter_save_dir:
                await asyncio.to_thread(
                    _publish_adapter, handle.adapter_save_dir, str(root), handle.adapter_name,
                )
            else:
                await asyncio.to_thread(_publish_native_adapter, handle.args, str(root))

        try:
            pool = self._pool
            if handle.adapter_slot is not None and pool is not None:
                await self._pool_run(pool, _run, tenant=handle.model_id)
                return
            await handle.lock.acquire()
            try:
                await _run()
            finally:
                handle.lock.release()
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="save_checkpoint", original_error=e,
            ) from e

    async def load_checkpoint(
        self,
        handle: MilesHandle,
        root: Path,
        optimizer: bool = False,
    ) -> None:
        if handle.adapter_slot is not None:
            # train_group.load_checkpoint is a full-model resume broadcast;
            # on shared rails it would clobber every co-tenant.
            raise BackendError(
                "load_checkpoint is not supported in multi-LoRA pool mode "
                "(adapter-scoped resume unimplemented)",
                backend="miles", operation="load_checkpoint",
            )
        await handle.lock.acquire()
        try:
            # The actors take Megatron's --load directory: the iter_* dir
            # recorded under the checkpoint root when it was published.
            load_dir = resolve_native_checkpoint(str(root))
            await handle.train_group.load_checkpoint(load_dir, load_optimizer=optimizer)
            handle.created_from_checkpoint = True

            # Sync loaded weights to inference engine
            if handle.rollout_manager is not None:
                await handle.train_group.update_weights()

            logger.info("Miles checkpoint loaded from %s", root)

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="load_checkpoint", original_error=e,
            ) from e
        finally:
            handle.lock.release()

    async def delete_model(self, handle: MilesHandle) -> None:
        if handle.adapter_slot is not None and self._pool is not None:
            await self._delete_pool_tenant(handle)
            return
        # Hold the op lock so teardown can't interleave with an in-flight
        # fb/step (delete-during-optim_step crash class).
        await handle.lock.acquire()
        try:
            resources_freed = []
            # Fallback for a pool handle that outlived its pool record:
            # kill the named controller so the next create_model can
            # register a fresh one.
            if handle.controller is not None:
                try:
                    await handle.controller.stop.remote()
                except Exception:
                    logger.warning("Multi-LoRA controller stop failed; killing", exc_info=True)
                ray.kill(handle.controller, no_restart=True)
                resources_freed.append("multi_lora_controller")

            for actor in handle.train_group._actor_handles:
                ray.kill(actor, no_restart=True)
                resources_freed.append("actor")

            if handle.rollout_manager is not None:
                ray.kill(handle.rollout_manager, no_restart=True)
                resources_freed.append("rollout_manager")

            # placement_group holds the create_placement_groups() dict of
            # (pg, bundle_indices, gpu_ids) tuples; pgs may be shared between
            # roles (colocate), so dedupe before removal.
            if handle.placement_group:
                seen = set()
                for pg_tuple in handle.placement_group.values():
                    pg_obj = pg_tuple[0] if isinstance(pg_tuple, tuple) else pg_tuple
                    # debug_train_only leaves the rollout entry as None
                    if pg_obj is not None and id(pg_obj) not in seen:
                        seen.add(id(pg_obj))
                        ray.util.remove_placement_group(pg_obj)
                        resources_freed.append("placement_group")

            logger.info("Miles model %s deleted, freed %d resources", handle.model_id, len(resources_freed))

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="delete_model", original_error=e,
            ) from e
        finally:
            handle.lock.release()

    async def _delete_pool_tenant(self, h: MilesHandle) -> None:
        """Pool-mode delete (M2): deregister this tenant's adapter; the last
        tenant out tears the pool down (controller name freed for reboot)."""
        async with self._pool_admin:
            pool = self._pool
            if pool is None:
                return
            if len(pool.tenants) > 1 or h.model_id not in pool.tenants:
                async def _retire() -> None:
                    await pool.controller.deregister_adapter.remote(h.adapter_name)
                    # Actors retire the slot: abort in-flight sampling,
                    # save the final adapter ckpt, clear slot weights /
                    # optimizer state / retained grads, free the slot.
                    await pool.train_group.reconcile_adapters()

                try:
                    await self._pool_run(pool, _retire, tenant=h.model_id)
                except Exception as e:
                    raise BackendError(
                        str(e), backend="miles", operation="delete_model", original_error=e,
                    ) from e
                pool.tenants.pop(h.model_id, None)
                logger.info(
                    "Pool tenant %s deregistered (slot %s); %d tenant(s) remain",
                    h.model_id, h.adapter_slot, len(pool.tenants),
                )
                return
            # Last tenant: the pool dies with it. Null the record even on a
            # partial teardown — a half-dead pool must not accept joins.
            try:
                # Drain + stop the dispatcher first: FIFO means every
                # already-queued op completes before the stop resolves, so
                # the kills below never race an in-flight broadcast.
                try:
                    await self._pool_submit(pool, _PoolOp(kind="stop"))
                except BackendError:
                    pass  # dispatcher already dead — proceed to kills
                resources_freed = []
                try:
                    await pool.controller.stop.remote()
                except Exception:
                    logger.warning("Multi-LoRA controller stop failed; killing", exc_info=True)
                ray.kill(pool.controller, no_restart=True)
                resources_freed.append("multi_lora_controller")
                for actor in pool.train_group._actor_handles:
                    ray.kill(actor, no_restart=True)
                    resources_freed.append("actor")
                if pool.rollout_manager is not None:
                    ray.kill(pool.rollout_manager, no_restart=True)
                    resources_freed.append("rollout_manager")
                seen = set()
                for pg_tuple in (pool.placement_group or {}).values():
                    pg_obj = pg_tuple[0] if isinstance(pg_tuple, tuple) else pg_tuple
                    if pg_obj is not None and id(pg_obj) not in seen:
                        seen.add(id(pg_obj))
                        ray.util.remove_placement_group(pg_obj)
                        resources_freed.append("placement_group")
            except Exception as e:
                raise BackendError(
                    str(e), backend="miles", operation="delete_model", original_error=e,
                ) from e
            finally:
                self._pool = None
            logger.info(
                "Miles pool torn down with last tenant %s, freed %d resources",
                h.model_id, len(resources_freed),
            )

    async def get_logprobs(
        self,
        handle: MilesHandle,
        data: List[Datum],
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
        handle: MilesHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample via per-request HTTP calls to the SGLang router.

        Pool mode honors pinned_version==0 by routing to the BASE weights
        (fresh-init LoRA delta is zero at v0, so v0 == base): a sampler
        saved before any optim step is a frozen reference — DPO's ref model
        with no second copy resident. Nonzero pins (and every pin outside
        pool mode) are still served from the live engine — the BUG-015
        aliasing class; logged loudly rather than silently aliased.
        """
        client = self._sglang_client(handle, "sample")

        # Pool mode: route to this model's adapter by engine-side slot name.
        lora_path = None
        if handle.adapter_slot is not None:
            if pinned_version == 0 and not handle.created_from_checkpoint:
                pass  # v0 == base: no lora_path
            else:
                if pinned_version is not None:
                    logger.warning(
                        "[%s] pinned_version=%s not honorable for %s "
                        "(nonzero or checkpoint-created); serving LIVE slot "
                        "weights (v%d) — BUG-015 aliasing risk",
                        request_id, pinned_version, handle.model_id, handle.weight_version,
                    )
                from miles.utils.multi_lora import slot_lora_name

                lora_path = slot_lora_name(handle.adapter_slot)
        elif pinned_version is not None and pinned_version != handle.weight_version:
            logger.warning(
                "[%s] pinned_version=%s not honored for %s (non-pool miles "
                "serves the live engine, v%d) — BUG-015 aliasing risk",
                request_id, pinned_version, handle.model_id, handle.weight_version,
            )

        sequences = []
        prompt_logprobs_result = None
        base_params = dict(sampling_params or {})
        if base_params.get("seed") is not None and not handle.args.sglang_enable_deterministic_inference:
            if not handle.seed_warned:
                handle.seed_warned = True
                logger.warning(
                    "[%s] sampling seed given but SGLang was booted without deterministic "
                    "inference: the seed is ignored. Set SLIME_SGLANG_DETERMINISTIC=1 before create_model.",
                    request_id,
                )
        for i in range(num_samples):
            params = dict(base_params)
            if params.get("seed") is not None:
                params["seed"] = int(params["seed"]) + i  # distinct stream per sample
            result = await client.generate(
                input_ids=prompt_tokens,
                sampling_params=params,
                prompt_logprobs=prompt_logprobs,
                lora_path=lora_path,
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

    async def prepare_for_generation(self, handle: MilesHandle) -> None:
        """SGLang router is always live for Miles — just validate it is routable."""
        self._sglang_client(handle, "prepare_for_generation")

    def _sglang_client(self, handle: MilesHandle, operation: str) -> SGLangClient:
        """The pooled client for the endpoint currently published for this
        model (core.routing), not the address the handle was booted with."""
        try:
            endpoint = routing.table.endpoint_for(handle.model_id)
        except routing.RoutingError as e:
            raise BackendError(
                "SGLang router not available", backend="miles", operation=operation,
            ) from e
        return self._sglang_pool.for_endpoint(endpoint.base_url)

    async def close(self) -> None:
        await self._sglang_pool.aclose()
