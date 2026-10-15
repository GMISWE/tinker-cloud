"""
The Automodel backend: one FSDP2 training group per (base model, LoRA
signature), one tenant per group in this build, and every Tinker primitive run
when it is called (R9).

A model's requests reach its group through the group's scheduler: a pass
coalesces with the consecutive same-key passes behind it (R15); a barrier
(optimizer step, delete) runs alone. One pass is: convert each request
(validated alone, so a bad request fails alone) -> coalesce into one batch
padded to a multiple of the rank count -> one shard per rank -> every rank
runs it -> the rows come back in rank order and are split per request (R3).

Faults: a non-finite grad norm fails that optimizer step only; the grads were
zeroed on every rank and the group goes on (R10). Any other failure inside a
rank call is the group's: its actors and bundles are released, the queued
requests fail, and every later call on its models names the cause (R5).
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from ...core.loss_registry import clip_thresholds
from ...models.requests import AdamParams, Datum, LoraConfig
from ...utils.model_config import model_context_length, read_raw_hf_config
from ..base import BackendError, BackendHandle, TrainingBackend, UnsupportedFeatureError
from ..objectives import classification_spec
from .config import AutomodelConfig
from .converter import Batch, LossFn, RequestRows, coalesce, convert_request, row_sums, shard, split_rows
from .group import TrainingGroup, sweep_leftovers
from .lora import LoraSpec, lora_spec, tenant_seed
from .scheduler import GroupScheduler, PassKey, pass_key
from .errors import NonFiniteGradNorm

logger = logging.getLogger(__name__)

BACKEND = "automodel"


@dataclass
class AutomodelHandle(BackendHandle):
    backend_type: str = BACKEND
    base_model: str = ""
    lora: Optional[LoraSpec] = None
    group_key: str = ""


@dataclass
class _Group:
    key: str
    ranks: TrainingGroup          # or a stand-in with the same four coroutines
    scheduler: GroupScheduler
    tenants: Set[str] = field(default_factory=set)
    lost: Optional[str] = None    # why; set before the teardown runs


def group_key(base_model: str, lora: LoraSpec) -> str:
    """Groups are keyed by base model and LoRA signature (R2.4). The key is
    also the actor and placement-group name, so it is short and plain."""
    return hashlib.sha256(f"{base_model}|{lora.signature}".encode()).hexdigest()[:12]


class AutomodelBackend(TrainingBackend[AutomodelHandle]):
    needs_ray = True
    config: AutomodelConfig
    SUPPORTED_LOSS_FNS = frozenset(m.value for m in LossFn)   # the service's table is keyed by the wire string

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        super().__init__(overrides)
        self.config = AutomodelConfig.from_env(overrides)
        self._groups: Dict[str, _Group] = {}
        self._lost: Dict[str, str] = {}      # group key -> why it was torn down
        self._tasks: Set[asyncio.Task] = set()
        self._swept = False

    # --- lifecycle -------------------------------------------------------
    async def create_model(
        self, model_id: str, request_id: str, base_model: str, num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None, parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None, rollout_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> AutomodelHandle:
        if parallelism is not None:
            raise UnsupportedFeatureError("parallelism_config", backend=BACKEND,
                                          suggestion="the group size is the server's AUTOMODEL_TRAIN_GPUS")
        if lora_config is None:
            raise BackendError("full fine-tuning is not served by this build: send a lora_config",
                               backend=BACKEND, operation="create_model")
        config = LoraConfig.model_validate(lora_config)
        try:
            lora = lora_spec(config)
        except ValueError as e:
            raise BackendError(str(e), backend=BACKEND, operation="create_model") from e
        if lora.rank > self.config.max_lora_rank:
            raise BackendError(f"LoRA rank {lora.rank} is above the server's max_lora_rank {self.config.max_lora_rank}",
                               backend=BACKEND, operation="create_model")
        raw = await asyncio.to_thread(read_raw_hf_config, base_model)
        classification = classification_spec(raw)
        if classification is not None:
            raise BackendError(
                f"the automodel backend is language-modeling only; {base_model!r} declares a "
                f"{classification.objective.value} head, which the megatron_bridge backend serves",
                backend=BACKEND, operation="create_model")

        if not self._swept:   # before the first group, so a crashed predecessor's leftovers do not hold its GPUs
            found = sweep_leftovers()
            if found:
                logger.warning("swept %d leftover training-group actors/placement groups", found)
            self._swept = True

        key = group_key(base_model, lora)
        if key in self._groups:
            tenants = ", ".join(sorted(self._groups[key].tenants))
            raise BackendError(f"training group {key} already serves {tenants}; this build holds one tenant per group",
                               backend=BACKEND, operation="create_model")
        ranks = TrainingGroup(key, self.config.train_gpus)
        logger.info("[%s] starting training group %s: %s, %d ranks, LoRA %s",
                    request_id, key, base_model, self.config.train_gpus, lora.signature)
        lora_numel = await ranks.start(base_model, lora, tenant_seed(model_id, config.seed), self.config.micro_batch_rows)
        scheduler = GroupScheduler(key, lambda pk, payloads: self._run_pass(key, pk, payloads))
        self._groups[key] = _Group(key, ranks, scheduler, {model_id})
        if key in self._lost:
            del self._lost[key]
        logger.info("[%s] training group %s up: LoRA %d elements", request_id, key, lora_numel)
        return AutomodelHandle(
            model_id=model_id, context_length=model_context_length(raw),
            hf_path=base_model, base_model=base_model, lora=lora, group_key=key)

    async def delete_model(self, handle: AutomodelHandle) -> None:
        key = handle.group_key
        if key not in self._groups:
            return   # torn down after a fault: nothing left to free
        group = self._groups[key]
        group.tenants.discard(handle.model_id)
        if group.tenants:
            return   # not in this build: one tenant per group
        await group.scheduler.close(f"model {handle.model_id} was deleted")
        await group.ranks.stop()
        if key in self._groups and self._groups[key] is group:
            del self._groups[key]

    async def close(self) -> None:
        for group in list(self._groups.values()):
            await group.scheduler.close("server shutdown")
            await group.ranks.stop()
        self._groups.clear()

    # --- training --------------------------------------------------------
    async def forward(self, handle: AutomodelHandle, data: List[Datum], loss_fn: str,
                      loss_fn_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return await self._submit(handle, "forward", data, loss_fn, loss_fn_config)

    async def forward_backward(self, handle: AutomodelHandle, data: List[Datum], loss_fn: str,
                               loss_fn_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        return await self._submit(handle, "forward_backward", data, loss_fn, loss_fn_config)

    async def get_logprobs(self, handle: AutomodelHandle, data: List[Datum]) -> List[Any]:
        result = await self.forward(handle, data, LossFn.CROSS_ENTROPY)
        return [out["logprobs"]["data"] for out in result["loss_fn_outputs"]]

    async def apply_optimizer_step(
        self, handle: AutomodelHandle, learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        group = self._live_group(handle, "apply_optimizer_step")
        if learning_rate is None or adam_params is None:
            raise BackendError(
                "the request carries no adam_params; the learning rate and clip are the request's (R11)",
                backend=BACKEND,
                operation="apply_optimizer_step",
            )
        adam = AdamParams(learning_rate=learning_rate, **adam_params)

        async def step() -> Dict[str, Any]:
            try:
                outs = await group.ranks.optim_step(adam)
            except NonFiniteGradNorm as e:
                # Every rank refused (the norm is all-reduced) and zeroed its
                # grads; this request fails, but the group remains usable.
                raise BackendError(str(e), backend=BACKEND, operation="apply_optimizer_step") from e
            except Exception as e:
                self._mark_lost(group, f"optim_step failed on a rank: {e!r}")
                raise BackendError(
                    "optim_step failed; the training group was released",
                    backend=BACKEND,
                    operation="apply_optimizer_step",
                    original_error=e,
                ) from e
            norm = float(outs[0]["grad_norm"])
            handle.weight_version += 1
            return {"grad_norm": norm, "weight_version": handle.weight_version}

        return await group.scheduler.submit_barrier(handle.model_id, step)

    async def _submit(self, handle: AutomodelHandle, kind: str, data: List[Datum], loss_fn: str,
                      loss_fn_config: Optional[Dict[str, float]]) -> Dict[str, Any]:
        group = self._live_group(handle, kind)
        try:
            rows = await asyncio.to_thread(convert_request, data, loss_fn)
        except ValueError as e:
            raise BackendError(str(e), backend=BACKEND, operation=kind) from e
        return await group.scheduler.submit_pass(pass_key(handle.model_id, kind, rows.loss_fn, loss_fn_config), rows)

    async def _run_pass(self, key: str, pk: PassKey, payloads: List[RequestRows]) -> List[Dict[str, Any]]:
        """The scheduler's run_pass: one coalesced pass on every rank, one result per request."""
        group = self._groups[key]
        if group.lost is not None:
            raise BackendError(f"training group {key} is lost: {group.lost}", backend=BACKEND, operation=pk.kind)
        n = group.ranks.num_ranks
        batch = await asyncio.to_thread(coalesce, payloads, n)
        shards = shard(batch, n)
        clip = clip_thresholds(dict(pk.loss_fn_config)) if batch.loss_fn is LossFn.PPO else None
        run = group.ranks.forward_backward if pk.kind == "forward_backward" else group.ranks.forward
        try:
            outs = await run(shards, clip)
        except Exception as e:
            self._mark_lost(group, f"{pk.kind} failed on a rank: {e!r}")
            raise BackendError(f"{pk.kind} failed; the training group was released",
                               backend=BACKEND, operation=pk.kind, original_error=e) from e
        return _per_request(batch, outs)

    # --- faults ----------------------------------------------------------
    def _live_group(self, handle: AutomodelHandle, op: str) -> _Group:
        """The handle's group, if it is up; otherwise the error names why it is not."""
        key = handle.group_key
        if key in self._groups and self._groups[key].lost is None:
            return self._groups[key]
        why = self._lost[key] if key in self._lost else "the model was deleted"
        raise BackendError(f"model {handle.model_id} has no training group: {why}", backend=BACKEND, operation=op)

    def _mark_lost(self, group: _Group, reason: str) -> None:
        """Runs inside the group's scheduler. close() waits for the op in
        progress, which is this one, so the teardown is a task of its own."""
        if group.lost is not None:
            return
        group.lost = reason
        self._lost[group.key] = reason
        logger.error("training group %s is lost: %s", group.key, reason)
        task = asyncio.create_task(self._teardown(group, reason))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _teardown(self, group: _Group, reason: str) -> None:
        await group.scheduler.close(reason)
        await group.ranks.stop()
        if group.key in self._groups and self._groups[group.key] is group:
            del self._groups[group.key]

    # --- not in this build -----------------------------------------------
    @staticmethod
    def _not_served(op: str, what: str) -> BackendError:
        return BackendError(f"{op} is not served by this build of the automodel backend ({what})",
                            backend=BACKEND, operation=op)

    async def update_inference_weights(self, handle: AutomodelHandle) -> None:
        raise self._not_served("update_inference_weights", "no sampler host")

    async def prepare_for_generation(self, handle: AutomodelHandle) -> None:
        raise self._not_served("prepare_for_generation", "no sampler host")

    async def sample(
        self, handle: AutomodelHandle, request_id: str, prompt_tokens: List[int], num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None, prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None, topk_sample_logprobs: int = 0, topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        raise self._not_served("sample", "no sampler host")

    async def save_checkpoint(self, handle: AutomodelHandle, root: Path, step: Optional[int] = None,
                              persist: bool = True) -> None:
        raise self._not_served("save_checkpoint", "no checkpoint writer")

    async def load_checkpoint(self, handle: AutomodelHandle, root: Path, optimizer: bool = False) -> None:
        raise self._not_served("load_checkpoint", "no checkpoint reader")


def _per_request(batch: Batch, outs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
    """Rank outputs, in rank order -> one result per request of the batch."""
    cat = {k: torch.cat([o[k] for o in outs]) for k in ("row_loss", "logprobs", "elementwise", "clamped")}
    losses = row_sums(batch, cat["row_loss"])
    clamped = row_sums(batch, cat["clamped"])
    logprobs = split_rows(batch, cat["logprobs"])
    elementwise = split_rows(batch, cat["elementwise"])
    results = []
    for i in range(len(batch.request_rows)):
        outputs = [{"logprobs": _tensor(lp), "elementwise_loss": _tensor(el)}
                   for lp, el in zip(logprobs[i], elementwise[i])]
        results.append({
            "loss_fn_output_type": batch.loss_fn.value,
            "loss": losses[i],
            "metrics": {"loss:sum": losses[i], "clamped_positions": int(clamped[i])},
            "loss_fn_outputs": outputs,
            "deferred": False,
        })
    return results


def _tensor(values: List[float]) -> Dict[str, Any]:
    return {"data": values, "shape": [len(values)], "dtype": "float32"}