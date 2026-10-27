"""
The Automodel backend: one FSDP2 training group per (base model, LoRA
signature), shared by every LoRA tenant with that signature, one vLLM per base
model, and every Tinker primitive run when it is called.

A model's requests reach its group through the group's scheduler: a pass
coalesces with the consecutive same-key passes behind it (R15); a barrier
(optimizer step, publish, join, leave, delete) runs alone. The ranks keep one
adapter per tenant and swap the op's tenant in (worker.py); a pass or step
reports the swap's seconds as `swap_seconds`. One pass is: convert each request
(validated alone, so a bad request fails alone) -> coalesce into one batch
padded to a multiple of the rank count -> one shard per rank -> every rank
runs it -> the rows come back in rank order and are split per request (R3).

Sampling (R4, R14, D20): vLLM never receives base weights after boot. The
adapter at weight version k is exported in HF PEFT format under the model's
native root and loaded into the base model's vLLM as `<model_id>@v<k>`, after
each optimizer step and before the first sample. A sample is served by the
version published when it was submitted and reports it; an older version is
unloaded once no sample of it is in flight. The export is a barrier on the
group (no pass may change the weights during the gather).

Checkpoints: a save writes the interchange adapter and the
LoRA parameters' AdamW state under the root the store hands over, as a barrier
on the group; an ephemeral sampler save writes nothing (vLLM holds the version
a sampler pins). A load checks the adapter config against the model's LoRA,
copies the adapter into the ranks, restores the optimizer state only when
asked and only whole, and publishes the loaded weights as the next adapter
version, so no version name is reused.

Faults: a non-finite grad norm fails that optimizer step only; the grads were
zeroed on every rank and the group goes on (R10). Any other failure inside a
rank call is the group's: its actors and bundles are released, the queued
requests fail, and every later call on its models names the cause (R5). A
vLLM that stops answering fails the samples and publishes that reach it; the
sampler is not torn down for its tenants (recovery is delete and create).
"""
import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import torch

from ...checkpoints.interchange import ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, HF_ADAPTER_DIRNAME, find_hf_adapter, read_adapter_config
from ...core.loss_registry import clip_thresholds
from ...models.requests import AdamParams, Datum, LoraConfig
from ...utils.model_config import model_context_length, read_raw_hf_config
from ..base import BackendError, BackendHandle, TrainingBackend, UnsupportedFeatureError
from ..common.sampling.generation import sample_over_http
from ..common.sampling.vllm_client import VllmLoraClient
from ..http_pool import HttpClientPool
from ..objectives import classification_spec
from .adapter import OPTIMIZER_DIRNAME, OPTIMIZER_FILE
from .config import AutomodelConfig
from .converter import Batch, LossFn, RequestRows, coalesce, convert_request, row_sums, shard, split_rows
from .errors import NonFiniteGradNorm, StateMismatch
from .group import TrainingGroup, sweep_leftovers
from .lora import LoraSpec, lora_spec, spec_mismatch, tenant_seed
from .sampler import MAX_LOGPROBS, SamplerGroup
from .scheduler import GroupScheduler, PassKey, pass_key

logger = logging.getLogger(__name__)

BACKEND = "automodel"
ADAPTERS_DIRNAME = "adapters"


@dataclass
class AutomodelHandle(BackendHandle):
    backend_type: str = BACKEND
    base_model: str = ""
    lora: Optional[LoraSpec] = None
    group_key: str = ""
    sampler_key: str = ""
    native_root: Optional[Path] = None       # the adapter exports live under it
    tokenizer: Any = None                    # decodes sampled tokens to text
    next_leader: int = 0                     # the shared client's cursor; one URL here
    generation_synced_version: int = -1      # adapter version vLLM holds; -1 = none published yet
    loaded_versions: List[int] = field(default_factory=list)   # versions vLLM holds, oldest first
    in_flight: Dict[int, int] = field(default_factory=dict)    # samples running, per version


@dataclass
class _Group:
    key: str
    ranks: TrainingGroup          # or a stand-in with the same nine coroutines
    scheduler: GroupScheduler
    tenants: Set[str] = field(default_factory=set)
    lost: Optional[str] = None    # why; set before the teardown runs


@dataclass
class _Sampler:
    key: str
    host: SamplerGroup            # or a stand-in with stop()
    url: str
    tenants: Set[str] = field(default_factory=set)


def group_key(base_model: str, lora: LoraSpec) -> str:
    """Groups are keyed by base model and LoRA signature (R2.4). The key is
    also the actor and placement-group name, so it is short and plain."""
    return hashlib.sha256(f"{base_model}|{lora.signature}".encode()).hexdigest()[:12]


def sampler_key(base_model: str) -> str:
    """One vLLM per base model (Q5); the key names its actor and placement group."""
    return hashlib.sha256(base_model.encode()).hexdigest()[:12]


def adapter_name(model_id: str, version: int) -> str:
    """The name vLLM knows one version of a tenant's adapter by (D20)."""
    return f"{model_id}@v{version}"


def adapter_dir(handle: AutomodelHandle, version: int) -> Path:
    assert handle.native_root is not None
    return handle.native_root / ADAPTERS_DIRNAME / f"v{version}"


class AutomodelBackend(TrainingBackend[AutomodelHandle]):
    needs_ray = True
    config: AutomodelConfig
    SUPPORTED_LOSS_FNS = frozenset(m.value for m in LossFn)   # the service's table is keyed by the wire string

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        super().__init__(overrides)
        self.config = AutomodelConfig.from_env(overrides)
        self._groups: Dict[str, _Group] = {}
        self._lost: Dict[str, str] = {}      # group key -> why it was torn down
        self._samplers: Dict[str, _Sampler] = {}
        self._sampler_boots: Dict[str, asyncio.Lock] = {}   # one vLLM boot per base model at a time
        self._membership: Dict[str, asyncio.Lock] = {}      # per group key: who opens, joins, leaves, releases
        self._tasks: Set[asyncio.Task] = set()
        self._swept = False
        # One connection pool for every vLLM this process talks to. No read
        # timeout: a generation is bounded by its max_tokens on the engine side.
        self._http_pool = HttpClientPool(
            keepalive_expiry=2.0, timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None))

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
        if native_root is None:
            raise BackendError("create_model needs native_root: the adapter exports vLLM loads live there",
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
        context_length = model_context_length(raw)

        if not self._swept:   # before the first group, so a crashed predecessor's leftovers do not hold its GPUs
            found = sweep_leftovers()
            if found:
                logger.warning("swept %d leftover training-group actors/placement groups", found)
            self._swept = True

        key = group_key(base_model, lora)
        seed = tenant_seed(model_id, config.seed)
        async with self._membership_lock(key):   # one create at a time per key: the check and the open or join are one step
            if key in self._groups and self._groups[key].lost is None:   # the signature's group is up: join it
                group = self._groups[key]
                await group.scheduler.submit_barrier(model_id, lambda: self._join(group, model_id, lora, seed))
                logger.info("[%s] model %s joined training group %s (%d tenants)", request_id, model_id, key, len(group.tenants))
            else:
                group = await self._open_group(key, base_model, lora, model_id, seed, request_id)

        try:
            from transformers import AutoTokenizer   # the training env's transformers
            tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, base_model)
            sampler = await self._sampler_for(base_model, context_length, request_id)
        except BaseException:   # a failed create cleans up what it started (R5)
            await self._drop_tenant(key, model_id, "create_model", f"create_model of {model_id} failed")
            raise
        sampler.tenants.add(model_id)
        return AutomodelHandle(
            model_id=model_id, context_length=context_length, max_topk_logprobs=MAX_LOGPROBS,
            hf_path=base_model, inference_endpoints=(sampler.url,),
            base_model=base_model, lora=lora, group_key=key, sampler_key=sampler.key,
            native_root=native_root, tokenizer=tokenizer)

    def _membership_lock(self, key: str) -> asyncio.Lock:
        """The lock a create holds to check for the key's group and open or
        join it, and a delete holds to leave or release it: two creates of a
        new signature open one group, and two deletes of a group's last two
        tenants leave, then release."""
        if key not in self._membership:
            self._membership[key] = asyncio.Lock()
        return self._membership[key]

    async def _open_group(self, key: str, base_model: str, lora: LoraSpec, model_id: str, seed: int,
                          request_id: str) -> _Group:
        """A new training group with `model_id` as its first tenant. Runs
        under the key's membership lock."""
        ranks = TrainingGroup(key, self.config.train_gpus)
        logger.info("[%s] starting training group %s: %s, %d ranks, LoRA %s",
                    request_id, key, base_model, self.config.train_gpus, lora.signature)
        lora_numel = await ranks.start(base_model, lora, self.config.micro_batch_rows)
        try:
            await ranks.add_tenant(model_id, lora, seed)
        except BaseException:
            await ranks.stop()
            raise
        scheduler = GroupScheduler(key, lambda pk, payloads: self._run_pass(key, pk, payloads))
        group = _Group(key, ranks, scheduler, {model_id})
        self._groups[key] = group
        if key in self._lost:
            del self._lost[key]
        logger.info("[%s] training group %s up: LoRA %d elements", request_id, key, lora_numel)
        return group

    async def _join(self, group: _Group, model_id: str, lora: LoraSpec, seed: int) -> None:
        """Runs inside the group's scheduler. The ranks stash the live tenant
        before they touch the adapter, so a refused join leaves the group as
        it was."""
        try:
            await group.ranks.add_tenant(model_id, lora, seed)
        except Exception as e:
            raise BackendError(f"model {model_id} could not join training group {group.key}: {e}",
                               backend=BACKEND, operation="create_model", original_error=e) from e
        group.tenants.add(model_id)

    async def _drop_tenant(self, key: str, model_id: str, op: str, reason: str) -> None:
        """Free what the key's group holds for the tenant: its slot when other
        tenants stay, the ranks and the placement group (closed with `reason`)
        when it was the last. A group torn down after a fault, or a tenant
        already gone from it, leaves nothing to free."""
        async with self._membership_lock(key):
            if key not in self._groups:
                return
            group = self._groups[key]
            if group.lost is not None or model_id not in group.tenants:
                return
            if group.tenants - {model_id}:
                await self._leave(group, model_id, op)
            else:
                group.tenants.discard(model_id)
                await self._release(group, reason)

    async def _leave(self, group: _Group, model_id: str, op: str) -> None:
        """Free the tenant's state on every rank, as a barrier; the group stays
        up for its other tenants. A rank fault here is the group's."""
        group.tenants.discard(model_id)
        ran = False

        async def leave() -> None:
            nonlocal ran
            ran = True
            try:
                await group.ranks.remove_tenant(model_id)
            except Exception as e:
                self._mark_lost(group, f"remove_tenant failed on a rank: {e!r}")
                raise BackendError(f"{op} failed; the training group was released",
                                   backend=BACKEND, operation=op, original_error=e) from e

        try:
            await group.scheduler.submit_barrier(model_id, leave)
        except BackendError:
            if ran or group.lost is None:   # the leave's own fault, or a closed scheduler: the caller hears it
                raise
            # the group was lost while the leave waited in its queue: the slot went with the ranks

    async def _sampler_for(self, base_model: str, max_model_len: int, request_id: str) -> _Sampler:
        """The base model's sampler, booted on first use."""
        key = sampler_key(base_model)
        if key not in self._sampler_boots:
            self._sampler_boots[key] = asyncio.Lock()
        async with self._sampler_boots[key]:
            if key in self._samplers:
                return self._samplers[key]
            host = SamplerGroup(key)
            logger.info("[%s] starting vLLM for %s (sampler %s)", request_id, base_model, key)
            url = await host.start(base_model, max_model_len, self.config)
            self._samplers[key] = _Sampler(key, host, url)
            logger.info("[%s] vLLM for %s up at %s", request_id, base_model, url)
            return self._samplers[key]

    async def delete_model(self, handle: AutomodelHandle) -> None:
        await self._drop_tenant(handle.group_key, handle.model_id, "delete_model", f"model {handle.model_id} was deleted")
        if handle.sampler_key in self._samplers:
            sampler = self._samplers[handle.sampler_key]
            sampler.tenants.discard(handle.model_id)
            if sampler.tenants:
                await self._unload_all(handle, sampler)
            else:   # last tenant of the base model: stop its vLLM (R5)
                del self._samplers[sampler.key]
                await sampler.host.stop()
        if handle.native_root is not None:
            await asyncio.to_thread(shutil.rmtree, handle.native_root / ADAPTERS_DIRNAME, ignore_errors=True)

    async def _release(self, group: _Group, reason: str) -> None:
        await group.scheduler.close(reason)
        await group.ranks.stop()
        if group.key in self._groups and self._groups[group.key] is group:
            del self._groups[group.key]

    async def close(self) -> None:
        for group in list(self._groups.values()):
            await self._release(group, "server shutdown")
        for sampler in list(self._samplers.values()):
            await sampler.host.stop()
        self._samplers.clear()
        await self._http_pool.aclose()

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
                outs = await group.ranks.optim_step(handle.model_id, adam)
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
            swap = _swap_seconds(outs)
            handle.weight_version += 1
            # R4 (iii): the new version goes to vLLM before the step is answered
            try:
                await self._publish(group, handle)
            except BackendError as e:
                raise BackendError(
                    f"the optimizer step was taken (weights at v{handle.weight_version}) but the adapter "
                    f"could not be published to vLLM: {e}",
                    backend=BACKEND, operation="apply_optimizer_step") from e
            await self._retire(handle)
            # step metrics are not reduced (the SDK reads `metrics` only): a bare key, like grad_norm
            return {"grad_norm": norm, "weight_version": handle.weight_version, "swap_seconds": swap,
                    "metrics": {"swap_seconds": swap}}

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
            outs = await run(pk.model_id, shards, clip)
        except Exception as e:
            self._mark_lost(group, f"{pk.kind} failed on a rank: {e!r}")
            raise BackendError(f"{pk.kind} failed; the training group was released",
                               backend=BACKEND, operation=pk.kind, original_error=e) from e
        return _per_request(batch, outs)

    # --- sampling --------------------------------------------------------
    async def sample(
        self, handle: AutomodelHandle, request_id: str, prompt_tokens: List[int], num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None, prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None, topk_sample_logprobs: int = 0, topk_prompt_logprobs: int = 0,
    ) -> Dict[str, Any]:
        """One /tinkercloud/v1/generate request to the base model's vLLM, naming
        the adapter version published at submission; the result reports it as
        weight_version and the current weights as latest_weight_version."""
        await self._ensure_published(handle, "sample")
        served = handle.generation_synced_version
        if pinned_version is not None and pinned_version != served:
            # D20: a retired version is not kept for a snapshot sampler; the
            # stamp below tells the client which weights answered.
            logger.warning("[%s] sampler pinned at v%d; serving v%d (reported in weight_version)",
                           request_id, pinned_version, served)
        handle.in_flight[served] = handle.in_flight.get(served, 0) + 1
        try:
            result = await sample_over_http(
                handle, self._http_pool, request_id, prompt_tokens, num_samples,
                sampling_params or {}, prompt_logprobs,
                topk_sample_logprobs=topk_sample_logprobs, topk_prompt_logprobs=topk_prompt_logprobs,
                model=adapter_name(handle.model_id, served))
        finally:
            handle.in_flight[served] -= 1
            await self._retire(handle)
        result["weight_version"] = served
        result["latest_weight_version"] = handle.weight_version
        return result

    async def prepare_for_generation(self, handle: AutomodelHandle) -> None:
        await self._ensure_published(handle, "prepare_for_generation")

    async def update_inference_weights(self, handle: AutomodelHandle) -> None:
        await self._ensure_published(handle, "update_inference_weights")

    async def _ensure_published(self, handle: AutomodelHandle, op: str) -> None:
        """vLLM holds the adapter at the current weight version when this
        returns. The publish is a barrier on the group; a version already
        published costs nothing."""
        group = self._live_group(handle, op)
        if handle.generation_synced_version == handle.weight_version:
            return
        await group.scheduler.submit_barrier(handle.model_id, lambda: self._publish(group, handle))
        await self._retire(handle)

    async def _publish(self, group: _Group, handle: AutomodelHandle) -> None:
        """Export the adapter at the current weight version and load it into
        vLLM under its versioned name. Runs inside the group's scheduler, so no
        pass changes the weights during the gather. A rank fault is the
        group's (R5); a vLLM fault is this publish's only."""
        version = handle.weight_version
        if handle.generation_synced_version == version:
            return
        dest = adapter_dir(handle, version)
        try:
            await group.ranks.export_adapter(handle.model_id, dest)
        except Exception as e:
            self._mark_lost(group, f"export_adapter failed on a rank: {e!r}")
            raise BackendError("export_adapter failed; the training group was released",
                               backend=BACKEND, operation="publish", original_error=e) from e
        await self._lora_client(handle).load_adapter(adapter_name(handle.model_id, version), str(dest))
        handle.loaded_versions.append(version)
        handle.generation_synced_version = version
        logger.info("model %s: adapter v%d published to vLLM", handle.model_id, version)

    async def _retire(self, handle: AutomodelHandle) -> None:
        """Unload every version below the published one that has no sample in
        flight (D20), and remove its export. An unload vLLM refuses is logged
        and retried at the next retire; the name stays listed until then."""
        current = handle.generation_synced_version
        for version in list(handle.loaded_versions):
            if version >= current or handle.in_flight.get(version, 0) > 0:
                continue
            try:
                await self._lora_client(handle).unload_adapter(adapter_name(handle.model_id, version))
            except BackendError as e:
                logger.warning("model %s: adapter v%d stays loaded: %s", handle.model_id, version, e)
                continue
            handle.loaded_versions.remove(version)
            await asyncio.to_thread(shutil.rmtree, adapter_dir(handle, version), ignore_errors=True)

    async def _unload_all(self, handle: AutomodelHandle, sampler: _Sampler) -> None:
        """A deleted tenant leaves a vLLM that stays up for the others; the
        service cancelled its in-flight samples before the delete (S12)."""
        client = VllmLoraClient(sampler.url, self._http_pool.for_url(sampler.url), BACKEND)
        for version in list(handle.loaded_versions):
            try:
                await client.unload_adapter(adapter_name(handle.model_id, version))
            except BackendError as e:
                logger.warning("model %s: adapter v%d not unloaded on delete: %s", handle.model_id, version, e)
            handle.loaded_versions.remove(version)

    def _lora_client(self, handle: AutomodelHandle) -> VllmLoraClient:
        if handle.sampler_key not in self._samplers:
            raise BackendError(f"model {handle.model_id} has no vLLM: its base model's sampler is gone",
                               backend=BACKEND, operation="publish")
        url = self._samplers[handle.sampler_key].url
        return VllmLoraClient(url, self._http_pool.for_url(url), BACKEND)

    # --- faults ----------------------------------------------------------
    def _live_group(self, handle: AutomodelHandle, op: str) -> _Group:
        """The handle's group, if it is up; otherwise the error names why it is not."""
        key = handle.group_key
        if key in self._groups and self._groups[key].lost is None:
            group = self._groups[key]
            if handle.model_id in group.tenants:
                return group
            # a stale handle must not reach the ranks: they would refuse the tenant, and a refusal is a fault
            raise BackendError(f"model {handle.model_id} has no training group: the model was deleted",
                               backend=BACKEND, operation=op)
        why = self._lost[key] if key in self._lost else "the model was deleted"
        raise BackendError(f"model {handle.model_id} has no training group: {why}", backend=BACKEND, operation=op)

    def _mark_lost(self, group: _Group, reason: str) -> None:
        """Runs inside the group's scheduler. The queue is abandoned here, at
        once, so no op behind the faulting one reaches the ranks; close()
        waits for the op in progress, which is this one, so the rest of the
        teardown is a task of its own."""
        if group.lost is not None:
            return
        group.lost = reason
        self._lost[group.key] = reason
        logger.error("training group %s is lost: %s", group.key, reason)
        group.scheduler.abandon(reason)
        task = asyncio.create_task(self._release(group, reason))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- checkpoints ------------------------------------------------------
    async def save_checkpoint(self, handle: AutomodelHandle, root: Path, step: Optional[int] = None,
                              persist: bool = True) -> None:
        """Rank 0 writes the interchange adapter (`hf_adapter/`) and the AdamW
        state of the LoRA parameters (`optimizer/`) under `root`, as a barrier
        on the group. An ephemeral sampler save writes nothing: vLLM holds the
        version the sampler pins."""
        if not persist:
            return
        group = self._live_group(handle, "save_checkpoint")

        async def save() -> None:
            try:
                await group.ranks.save_state(handle.model_id, root)
            except Exception as e:
                self._mark_lost(group, f"save_state failed on a rank: {e!r}")
                raise BackendError("save_checkpoint failed; the training group was released",
                                   backend=BACKEND, operation="save_checkpoint", original_error=e) from e

        await group.scheduler.submit_barrier(handle.model_id, save)
        logger.info("model %s: checkpoint of v%d written under %s", handle.model_id, handle.weight_version, root)

    async def load_checkpoint(self, handle: AutomodelHandle, root: Path, optimizer: bool = False) -> None:
        """Copy the adapter under `root` into the ranks and, when asked, the
        AdamW state: restored whole or refused. The loaded weights are
        published to vLLM as the next version."""
        adapter = find_hf_adapter(str(root))
        if adapter is None:
            raise BackendError(f"checkpoint {root} holds no {HF_ADAPTER_DIRNAME}/{ADAPTER_WEIGHTS_FILE}",
                               backend=BACKEND, operation="load_checkpoint")
        config = read_adapter_config(str(root))
        if config is None:
            raise BackendError(f"checkpoint {root} has no readable {HF_ADAPTER_DIRNAME}/{ADAPTER_CONFIG_FILE}",
                               backend=BACKEND, operation="load_checkpoint")
        assert handle.lora is not None
        mismatch = spec_mismatch(config, handle.lora)
        if mismatch is not None:
            raise BackendError(f"checkpoint {root} was saved for another LoRA ({mismatch})",
                               backend=BACKEND, operation="load_checkpoint")
        optimizer_dir = root / OPTIMIZER_DIRNAME if optimizer else None
        if optimizer_dir is not None and not (optimizer_dir / OPTIMIZER_FILE).is_file():
            raise BackendError(
                f"checkpoint {root} carries no optimizer state ({OPTIMIZER_DIRNAME}/{OPTIMIZER_FILE}): load it with "
                "optimizer=false, or resume from a checkpoint the automodel backend wrote",
                backend=BACKEND, operation="load_checkpoint")
        group = self._live_group(handle, "load_checkpoint")

        async def load() -> None:
            try:
                await group.ranks.load_state(handle.model_id, Path(adapter), optimizer_dir)
            except StateMismatch as e:   # refused on every rank before any copy; the group is intact
                raise BackendError(f"checkpoint {root} does not fit this model: {e}",
                                   backend=BACKEND, operation="load_checkpoint") from e
            except Exception as e:
                self._mark_lost(group, f"load_state failed on a rank: {e!r}")
                raise BackendError("load_checkpoint failed; the training group was released",
                                   backend=BACKEND, operation="load_checkpoint", original_error=e) from e
            handle.weight_version += 1
            try:
                await self._publish(group, handle)
            except BackendError as e:
                raise BackendError(
                    f"the checkpoint was loaded (weights at v{handle.weight_version}) but the adapter could not "
                    f"be published to vLLM: {e}", backend=BACKEND, operation="load_checkpoint") from e

        await group.scheduler.submit_barrier(handle.model_id, load)
        await self._retire(handle)
        logger.info("model %s: loaded %s from %s; the weights are v%d", handle.model_id,
                    "weights and optimizer state" if optimizer else "weights (the optimizer keeps its state)",
                    root, handle.weight_version)


def _per_request(batch: Batch, outs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank outputs, in rank order -> one result per request of the batch.
    The swap served the whole pass, so every request reports it."""
    cat = {k: torch.cat([o[k] for o in outs]) for k in ("row_loss", "logprobs", "elementwise", "clamped")}
    swap = _swap_seconds(outs)
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
            "metrics": {"loss:sum": losses[i], "clamped_positions:sum": int(clamped[i]), "swap_seconds:max": swap},
            "loss_fn_outputs": outputs,
            "deferred": False,
        })
    return results


def _swap_seconds(outs: List[Dict[str, Any]]) -> float:
    """The slowest rank's swap: the op waited for it."""
    return max(float(o["swap_seconds"]) for o in outs)


def _tensor(values: List[float]) -> Dict[str, Any]:
    return {"data": values, "shape": [len(values)], "dtype": "float32"}
