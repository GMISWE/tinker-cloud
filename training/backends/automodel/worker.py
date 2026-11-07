"""
One FSDP2 rank of an Automodel training group (R1): a Ray actor on one GPU that
holds its shard of the base model and, for every tenant of the group, the
tenant's LoRA shard, grads and optimizer state (the adapter store).

Automodel patches one LoRA adapter into the model and FSDP2 shards it, so the
model has one set of LoRA tensors. One tenant is live in them; every other
tenant's shard, grads and optimizer state wait in a pinned-CPU slot. An op of a
non-live tenant swaps it in first: the live tenant's state is copied into its
slot, then the requested tenant's slot is copied into the model's tensors.
Both are `copy_()` on the local shard, no collective. Grads and the optimizer
state move with the tenant, and the LoRA modules take the tenant's dropout
(the shapes are the group's, the dropout is each tenant's), so passes of
several tenants interleave freely and each tenant computes exactly what it
would compute alone. Export and save read a slot without swapping it in; a
load swaps the tenant in and writes.

The driver calls every rank with the same method at once and gathers the
results in rank order. Each call is one Tinker primitive; there is no training
loop here (Automodel's recipe owns its loop, Tinker's client owns this one):
  forward_backward  runs now and returns the real loss; grads accumulate (R9)
  optim_step        writes the request's params into the tenant's family's optimizer, clips, steps
  save_state        gathers the LoRA weights and optimizer state; rank 0 writes
  load_state        copies a checkpoint's weights, and its optimizer state when
                    asked, into every rank's shard
A pass and a step report `swap_seconds`, the time their swap took (0 when
the tenant was live).

Lockstep: FSDP2 runs collectives in every forward and backward, so every rank
must run the same number of them. shard() gives each rank the same row count
and micro-batches are a fixed row count, so the step count is equal; a
micro-batch of padding rows still runs its forward and backward.

Rendezvous takes two calls: rank 0's rendezvous_address() picks a port on its
node, then every rank's setup() joins it and builds the model.
"""
import hashlib
import math
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dion
import ray
import torch
import torch.distributed as dist
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.training.utils import clip_grad_norm
from safetensors.torch import load_file
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor

from ...checkpoints.interchange import ADAPTER_WEIGHTS_FILE, HF_ADAPTER_DIRNAME
from ...models.requests import AdamParams, OptimParams
from ...utils.model_config import read_raw_hf_config
from .adapter import (OPTIMIZER_DIRNAME, SHARD_KEYS, STATE_KEYS, from_peft, peft_key, read_optimizer_state,
                      write_optimizer_state, write_peft_adapter)
from .converter import Batch
from .errors import NonFiniteGradNorm, StateMismatch
from .losses import compute_loss
from .lora import LoraSpec
from .peft import build_overrides, lora_in_automodel, peft_config

OptimState = Dict[str, torch.Tensor]   # one parameter's state in the tenant's family (adapter.STATE_KEYS)


@dataclass
class _Slot:
    """One tenant's state on this rank while it is not live: the local shards
    in pinned host memory. A grad or optimizer entry is None when the tenant
    has none (no pass since its last step; no step yet). The optimizer state
    is the tenant's family's (its LoraSpec.optimizer)."""
    weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    grads: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    optim: Dict[str, Optional[OptimState]] = field(default_factory=dict)


class AutomodelWorker:
    """Wrap with ray.remote(num_gpus=1, num_cpus=1) and place in the group's PG."""

    def __init__(self, rank: int, world_size: int, base_model: str, lora: LoraSpec, micro_batch_rows: int):
        self.rank = rank
        self.world_size = world_size
        self.base_model = base_model
        self.lora = lora
        self.micro_batch_rows = micro_batch_rows
        self.live: Optional[str] = None       # the tenant in the model's LoRA tensors
        self.slots: Dict[str, _Slot] = {}     # every other tenant
        self.tenant_lora: Dict[str, LoraSpec] = {}   # each tenant's spec: the group's signature, its own dropout

    def rendezvous_address(self) -> str:
        assert self.rank == 0, "rank 0 hosts the rendezvous"
        host = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
        return f"tcp://{host}:{port}"

    def setup(self, init_method: str) -> int:
        """Join the group, build the model with LoRA and the optimizer; no
        tenant yet. Returns the number of trainable (LoRA) elements in the
        full model."""
        # Ray exposes the bundle's one GPU as device 0
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", init_method=init_method, rank=self.rank,
                                world_size=self.world_size, device_id=device)
        distributed_setup = DistributedSetup.build(
            "fsdp2", ParallelismSizes(dp_size=self.world_size), world_size=self.world_size)
        with lora_in_automodel(self.lora):
            self.model = NeMoAutoModelForCausalLM.from_pretrained(
                self.base_model, distributed_setup=distributed_setup, peft_config=peft_config(self.lora),
                torch_dtype=torch.bfloat16, use_liger_kernel=False,
                **build_overrides(read_raw_hf_config(self.base_model)))
        self.lora_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        _check_targets(self.lora_params, self.lora.target_modules)
        self.lora_modules = [m for m in self.model.modules() if isinstance(m, LinearLoRA)]
        assert self.lora_modules, "the model has LoRA parameters but no LoRA module"
        params = [p for _, p in self.lora_params]
        # One optimizer object per family over the same parameters: a second
        # training group would take as many more whole GPUs. A tenant's family
        # (LoraSpec.optimizer) picks one and its slot holds that family's state.
        # lr and the per-step fields are rewritten from each request before
        # every step; the constructors' values are never used. dion refuses
        # DTensor parameters without their own mesh ("Must create optimizer
        # with DeviceMesh"). Its group-level `step` counter is not
        # checkpointed: only dion's element-wise adamw update reads it, and
        # every parameter here takes the Muon update.
        mesh = params[0].device_mesh if isinstance(params[0], DTensor) else None   # dion takes no mesh for a plain tensor
        self.optimizers: Dict[str, torch.optim.Optimizer] = {
            "adamw": torch.optim.AdamW(params),
            "dimuon": dion.Muon(params, distributed_mesh=mesh, weight_decay=0.0, nesterov=False,
                                adjust_lr="spectral_norm"),
        }
        self._prime_muon(params)
        return sum(p.numel() for _, p in self.lora_params)

    def _prime_muon(self, params: List[torch.nn.Parameter]) -> None:
        """One Muon step on zero gradients, its state discarded; the weights stay
        unchanged (orth(0) = 0). dion's update functions are torch.compile'd and
        meet several shapes in one step (A factors, transposed B factors);
        priming compiles them for those shapes before any tenant's first step,
        so every process, including one that resumes a run, has the same compile
        history. test_dimuon_checkpoint_round_trip_and_family_mismatch in
        tests/gpu/test_automodel_worker.py pins the result: a fresh group's
        resumed step is bit-equal to the uninterrupted one."""
        muon = self.optimizers["dimuon"]
        for p in params:
            p.grad = torch.zeros_like(p)
        muon.step()
        muon.zero_grad(set_to_none=True)
        muon.state.clear()   # every parameter had a grad, so every one has state to drop

    # --- tenants ---------------------------------------------------------
    def add_tenant(self, tenant: str, lora: LoraSpec, seed: int) -> None:
        """Make `tenant` live with a freshly initialised adapter. `lora` has
        the group's signature and the tenant's dropout. The tenant that was
        live goes to its slot first, so a failure here leaves every other
        tenant intact."""
        if tenant == self.live or tenant in self.slots:
            raise ValueError(f"tenant {tenant} is already in this group")
        assert lora.signature == self.lora.signature, (lora, self.lora)
        self._stash_live()
        self.live = None
        _init_lora(self.lora_params, seed)
        for _, p in self.lora_params:
            p.grad = None
            self._drop_state(p)
        self.tenant_lora[tenant] = lora
        self._set_dropout(lora.dropout)
        self.live = tenant

    def remove_tenant(self, tenant: str) -> None:
        """Drop the tenant's state. A live tenant leaves the model's tensors
        holding nothing: the next op swaps its tenant in."""
        if tenant == self.live:
            self.live = None
        elif tenant in self.slots:
            del self.slots[tenant]
        else:
            raise ValueError(f"tenant {tenant} is not in this group")
        del self.tenant_lora[tenant]

    def _set_dropout(self, p: float) -> None:
        for m in self.lora_modules:
            m.dropout_p = p

    def _swap_to(self, tenant: str) -> float:
        """Make `tenant` live; returns the seconds it took (0 when it was)."""
        if tenant == self.live:
            return 0.0
        if tenant not in self.slots:
            raise ValueError(f"tenant {tenant} is not in this group")
        torch.cuda.synchronize()
        started = time.perf_counter()
        self._stash_live()
        slot = self.slots.pop(tenant)
        family = self.tenant_lora[tenant].optimizer
        state = self.optimizers[family].state
        with torch.no_grad():
            for n, p in self.lora_params:
                _local(p).copy_(slot.weights[n], non_blocking=True)
                grad = slot.grads[n]
                if grad is None:
                    p.grad = None
                else:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    _local(p.grad).copy_(grad, non_blocking=True)
                for other, optimizer in self.optimizers.items():   # the other family's buffers are a previous tenant's
                    if other != family:
                        optimizer.state.pop(p, None)
                st = slot.optim[n]
                if st is None:
                    state.pop(p, None)
                elif p in state:   # the previous tenant's buffers, same family and shapes: reuse them
                    for k in SHARD_KEYS[family]:
                        _local(state[p][k]).copy_(st[k], non_blocking=True)
                    for k in _scalar_keys(family):
                        state[p][k].copy_(st[k])
                else:
                    state[p] = _state_on_device(p, family, st)
        torch.cuda.synchronize()
        self._set_dropout(self.tenant_lora[tenant].dropout)
        self.live = tenant
        return time.perf_counter() - started

    def _stash_live(self) -> None:
        """Copy the live tenant's shards, grads and optimizer state into its slot."""
        if self.live is None:
            return
        slot = _Slot()
        family = self.tenant_lora[self.live].optimizer
        state = self.optimizers[family].state
        with torch.no_grad():
            for n, p in self.lora_params:
                slot.weights[n] = _pinned(_local(p))
                slot.grads[n] = None if p.grad is None else _pinned(_local(p.grad))
                slot.optim[n] = None if p not in state else _state_to_host(family, state[p])
        torch.cuda.synchronize()
        self.slots[self.live] = slot
        self.live = None

    def _require_tenant(self, tenant: str) -> None:
        if tenant != self.live and tenant not in self.slots:
            raise ValueError(f"tenant {tenant} is not in this group")

    def _drop_state(self, p: torch.nn.Parameter) -> None:
        """Forget every family's state for `p` (a tenant leaves the tensors)."""
        for optimizer in self.optimizers.values():
            optimizer.state.pop(p, None)

    # --- passes and steps ---------------------------------------------------
    def forward_backward(self, tenant: str, shard: Batch, clip: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        swap = self._swap_to(tenant)
        self.model.train()
        if isinstance(self.model, FSDPModule):  # Automodel leaves a 1-rank model unwrapped
            # optim_step is a later call, so no micro-batch is known to be the
            # last: every backward reduce-scatters into the sharded grads
            self.model.set_requires_gradient_sync(True)
        out = self._run(shard, clip, backward=True)
        out["swap_seconds"] = swap
        return out

    def forward(self, tenant: str, shard: Batch, clip: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        swap = self._swap_to(tenant)
        self.model.eval()
        with torch.no_grad():
            out = self._run(shard, clip, backward=False)
        _reshard(self.model)
        out["swap_seconds"] = swap
        return out

    def optim_step(self, tenant: str, params: OptimParams) -> Dict[str, float]:
        """Raises NonFiniteGradNorm. The norm is all-reduced, so every rank
        takes the same branch: all step or all refuse. `params` is the tenant's
        own family (the service checked)."""
        family = self.tenant_lora[tenant].optimizer
        assert params.type == family, (tenant, family, params.type)   # an invariant, not validation: the API refused mismatches
        swap = self._swap_to(tenant)
        optimizer = self.optimizers[family]
        for group in optimizer.param_groups:
            group["lr"] = params.learning_rate
            if isinstance(params, AdamParams):
                group["betas"] = (params.beta1, params.beta2)
                group["eps"] = params.eps
                group["weight_decay"] = params.weight_decay
            else:
                group["mu"] = params.beta1   # dion's momentum coefficient; weight decay stays 0 (constructor)
        # max_norm inf computes the norm and scales by 1: a nonpositive grad_clip_norm means no clipping
        max_norm = params.grad_clip_norm if params.grad_clip_norm > 0 else math.inf
        norm = float(clip_grad_norm(max_norm, [self.model]))
        if not math.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise NonFiniteGradNorm(f"grad norm is {norm}; the optimizer step was not taken and the grads were zeroed")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {"grad_norm": norm, "swap_seconds": swap}

    # --- state in and out ---------------------------------------------------
    def lora_state(self, tenant: str) -> Dict[str, torch.Tensor]:
        """The tenant's full fp32 LoRA weights on CPU, on every rank, from the
        model when it is live and from its slot otherwise. Gathers the
        shards: call it on all ranks at once."""
        self._require_tenant(tenant)
        if tenant == self.live:
            return {n: _full(p).float() for n, p in self.lora_params}
        slot = self.slots[tenant]
        return {n: _full(_on_device(p, slot.weights[n])).float() for n, p in self.lora_params}

    def _optimizer_state(self, tenant: str) -> Tuple[str, Dict[str, Dict[str, torch.Tensor]]]:
        """The tenant's family and its full optimizer state per LoRA parameter
        on CPU, keyed as the adapter is; empty until its first step. Both
        optimizers create a parameter's state at its first step with a grad
        (`torch/optim/adam.py`, and `Muon._create_muon_tasks` in dion's `muon.py`
        at 7452a58, the commit nemo_automodel's lock pins; each filters on
        `p.grad is not None`), so the state is all or none here, every LoRA
        parameter being in the forward path; a partial state is a bug and is
        refused, never written."""
        self._require_tenant(tenant)
        family = self.tenant_lora[tenant].optimizer
        keys, shards = STATE_KEYS[family], SHARD_KEYS[family]
        per_param: list   # (name, param, that param's state as local shards, or None)
        if tenant == self.live:
            state = self.optimizers[family].state
            per_param = [(n, p, {k: _local(state[p][k]) for k in keys} if p in state else None)
                         for n, p in self.lora_params]
        else:
            slot = self.slots[tenant]
            per_param = [(n, p, slot.optim[n]) for n, p in self.lora_params]
        stepped = [st is not None for _, _, st in per_param]
        if any(stepped) and not all(stepped):
            raise RuntimeError(f"{family} holds state for {sum(stepped)} of {len(stepped)} LoRA parameters; "
                               "a checkpoint of it would be partial")
        return family, {peft_key(n): {k: _full(_on_device(p, st[k]) if k in shards else st[k]) for k in keys}
                        for n, p, st in per_param if st is not None}

    def save_state(self, tenant: str, root: str) -> None:
        """Rank 0 writes the tenant's adapter (`hf_adapter/`) and the optimizer
        state of its LoRA parameters (`optimizer/`, its family's file) under
        `root` (adapter.py), without swapping the tenant in. Both gathers are
        collective: call it on all ranks at once."""
        lora = self.lora_state(tenant)
        family, state = self._optimizer_state(tenant)
        if self.rank == 0:
            write_peft_adapter(lora, self.tenant_lora[tenant], self.base_model, Path(root) / HF_ADAPTER_DIRNAME)
            write_optimizer_state(family, state, Path(root) / OPTIMIZER_DIRNAME)

    def load_state(self, tenant: str, adapter_dir: str, optimizer_dir: Optional[str]) -> None:
        """Copy the adapter's weights into the tenant's shards and, given
        optimizer_dir, the optimizer state; without it the optimizer keeps what
        it has (fresh on a tenant that has not trained). The files' keys must
        be exactly this model's LoRA parameters and the state file the tenant's
        family's: StateMismatch before anything is copied.
        Collective: call it on all ranks at once."""
        self._require_tenant(tenant)
        family = self.tenant_lora[tenant].optimizer
        try:
            weights = from_peft(load_file(str(Path(adapter_dir) / ADAPTER_WEIGHTS_FILE)),
                                [n for n, _ in self.lora_params], self.tenant_lora[tenant])
        except ValueError as e:
            raise StateMismatch(f"adapter: {e}") from e
        saved = None if optimizer_dir is None else read_optimizer_state(Path(optimizer_dir))
        if saved is not None:
            saved_family, state = saved
            if saved_family != family:
                raise StateMismatch(f"the checkpoint's optimizer state is {saved_family}'s but this model's optimizer is {family}")
            if state:
                self._require_keys(state, "optimizer state")
        self._swap_to(tenant)
        with torch.no_grad():
            for n, p in self.lora_params:
                _copy_full(p, weights[peft_key(n)].float())
        if saved is None:   # a weights-only load: the optimizer keeps what it has (a request shape, not a gap)
            return
        optimizer = self.optimizers[family]
        for n, p in self.lora_params:
            if not state:   # saved before the first step: no state to restore
                optimizer.state.pop(p, None)
                continue
            st = state[peft_key(n)]
            optimizer.state[p] = {**{k: _like(p, st[k]) for k in SHARD_KEYS[family]},
                                  **{k: st[k].clone() for k in _scalar_keys(family)}}

    def _require_keys(self, have: Dict[str, Any], what: str) -> None:
        want = {peft_key(n) for n, _ in self.lora_params}
        missing, unknown = sorted(want - set(have)), sorted(set(have) - want)
        if missing or unknown:
            raise StateMismatch(f"{what} keys are not this model's LoRA parameters: {len(missing)} missing "
                                f"(first: {missing[:1]}), {len(unknown)} unknown (first: {unknown[:1]})")

    def export_adapter(self, tenant: str, dest: str) -> None:
        """Write the tenant's HF PEFT adapter at `dest` (adapter.py) without
        swapping it in. The gather is collective: call it on all ranks at
        once; rank 0 writes."""
        state = self.lora_state(tenant)
        if self.rank == 0:
            write_peft_adapter(state, self.tenant_lora[tenant], self.base_model, Path(dest))

    def _run(self, shard: Batch, clip: Optional[Tuple[float, float]], backward: bool) -> Dict[str, Any]:
        rows, width = shard.input_ids.shape
        out: Dict[str, Any] = {
            "row_loss": torch.zeros(rows),
            "logprobs": torch.zeros(rows, width),
            "elementwise": torch.zeros(rows, width),
            "clamped": torch.zeros(rows),
            "seq_log_ratio": torch.zeros(rows),
            "seq_clipped": torch.zeros(rows),
        }
        for start in range(0, rows, self.micro_batch_rows):
            end = min(start + self.micro_batch_rows, rows)
            mb = _micro_batch(shard, start, end)
            # right padding under a causal mask: a real position never attends
            # to padding, so no attention mask is passed
            logits = self.model(input_ids=mb.input_ids.cuda()).logits
            res = compute_loss(logits, mb, clip)
            if backward:
                # FSDP2 averages grads over ranks; this restores the sum (losses.py)
                (res.loss * self.world_size).backward()
            w = min(mb.input_ids.shape[1], width)   # a micro-batch padded past the shard's width
            out["row_loss"][start:end] = res.row_loss.cpu()
            out["logprobs"][start:end, :w] = res.logprobs[:, :w].cpu()
            out["elementwise"][start:end, :w] = res.elementwise[:, :w].cpu()
            out["clamped"][start:end] = res.clamped.cpu()
            out["seq_log_ratio"][start:end] = res.seq_log_ratio.cpu()
            out["seq_clipped"][start:end] = res.seq_clipped.cpu()
        return out


def _reshard(model: torch.nn.Module) -> None:
    """Free the unsharded parameter copies a forward without a backward
    leaves in FSDP2 units. While they exist, `model.parameters()` yields
    them instead of the sharded parameters: the next optimizer step's grad
    norm skips them (they carry no grad), and the next forward of another
    tenant would read them, which hold the weights of the tenant that ran.
    A backward reshards on its own. Pinned by the GPU test
    `test_tenant_eval_forward_in_between_leaves_the_step_and_the_scores_solo`."""
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


# Transformer Engine's fused attention (Qwen3.5's full-attention layers) refuses a backward
# at sequence length 1 ("cuDNN Error: s_q = s_kv = 1 is not supported"); one rank raising
# there leaves its peers in a collective. Columns past a row's length are masked out of the loss.
MIN_MICRO_BATCH_WIDTH = 2


def _micro_batch(shard: Batch, start: int, end: int) -> Batch:
    """Rows [start, end), cut to their longest row but at least
    MIN_MICRO_BATCH_WIDTH columns (an all-padding micro-batch still runs), with
    zero columns added when the shard is narrower."""
    width = max(MIN_MICRO_BATCH_WIDTH, int(shard.lengths[start:end].max()))
    pad = max(0, width - shard.input_ids.shape[1])

    def cut(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(t[start:end, :width], (0, pad))

    return Batch(
        loss_fn=shard.loss_fn,
        input_ids=cut(shard.input_ids),
        lengths=shard.lengths[start:end],
        inputs={k: cut(v) for k, v in shard.inputs.items()},
        request_rows=(),
    )


def _check_targets(params, targets: Tuple[str, ...]) -> None:
    for name, _ in params:
        if ".lora_A." not in name and ".lora_B." not in name:
            raise RuntimeError(f"trainable parameter {name} is not a LoRA weight")
    missing = [m for m in targets if not any(f".{m}.lora_A." in n for n, _ in params)]
    if missing:
        raise ValueError(f"LoRA targets {', '.join(missing)} match no module of this model")


def _init_lora(params, seed: int) -> None:
    """A = kaiming_uniform(a=sqrt(5)) (PEFT's init), B = 0. Each full matrix
    comes from a generator seeded by (seed, parameter name), and each rank
    keeps its slice, so the adapter does not depend on the rank count (R2.6).
    Every rank walks the parameters in the same order (distribute_tensor may
    communicate)."""
    for name, p in params:
        full = torch.empty(p.shape, dtype=torch.float32)
        if ".lora_A." in name:
            digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
            gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)
            torch.nn.init.kaiming_uniform_(full, a=math.sqrt(5), generator=gen)
        else:
            full.zero_()
        with torch.no_grad():
            _copy_full(p, full)


def _local(t: torch.Tensor) -> torch.Tensor:
    """This rank's shard of a DTensor (a view of its storage), else `t`."""
    return t.to_local() if isinstance(t, DTensor) else t


def _pinned(local: torch.Tensor) -> torch.Tensor:
    """A copy of `local` in pinned host memory."""
    buf = torch.empty(local.shape, dtype=local.dtype, device="cpu", pin_memory=True)
    buf.copy_(local, non_blocking=True)
    return buf


def _on_device(p: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """`local`, this rank's shard, laid out as `p` is: a DTensor on p's mesh
    (whose full_tensor() is collective), else a tensor on p's device."""
    if isinstance(p, DTensor):
        return DTensor.from_local(local.to(p.to_local().device, non_blocking=True), p.device_mesh, p.placements)
    return local.to(p.device)


def _full(t: torch.Tensor) -> torch.Tensor:
    """The whole tensor on CPU; collective for a DTensor."""
    return (t.full_tensor() if isinstance(t, DTensor) else t).detach().cpu()


def _like(p: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
    """`full` laid out as `p` is: this rank's shard of a DTensor, else on p's device."""
    if isinstance(p, DTensor):
        return distribute_tensor(full.to(p.to_local().device), p.device_mesh, p.placements)
    return full.to(p.device)


def _scalar_keys(family: str) -> Tuple[str, ...]:
    """The family's state keys that are not laid out as the parameter (AdamW's `step`)."""
    return tuple(k for k in STATE_KEYS[family] if k not in SHARD_KEYS[family])


def _state_to_host(family: str, state: OptimState) -> OptimState:
    """One parameter's optimizer state as a slot holds it: shards pinned on the host, scalars cloned."""
    return {**{k: _pinned(_local(state[k])) for k in SHARD_KEYS[family]},
            **{k: state[k].clone() for k in _scalar_keys(family)}}


def _state_on_device(p: torch.Tensor, family: str, slot_state: OptimState) -> OptimState:
    """The inverse of _state_to_host: shards laid out as `p` is, scalars cloned."""
    return {**{k: _on_device(p, slot_state[k]) for k in SHARD_KEYS[family]},
            **{k: slot_state[k].clone() for k in _scalar_keys(family)}}


def _copy_full(p: torch.Tensor, full: torch.Tensor) -> None:
    """Copy `full` into `p` in place; a DTensor takes its own slice."""
    if isinstance(p, DTensor):
        local = _like(p, full)
        assert isinstance(local, DTensor)
        p.to_local().copy_(local.to_local())
    else:
        p.copy_(full)
