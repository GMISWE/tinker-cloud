"""
One FSDP2 rank of an Automodel training group (R1): a Ray actor on one GPU that
holds its shard of the base model and, for every tenant of the group, the
tenant's LoRA shard, grads and AdamW state (the adapter store).

Automodel patches one LoRA adapter into the model and FSDP2 shards it, so the
model has one set of LoRA tensors. One tenant is live in them; every other
tenant's shard, grads and AdamW state wait in a pinned-CPU slot. An op of a
non-live tenant swaps it in first: the live tenant's state is copied into its
slot, then the requested tenant's slot is copied into the model's tensors.
Both are `copy_()` on the local shard, no collective. Grads and the optimizer
state move with the tenant, so passes of several tenants interleave freely
and each tenant computes exactly what it would compute alone. Export and save
read a slot without swapping it in; a load swaps the tenant in and writes.

The driver calls every rank with the same method at once and gathers the
results in rank order. Each call is one Tinker primitive; there is no training
loop here (Automodel's recipe owns its loop, Tinker's client owns this one):
  forward_backward  runs now and returns the real loss; grads accumulate (R9)
  optim_step        writes the request's AdamParams, clips, steps (R11)
  save_state        gathers the LoRA weights and AdamW state; rank 0 writes
  load_state        copies a checkpoint's weights, and its AdamW state when
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
from typing import Any, Dict, Optional, Tuple

import ray
import torch
import torch.distributed as dist
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.training.utils import clip_grad_norm
from safetensors.torch import load_file
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, distribute_tensor

from ...checkpoints.interchange import ADAPTER_WEIGHTS_FILE, HF_ADAPTER_DIRNAME
from ...models.requests import AdamParams
from .adapter import ADAM_STATE_KEYS, OPTIMIZER_DIRNAME, peft_key, read_adam_state, write_adam_state, write_peft_adapter
from .converter import Batch
from .errors import NonFiniteGradNorm, StateMismatch
from .losses import compute_loss
from .lora import LoraSpec

AdamState = Dict[str, torch.Tensor]   # exp_avg, exp_avg_sq, step


@dataclass
class _Slot:
    """One tenant's state on this rank while it is not live: the local shards
    in pinned host memory. A grad or Adam entry is None when the tenant has
    none (no pass since its last step; no step yet)."""
    weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    grads: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    adam: Dict[str, Optional[AdamState]] = field(default_factory=dict)


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
        peft = PeftConfig(
            target_modules=[f"*.{m}" for m in self.lora.target_modules],
            dim=self.lora.rank,
            alpha=self.lora.alpha,
            dropout=self.lora.dropout,
            # bf16 LoRA weights would round away small AdamW updates
            lora_dtype=torch.float32,
        )
        self.model = NeMoAutoModelForCausalLM.from_pretrained(
            self.base_model, distributed_setup=distributed_setup, peft_config=peft,
            torch_dtype=torch.bfloat16, use_liger_kernel=False)
        self.lora_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        _check_targets(self.lora_params, self.lora.target_modules)
        # lr, betas, eps and weight_decay are rewritten from each request's AdamParams
        # before every step (R11); the constructor's values are never used
        self.optimizer = torch.optim.AdamW([p for _, p in self.lora_params])
        return sum(p.numel() for _, p in self.lora_params)

    # --- tenants ---------------------------------------------------------
    def add_tenant(self, tenant: str, seed: int) -> None:
        """Make `tenant` live with a freshly initialised adapter. The
        tenant that was live goes to its slot first, so a failure here leaves
        every other tenant intact."""
        if tenant == self.live or tenant in self.slots:
            raise ValueError(f"tenant {tenant} is already in this group")
        self._stash_live()
        self.live = None
        _init_lora(self.lora_params, seed)
        for _, p in self.lora_params:
            p.grad = None
            self.optimizer.state.pop(p, None)
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
        state = self.optimizer.state
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
                adam = slot.adam[n]
                if adam is None:
                    state.pop(p, None)
                elif p in state:   # the previous tenant's buffers, same shapes: reuse them
                    for k in ("exp_avg", "exp_avg_sq"):
                        _local(state[p][k]).copy_(adam[k], non_blocking=True)
                    state[p]["step"].copy_(adam["step"])
                else:
                    state[p] = {"exp_avg": _on_device(p, adam["exp_avg"]), "exp_avg_sq": _on_device(p, adam["exp_avg_sq"]),
                                "step": adam["step"].clone()}
        torch.cuda.synchronize()
        self.live = tenant
        return time.perf_counter() - started

    def _stash_live(self) -> None:
        """Copy the live tenant's shards, grads and Adam state into its slot."""
        if self.live is None:
            return
        slot = _Slot()
        state = self.optimizer.state
        with torch.no_grad():
            for n, p in self.lora_params:
                slot.weights[n] = _pinned(_local(p))
                slot.grads[n] = None if p.grad is None else _pinned(_local(p.grad))
                slot.adam[n] = None if p not in state else {
                    "exp_avg": _pinned(_local(state[p]["exp_avg"])), "exp_avg_sq": _pinned(_local(state[p]["exp_avg_sq"])),
                    "step": state[p]["step"].clone()}
        torch.cuda.synchronize()
        self.slots[self.live] = slot
        self.live = None

    def _require_tenant(self, tenant: str) -> None:
        if tenant != self.live and tenant not in self.slots:
            raise ValueError(f"tenant {tenant} is not in this group")

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
        out["swap_seconds"] = swap
        return out

    def optim_step(self, tenant: str, adam: AdamParams) -> Dict[str, float]:
        """Raises NonFiniteGradNorm. The norm is all-reduced, so every rank
        takes the same branch: all step or all refuse."""
        swap = self._swap_to(tenant)
        for group in self.optimizer.param_groups:
            group["lr"] = adam.learning_rate
            group["betas"] = (adam.beta1, adam.beta2)
            group["eps"] = adam.eps
            group["weight_decay"] = adam.weight_decay
        # max_norm inf computes the norm and scales by 1: grad_clip_norm 0 means no clipping
        max_norm = adam.grad_clip_norm if adam.grad_clip_norm > 0 else math.inf
        norm = float(clip_grad_norm(max_norm, [self.model]))
        if not math.isfinite(norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteGradNorm(f"grad norm is {norm}; the optimizer step was not taken and the grads were zeroed")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
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

    def _adam_state(self, tenant: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """The tenant's full AdamW moments and step per LoRA parameter on CPU,
        keyed as the adapter is; empty until its first step. torch's Adam
        creates a parameter's state at its first step with a grad
        (`torch/optim/adam.py`, `if p.grad is not None`), so the state is all
        or none here, every LoRA parameter being in the forward path; a
        partial state is a bug and is refused, never written."""
        self._require_tenant(tenant)
        per_param: list   # (name, param, that param's state as local shards, or None)
        if tenant == self.live:
            state = self.optimizer.state
            per_param = [(n, p, {k: _local(state[p][k]) for k in ADAM_STATE_KEYS} if p in state else None)
                         for n, p in self.lora_params]
        else:
            slot = self.slots[tenant]
            per_param = [(n, p, slot.adam[n]) for n, p in self.lora_params]
        stepped = [st is not None for _, _, st in per_param]
        if any(stepped) and not all(stepped):
            raise RuntimeError(f"AdamW holds state for {sum(stepped)} of {len(stepped)} LoRA parameters; "
                               "a checkpoint of it would be partial")
        return {peft_key(n): {k: _full(st[k] if k == "step" else _on_device(p, st[k])) for k in ADAM_STATE_KEYS}
                for n, p, st in per_param if st is not None}

    def save_state(self, tenant: str, root: str) -> None:
        """Rank 0 writes the tenant's adapter (`hf_adapter/`) and the AdamW
        state of its LoRA parameters (`optimizer/`) under `root` (adapter.py),
        without swapping the tenant in. Both gathers are collective: call it
        on all ranks at once."""
        lora = self.lora_state(tenant)
        adam = self._adam_state(tenant)
        if self.rank == 0:
            write_peft_adapter(lora, self.lora, self.base_model, Path(root) / HF_ADAPTER_DIRNAME)
            write_adam_state(adam, Path(root) / OPTIMIZER_DIRNAME)

    def load_state(self, tenant: str, adapter_dir: str, optimizer_dir: Optional[str]) -> None:
        """Copy the adapter's weights into the tenant's shards and, given
        optimizer_dir, the AdamW state; without it the optimizer keeps what it
        has (fresh on a tenant that has not trained). The files' keys must
        be exactly this model's LoRA parameters: StateMismatch before anything
        is copied. Collective: call it on all ranks at once."""
        self._require_tenant(tenant)
        weights = load_file(str(Path(adapter_dir) / ADAPTER_WEIGHTS_FILE))
        self._require_keys(weights, "adapter")
        adam = None if optimizer_dir is None else read_adam_state(Path(optimizer_dir))
        if adam:
            self._require_keys(adam, "optimizer state")
        self._swap_to(tenant)
        with torch.no_grad():
            for n, p in self.lora_params:
                _copy_full(p, weights[peft_key(n)].float())
        if adam is None:
            return
        for n, p in self.lora_params:
            if not adam:   # saved before the first step: no state to restore
                self.optimizer.state.pop(p, None)
                continue
            st = adam[peft_key(n)]
            self.optimizer.state[p] = {"exp_avg": _like(p, st["exp_avg"]), "exp_avg_sq": _like(p, st["exp_avg_sq"]),
                                       "step": st["step"].clone()}

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
            write_peft_adapter(state, self.lora, self.base_model, Path(dest))

    def _run(self, shard: Batch, clip: Optional[Tuple[float, float]], backward: bool) -> Dict[str, Any]:
        rows, width = shard.input_ids.shape
        out: Dict[str, Any] = {
            "row_loss": torch.zeros(rows),
            "logprobs": torch.zeros(rows, width),
            "elementwise": torch.zeros(rows, width),
            "clamped": torch.zeros(rows),
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
            w = mb.input_ids.shape[1]
            out["row_loss"][start:end] = res.row_loss.cpu()
            out["logprobs"][start:end, :w] = res.logprobs.cpu()
            out["elementwise"][start:end, :w] = res.elementwise.cpu()
            out["clamped"][start:end] = res.clamped.cpu()
        return out


def _micro_batch(shard: Batch, start: int, end: int) -> Batch:
    """Rows [start, end), cut to their longest row (at least 1 column, so an
    all-padding micro-batch still runs)."""
    width = max(1, int(shard.lengths[start:end].max()))
    return Batch(
        loss_fn=shard.loss_fn,
        input_ids=shard.input_ids[start:end, :width],
        lengths=shard.lengths[start:end],
        inputs={k: v[start:end, :width] for k, v in shard.inputs.items()},
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


def _copy_full(p: torch.Tensor, full: torch.Tensor) -> None:
    """Copy `full` into `p` in place; a DTensor takes its own slice."""
    if isinstance(p, DTensor):
        local = _like(p, full)
        assert isinstance(local, DTensor)
        p.to_local().copy_(local.to_local())
    else:
        p.copy_(full)
