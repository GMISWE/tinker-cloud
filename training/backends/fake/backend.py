"""
FakeBackend — a deterministic, in-process TrainingBackend with no GPU and no Ray.

It exists so the HTTP/session/futures/checkpoint protocol above the backends
can be exercised by the real `tinker` SDK in CI. Every observable is a pure
function of the inputs:

- logprobs[i] = -(token_i % 7) / 7, one per model_input token
- grad_norm   = number of optimizer steps taken on the model (1-based)
- sample      = tokens derived from (prompt, seed, sample index); the same
                seed always yields the same tokens
- weights     = a scalar `w` advanced by lr * pending_microbatches per step,
                so a save/load round trip is checkable

Set FAKE_BACKEND_TRACE=<path> to append one JSON line per backend call
({"op", "model_id", ...args}) — tests read this instead of poking internals.
"""
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import BackendError, BackendHandle, TrainingBackend
from ...models.requests import Datum, ModelInput
from ...core.loss_registry import LOSS_FNS

logger = logging.getLogger(__name__)

STATE_FILE = "fake_state.json"


@dataclass
class FakeHandle(BackendHandle):
    base_model: str = ""
    lora_config: Optional[Dict[str, Any]] = None
    hf_path: str = ""
    w: float = 0.0
    step_count: int = 0
    pending: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


def _tokens_of(model_input: ModelInput) -> List[int]:
    if model_input.tokens:
        return list(model_input.tokens)
    if model_input.input_ids:
        return list(model_input.input_ids)
    return [t for chunk in model_input.chunks or [] for t in (chunk.tokens or [])]


def _datum_parts(datum: Datum):
    return datum.model_input, datum.loss_fn_inputs


def decode(tokens: List[int]) -> str:
    """The fake tokenizer: token t renders as its decimal digits, space-joined.
    Deterministic and invertible, so stop-string tests need no real tokenizer."""
    return " ".join(str(t) for t in tokens)


def logprobs_for(tokens: List[int]) -> List[float]:
    return [-(t % 7) / 7.0 for t in tokens]


class FakeBackend(TrainingBackend):
    needs_ray = False
    SUPPORTED_LOSS_FNS = frozenset(LOSS_FNS)

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._models: Dict[str, FakeHandle] = {}
        self._trace_path = os.getenv("FAKE_BACKEND_TRACE")

    # --- tracing ---------------------------------------------------------
    def _trace(self, op: str, model_id: str, **kw: Any) -> None:
        if not self._trace_path:
            return
        with open(self._trace_path, "a") as f:
            f.write(json.dumps({"op": op, "model_id": model_id, **kw}, default=str) + "\n")

    def _handle(self, handle: BackendHandle, op: str) -> FakeHandle:
        h = self._models.get(handle.model_id)
        if h is None:
            raise BackendError(f"Model {handle.model_id} not found", backend="fake", operation=op)
        return h

    # --- lifecycle -------------------------------------------------------
    async def create_model(
        self, model_id: str, request_id: str, base_model: str, num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None, parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None, rollout_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False, resume_from: Optional[Path] = None,
        max_batch_size: int = 4096, max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None, wandb_config: Optional[Dict[str, Any]] = None,
        staleness_k: int = 0, objective: str = "language_modeling",
        num_labels: Optional[int] = None, head_config: Optional[Dict[str, Any]] = None,
        native_root: Optional[Path] = None,
    ) -> BackendHandle:
        if objective != "language_modeling":
            raise BackendError(f"objective {objective!r} unsupported", backend="fake", operation="create_model")
        h = FakeHandle(model_id=model_id, backend_type="fake", base_model=base_model,
                       lora_config=lora_config, hf_path=base_model, context_length=max_seq_len)
        if resume_from:
            self._load_into(h, resume_from)
        self._models[model_id] = h
        self._trace("create_model", model_id, base_model=base_model, lora_config=lora_config,
                    resume_from=resume_from, native_root=native_root)
        return h

    async def delete_model(self, handle: BackendHandle) -> None:
        self._models.pop(handle.model_id, None)
        self._trace("delete_model", handle.model_id)

    # --- training --------------------------------------------------------
    def _outputs(self, data: List[Any]):
        outputs, losses = [], []
        for datum in data:
            model_input, _ = _datum_parts(datum)
            lp = logprobs_for(_tokens_of(model_input))
            losses.append(-sum(lp) / max(len(lp), 1))
            outputs.append({"logprobs": {"data": lp, "shape": [len(lp)], "dtype": "float32"}})
        return outputs, (sum(losses) / len(losses) if losses else 0.0)

    async def forward(self, handle: BackendHandle, data: List[Any], loss_fn: str,
                      loss_fn_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        h = self._handle(handle, "forward")
        outputs, _ = self._outputs(data)
        self._trace("forward", h.model_id, n=len(data), loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        return {"type": "forward", "loss_fn_output_type": loss_fn, "loss_fn_outputs": outputs, "metrics": {}}

    async def forward_backward(self, handle: BackendHandle, data: List[Any], loss_fn: str,
                               loss_fn_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        h = self._handle(handle, "forward_backward")
        outputs, loss = self._outputs(data)
        h.pending += 1
        self._trace("forward_backward", h.model_id, n=len(data), loss_fn=loss_fn, loss_fn_config=loss_fn_config)
        metrics = {"loss:mean": loss}
        for k, v in (loss_fn_config or {}).items():
            metrics[f"{k}:mean"] = float(v)
        return {
            "loss_fn_output_type": loss_fn,
            "loss": loss,
            "metrics": metrics,
            "loss_fn_outputs": outputs,
            "deferred": False,
        }

    async def apply_optimizer_step(
        self, handle: BackendHandle, learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        h = self._handle(handle, "apply_optimizer_step")
        lr = learning_rate if learning_rate is not None else 0.0
        h.step_count += 1
        h.w += lr * h.pending
        h.pending = 0
        h.weight_version += 1
        self._trace("apply_optimizer_step", h.model_id, learning_rate=lr, adam_params=adam_params)
        return {
            "success": True,
            "grad_norm": float(h.step_count),
            "learning_rates": [],
            "model_id": h.model_id,
            "metrics": {"total_loss": 0.0, "fake_w": h.w},
            "weight_version": h.weight_version,
        }

    async def get_logprobs(self, handle: BackendHandle, data: List[Any]) -> List[Any]:
        self._handle(handle, "get_logprobs")
        return [logprobs_for(_tokens_of(_datum_parts(d)[0])) for d in data]

    # --- generation ------------------------------------------------------
    async def update_inference_weights(self, handle: BackendHandle) -> None:
        self._trace("update_inference_weights", handle.model_id)

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        self._handle(handle, "prepare_for_generation")

    async def sample(
        self, handle: BackendHandle, request_id: str, prompt_tokens: List[int], num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None, prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        h = self._handle(handle, "sample")
        params = dict(sampling_params or {})
        max_tokens = int(params.get("max_tokens") or 4)
        seed = params.get("seed")
        stop_ids = set(params.get("stop_token_ids") or [])
        stop_strs = [str(x) for x in (params.get("stop") or [])]
        self._trace("sample", h.model_id, prompt_len=len(prompt_tokens), num_samples=num_samples,
                    sampling_params=params, prompt_logprobs=prompt_logprobs, pinned_version=pinned_version)
        sequences = []
        for i in range(num_samples):
            rng = random.Random((seed if seed is not None else sum(prompt_tokens)) * 1000 + i)
            toks, stop_reason = [], "length"
            for _ in range(max_tokens):
                t = rng.randrange(4, 1000)
                # Contract (API-CONTRACT S4/S5): the token that triggered the stop
                # is part of the returned sequence, for token ids and strings alike.
                toks.append(t)
                if t in stop_ids or (stop_strs and any(ss in decode(toks) for ss in stop_strs)):
                    stop_reason = "stop"
                    break
            sequences.append({"tokens": toks, "logprobs": logprobs_for(toks),
                              "text": decode(toks), "stop_reason": stop_reason})
        out: Dict[str, Any] = {
            "sequences": sequences,
            "prompt_logprobs": ([None] + logprobs_for(prompt_tokens)[1:]) if prompt_logprobs else None,
            "weight_version": h.weight_version,
            "latest_weight_version": h.weight_version,
        }
        return out

    # --- checkpoints -----------------------------------------------------
    async def save_checkpoint(self, handle: BackendHandle, root: Path, step: Optional[int] = None,
                              persist: bool = True) -> None:
        h = self._handle(handle, "save_checkpoint")
        if persist:
            # a name starting with "slow-" writes slowly, so the protocol suite
            # can observe the store's pending state (overrides: save_delay_s)
            if root.name.startswith("slow-"):
                await asyncio.sleep(float(self.overrides.get("save_delay_s", 1.0)))
            with open(os.path.join(root, STATE_FILE), "w") as f:
                json.dump({"w": h.w, "weight_version": h.weight_version, "base_model": h.base_model,
                           "lora_config": h.lora_config, "step": step,
                           "optimizer": {"step_count": h.step_count}}, f)
        self._trace("save_checkpoint", h.model_id, root=str(root), step=step, persist=persist)

    def _load_into(self, h: FakeHandle, root: Path, optimizer: bool = False) -> None:
        path = os.path.join(root, STATE_FILE)
        if not os.path.exists(path):
            raise BackendError(f"Checkpoint {root} holds no {STATE_FILE}",
                               backend="fake", operation="load_checkpoint")
        with open(path) as f:
            st = json.load(f)
        if optimizer and "optimizer" not in st:
            raise BackendError(f"Checkpoint {root} carries no optimizer state",
                               backend="fake", operation="load_checkpoint")
        h.w = st["w"]
        h.weight_version = st["weight_version"]
        if optimizer:
            h.step_count = st["optimizer"]["step_count"]

    async def load_checkpoint(self, handle: BackendHandle, root: Path,
                              optimizer: bool = False) -> None:
        h = self._handle(handle, "load_checkpoint")
        self._load_into(h, root, optimizer=optimizer)
        self._trace("load_checkpoint", h.model_id, root=str(root), optimizer=optimizer)
