"""Ray GPU worker for the megatron_bridge backend.

Holds the Evo2 classifier + megatron distributed context INSIDE a Ray actor, so
the tinker-cloud server stays a plain process (like automodel in-process / nemo_rl
+ miles Ray-actor delegation) — NOT run under torchrun. Each model gets its own
actor, so megatron's global state (parallel_state, microbatch calculator) is
isolated per model: multi-model works and restarts are clean.

Imports megatron.bridge + the bionemo-recipes evo2_classifier, which exist only in
the cu13 recipe venv (deploy_tinkercloud.sh --profile megatron_bridge). Ray actors
inherit the driver (server) venv, so those are available in the actor.
"""
import os

import ray


def _sdk_fb_output(logits_BC, loss: float, accuracy: float) -> dict:
    """SDK ForwardBackwardOutput shape: per-sample class logits as TensorData +
    loss/accuracy in metrics (keys need the SDK's name:reduction format)."""
    outs = []
    if logits_BC is not None:
        arr = logits_BC.float().cpu().numpy()
        for row in arr:
            outs.append({"logits": {"data": [float(x) for x in row.tolist()],
                                    "dtype": "float32", "shape": [int(row.shape[0])]}})
    metrics = {}
    if loss == loss:
        metrics["loss:mean"] = float(loss)
    if accuracy == accuracy:
        metrics["accuracy:mean"] = float(accuracy)
    return {"loss_fn_output_type": "TorchLossReturn",
            "loss_fn_outputs": outs, "metrics": metrics}


def _reduce(losses, key: str) -> float:
    if not losses:
        return float("nan")
    num = den = 0.0
    for d in losses:
        if isinstance(d, dict) and key in d:
            v = d[key]
            num += float(v[0]); den += float(v[1])
    return num / den if den else float("nan")


def _pick_free_port() -> int:
    """Bind an ephemeral port and hand it back (torch's _find_free_port idiom).

    Chosen INSIDE the actor, immediately before init_process_group, so the same
    process that reads the port is the one that binds it — no driver->actor port
    passing, minimal TOCTOU window, and each actor gets a distinct port."""
    import socket
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@ray.remote(num_gpus=1)
class MegatronBridgeWorker:
    """One Evo2 classifier + optimizer, per model, in its own GPU actor."""

    def __init__(self, cfg_kwargs: dict, recipe_examples: str):
        # single-process distributed group (tp=1); Ray assigns the GPU via
        # CUDA_VISIBLE_DEVICES, we provide the 1-rank rendezvous env megatron needs.
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(_pick_free_port())

        import sys
        if recipe_examples not in sys.path:
            sys.path.insert(0, recipe_examples)
        from megatron.bridge.training.state import GlobalState
        from megatron.bridge.training.setup import setup
        from megatron.bridge.training.config import runtime_config_update
        from megatron.bridge.data.utils import get_dataset_provider
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from evo2_classifier import evo2_1b_classifier_config, classifier_forward_step

        from pathlib import Path
        for k in ("base_ckpt_dir", "train_jsonl", "val_jsonl", "test_jsonl",
                  "result_dir", "tokenizer_path"):
            if cfg_kwargs.get(k) is not None:
                cfg_kwargs[k] = Path(cfg_kwargs[k])

        cfg = evo2_1b_classifier_config(**cfg_kwargs)
        runtime_config_update(cfg)
        state = GlobalState()
        state.cfg = cfg
        so = setup(state, get_dataset_provider(cfg.dataset))
        self.state = state
        self.model = so.model
        self.optimizer = so.optimizer
        self.scheduler = so.scheduler
        self.fbfunc = get_forward_backward_func()
        self.seq_length = cfg.model.seq_length
        self.needs_zero = True
        self._cap = {}

        def _capturing_step(*a, **kw):
            logits, loss_fn = classifier_forward_step(state, *a, **kw)
            self._cap["logits"] = logits.detach()
            return logits, loss_fn

        self.forward_step = _capturing_step

    def ready(self) -> bool:
        return True

    def _run(self, batch: dict, forward_only: bool):
        import torch
        dev = torch.cuda.current_device()
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
        if not forward_only and self.needs_zero:
            for mc in self.model:
                if hasattr(mc, "zero_grad_buffer"):
                    mc.zero_grad_buffer()
            self.optimizer.zero_grad()
            self.needs_zero = False
        losses = self.fbfunc(
            forward_step_func=self.forward_step, data_iterator=iter([batch]),
            model=self.model, num_microbatches=1, seq_length=self.seq_length,
            micro_batch_size=batch["input_ids"].shape[0], forward_only=forward_only,
        )
        return _sdk_fb_output(self._cap.get("logits"),
                              _reduce(losses, "ce loss"), _reduce(losses, "accuracy"))

    def forward(self, batch: dict) -> dict:
        return self._run(batch, forward_only=True)

    def forward_backward(self, batch: dict) -> dict:
        return self._run(batch, forward_only=False)

    def apply_optimizer_step(self, learning_rate=None) -> dict:
        step_out = self.optimizer.step()
        grad_norm = step_out[1] if isinstance(step_out, (tuple, list)) and len(step_out) >= 2 else None
        self.scheduler.step(increment=1)
        self.needs_zero = True
        lr = self.optimizer.param_groups[0]["lr"] if getattr(self.optimizer, "param_groups", None) else None
        metrics = {}
        if grad_norm is not None:
            metrics["grad_norm:mean"] = float(grad_norm)
        if lr is not None or learning_rate is not None:
            metrics["learning_rate:mean"] = float(lr if lr is not None else learning_rate)
        return {"metrics": metrics}

    def save_checkpoint(self, checkpoint_path: str) -> str:
        from megatron.bridge.training.checkpointing import save_checkpoint as _save
        _save(state=self.state, model=self.model, optimizer=self.optimizer,
              opt_param_scheduler=self.scheduler, num_floating_point_operations_so_far=0)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        from megatron.bridge.training.checkpointing import load_checkpoint as _load
        _load(state=self.state, model=self.model, optimizer=self.optimizer,
              opt_param_scheduler=self.scheduler, checkpoint_path=checkpoint_path)
