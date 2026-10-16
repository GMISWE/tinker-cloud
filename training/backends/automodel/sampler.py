"""
One base model's vLLM (Q5, R4): a placement group of one {GPU:1, CPU:1}
bundle, a SamplerHost actor in it, and the vLLM server the actor runs as a
subprocess from /opt/vllm-env (docker/automodel). The training env never
imports vLLM; the backend talks to the server over HTTP and loads tenant
adapters into it by name (backend.py).

The subprocess is bound to the actor: it inherits Ray's CUDA_VISIBLE_DEVICES,
asks the kernel for SIGKILL when the actor process dies (PR_SET_PDEATHSIG), and
runs in its own session so stop() signals the API server and vLLM's engine
process together. Killing the actor, by delete or by the leftover sweep (R5),
therefore frees the GPU.

Names: placement group `am-infer-<key>`, actor `am-infer-<key>-host`, key =
hash of the base model name. Neither is detached.
"""
import asyncio
import ctypes
import http.client
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import ray
from ray.exceptions import RayActorError, RayTaskError
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..base import BackendError
from .config import AutomodelConfig
from .group import NAME_PREFIX, PG_READY_TIMEOUT_S

logger = logging.getLogger(__name__)

VLLM_SERVER = "/opt/vllm-env/bin/vllm-api-server"
# Weights load, torch.compile and CUDA-graph capture; an 8B model takes minutes.
BOOT_TIMEOUT_S = 900.0
# vLLM's --max-logprobs: the top-k it computes per position (handle.max_topk_logprobs).
MAX_LOGPROBS = 20
PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Runs in the child between fork and exec; survives the exec."""
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_PDEATHSIG): {os.strerror(errno)}")


def _healthy(url: str) -> bool:
    """One GET /health. False while the server is not listening or not ready
    yet; the caller polls."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2.0) as response:
            return response.status == 200
    except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
        return False


class SamplerHost:
    """Wrap with ray.remote(num_gpus=1, num_cpus=1); one per base model."""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None
        self.log_path: Optional[str] = None

    def start(self, base_model: str, max_model_len: int, max_loras: int, max_lora_rank: int,
              max_cpu_loras: int) -> str:
        """Start vLLM on this bundle's GPU and wait until /health answers.
        Returns the server URL. Raises RuntimeError with the log tail if the
        process exits or the boot timeout passes."""
        host = ray.util.get_node_ip_address()
        with socket.socket() as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
        self.url = f"http://{host}:{port}"
        self.log_path = f"/tmp/{NAME_PREFIX}vllm-{port}.log"
        args = [
            VLLM_SERVER, "--model", base_model, "--host", host, "--port", str(port),
            "--enable-lora", "--max-loras", str(max_loras), "--max-lora-rank", str(max_lora_rank),
            "--max-cpu-loras", str(max_cpu_loras), "--max-model-len", str(max_model_len),
            "--max-logprobs", str(MAX_LOGPROBS),
        ]
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, preexec_fn=_die_with_parent)
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM for {base_model} exited with {self.proc.returncode} while starting "
                    f"(log {self.log_path}):\n{self._tail()}")
            if _healthy(self.url):
                return self.url
            time.sleep(2.0)
        self.stop()
        raise RuntimeError(
            f"vLLM for {base_model} did not answer /health within {BOOT_TIMEOUT_S:.0f}s "
            f"(log {self.log_path}):\n{self._tail()}")

    def stop(self) -> None:
        """SIGTERM the server's session (API server and engine process), SIGKILL
        it if it has not exited in 30 s. Safe to call twice."""
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()

    def _tail(self, lines: int = 30) -> str:
        assert self.log_path is not None
        with open(self.log_path, "rb") as log:
            return b"".join(log.readlines()[-lines:]).decode(errors="replace")


class SamplerGroup:
    """The driver's side of one base model's sampler: its placement group,
    its SamplerHost and the URL the host reported."""

    def __init__(self, key: str):
        self.key = key
        self.name = f"{NAME_PREFIX}infer-{key}"
        self.pg: Optional[Any] = None
        self.host: Optional[Any] = None
        self.url: Optional[str] = None

    async def start(self, base_model: str, max_model_len: int, config: AutomodelConfig) -> str:
        """Place the bundle, start the host, boot vLLM; returns its URL. A
        failure stops what was started, then re-raises (R5)."""
        try:
            pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK", name=self.name)
            self.pg = pg
            ready, _ = await asyncio.to_thread(ray.wait, [pg.ready()], timeout=PG_READY_TIMEOUT_S)
            if not ready:
                raise BackendError(
                    f"a free GPU for the {base_model} sampler was not placed within {PG_READY_TIMEOUT_S:.0f}s",
                    backend="automodel", operation="create_model")
            host_cls = ray.remote(num_gpus=1, num_cpus=1)(SamplerHost)
            host = host_cls.options(
                name=f"{self.name}-host",
                scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=0),
            ).remote()
            self.host = host
            try:
                url: str = await host.start.remote(
                    base_model, max_model_len, config.max_loras, config.max_lora_rank, config.max_cpu_loras)
            except RayTaskError as e:
                # the host's own sentence (with the log tail), not Ray's traceback
                raise BackendError(str(e.cause), backend="automodel", operation="create_model") from e
            self.url = url
            return url
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop vLLM, kill the host, free the bundle. Safe to call twice."""
        if self.host is not None:
            try:
                await self.host.stop.remote()   # graceful: vLLM reaps its engine process
            except RayActorError:
                pass   # the host is already dead; its vLLM died with it (PDEATHSIG)
            ray.kill(self.host, no_restart=True)
            self.host = None
        if self.pg is not None:
            remove_placement_group(self.pg)
            self.pg = None
