"""Engine → gateway registration: register once the socket serves, heartbeat, unregister on exit."""
import asyncio
import logging
import os
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class RegisterClient:
    """Keeps this engine's row current in the gateway's registry. Never raises into the engine."""

    def __init__(self, gateway_socket: str, name: str, backend: str, gpus: Optional[List[int]], socket: str,
                 interval_s: float) -> None:
        self._name = name
        self._row = {"name": name, "backend": backend, "gpus": gpus, "socket": socket, "pid": os.getpid()}
        self._interval = interval_s
        self._client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=gateway_socket),
                                         base_url="http://gateway", timeout=5.0)
        self._task: Optional[asyncio.Task] = None
        self.registered = asyncio.Event()
        self.refused = asyncio.Event()   # the gateway answered 4xx: a conflict or a bad row, not retried

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"register-{self._name}")

    async def _run(self) -> None:
        delay = 0.2
        while True:
            try:
                r = await self._client.post("/engines/register", json=self._row)
                if 400 <= r.status_code < 500:   # refused for good: another engine holds the gateway, or our row is wrong
                    logger.error("The gateway refused engine %s: %s %s; not retrying", self._name, r.status_code, r.text)
                    self.refused.set()
                    return
                r.raise_for_status()
                self.registered.set()
                logger.info("Registered with the gateway as %s", self._name)
                delay = 0.2
                await self._heartbeat()      # returns when the gateway no longer knows us
            except httpx.HTTPError as exc:   # gateway not up yet, restarting, or failing
                self.registered.clear()
                logger.warning("Gateway registration of %s failed (%s); retrying in %.1fs", self._name, exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            r = await self._client.post(f"/engines/{self._name}/heartbeat")
            if r.status_code == 404:      # the gateway restarted and lost its registry
                self.registered.clear()
                logger.warning("The gateway lost engine %s; registering again", self._name)
                return
            r.raise_for_status()

    async def sampler_created(self, sampler_id: str, model_id: str, pinned_version: Optional[int],
                              model_path: Optional[str], base_model: str) -> None:
        """Record an ephemeral sampler (save_weights_for_sampler) with the gateway before its id
        reaches the SDK. A gateway that refuses it raises: a sampler the gateway cannot route is
        no sampler."""
        r = await self._client.post(f"/engines/{self._name}/samplers", json={
            "sampler_id": sampler_id, "model_id": model_id, "pinned_version": pinned_version,
            "model_path": model_path, "base_model": base_model})
        r.raise_for_status()

    async def model_failed(self, model_id: str) -> None:
        """A create the gateway recorded at submit is not coming: its row goes now, not at the next delete."""
        r = await self._client.delete(f"/engines/{self._name}/models/{model_id}")
        if r.status_code != 404:   # no row: the gateway already dropped it
            r.raise_for_status()

    async def close(self) -> None:
        """Unregister. Idempotent; a gateway that is already gone is logged, not raised."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            if not self.refused.is_set():
                try:
                    await self._client.delete(f"/engines/{self._name}", timeout=1.0)
                except httpx.HTTPError as exc:
                    logger.warning("Unregistering %s failed: %s", self._name, exc)
            self.registered.clear()
        await self._client.aclose()
