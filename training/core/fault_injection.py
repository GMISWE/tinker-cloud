"""Test-only fault injection (specs/028 P0).

``TINKERCLOUD_FAULT_HANG=<backend>.<op>`` makes every call that reaches the
named hang point block forever, the way a NCCL stall or a wedged engine does.
Unset, the production state, every hang point returns at once.
"""
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

ENV = "TINKERCLOUD_FAULT_HANG"


async def hang_point(name: str) -> None:
    """Block forever when `name` is the configured hang point."""
    if os.getenv(ENV, "") != name:
        return
    logger.error("%s=%s: this op blocks forever", ENV, name)
    await asyncio.Event().wait()
