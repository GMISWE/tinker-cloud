"""core.fault_injection: a hang point blocks only the op the env var names."""
import asyncio

import pytest

from tinkercloud.training.core.fault_injection import ENV, hang_point


def test_hang_point_returns_when_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    asyncio.run(hang_point("automodel.apply_optimizer_step"))


def test_hang_point_blocks_only_the_named_op(monkeypatch):
    monkeypatch.setenv(ENV, "automodel.apply_optimizer_step")

    async def main():
        await hang_point("automodel.forward_backward")   # another op: returns
        hung = asyncio.create_task(hang_point("automodel.apply_optimizer_step"))
        await asyncio.sleep(0.05)
        assert not hung.done()
        hung.cancel()
        with pytest.raises(asyncio.CancelledError):
            await hung

    asyncio.run(main())
