"""delete_model releases the clusters' placement groups even when a shutdown before them raises."""
import asyncio

import pytest

from tinkercloud.training.backends.base import BackendError
from tinkercloud.training.backends.nemo_rl import backend as nb
from tinkercloud.training.backends.nemo_rl.backend import NemoRLHandle


def test_a_policy_shutdown_that_raises_still_releases_both_clusters():
    """Regression guarded: a policy shutdown that raised skipped the cluster shutdowns, so the
    placement groups stayed held and the next create_model blocked."""
    calls = []

    class Policy:
        def shutdown(self):
            raise RuntimeError("policy worker died")

    class Cluster:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            calls.append(self.name)

    handle = NemoRLHandle(model_id="m", backend_type="nemo_rl", policy=Policy(),
                          inference_cluster=Cluster("inference"), cluster=Cluster("train"))
    backend = nb.NemoRLBackend()

    async def run():
        backend._open_scheduler(handle)
        with pytest.raises(BackendError, match="policy worker died"):
            await backend.delete_model(handle)
        await backend.close()

    asyncio.run(run())
    assert calls == ["inference", "train"]
    assert handle.torn_down is False
