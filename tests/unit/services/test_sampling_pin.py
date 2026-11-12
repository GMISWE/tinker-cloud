"""The sampling service takes the sampler's pin from every caller: `None` says "unpinned", silence says nothing."""
import asyncio

import pytest

from tinkercloud.training.backends.fake.backend import FakeBackend
from tinkercloud.training.services.sampling_service import SamplingService


@pytest.mark.parametrize("call", [
    lambda s: s.async_sample(request_id="r", prompt_tokens=[1, 2], num_samples=1, sampling_params=None,
                             prompt_logprobs=False, training_clients={}),
    lambda s: s.sync_sample(request_id="r", prompts=[[1, 2]], num_samples=1, sampling_params=None,
                            training_clients={}),
])
def test_a_call_that_names_no_pin_is_an_error(call):
    """With a default of None a route that forgot the pin sampled live weights and nothing said so."""
    with pytest.raises(TypeError, match="pinned_version"):
        asyncio.run(call(SamplingService(FakeBackend())))
