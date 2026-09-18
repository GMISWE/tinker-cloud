"""
Routing table: model_id -> inference endpoint, the one place the address of
a model's HTTP inference engine is kept.

Backends record the endpoint they booted on the handle; the model service
publishes it here at create_model and withdraws it at delete_model; the
sample path looks the address up per request instead of reading it off the
handle. A model served by an in-process engine (NeMo RL, veRL, fake) has no
entry. Reads are lock-free: the table is an immutable mapping replaced whole
on every write (writes happen once per model lifetime; reads once per sample).
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class InferenceEndpoint:
    """Base URL of an HTTP inference engine, e.g. http://10.0.0.7:30000."""

    base_url: str

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"inference endpoint must be an http(s) URL, got {self.base_url!r}")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))


class RoutingError(LookupError):
    """No inference endpoint is published for the model."""


class RoutingTable:
    def __init__(self) -> None:
        self._routes: Mapping[str, InferenceEndpoint] = MappingProxyType({})

    def publish(self, model_id: str, endpoint: InferenceEndpoint) -> None:
        routes = dict(self._routes)
        routes[model_id] = endpoint
        self._routes = MappingProxyType(routes)

    def withdraw(self, model_id: str) -> Optional[InferenceEndpoint]:
        """Remove the model's route; None if it had none (in-process engine)."""
        if model_id not in self._routes:
            return None
        routes = dict(self._routes)
        endpoint = routes.pop(model_id)
        self._routes = MappingProxyType(routes)
        return endpoint

    def endpoint_for(self, model_id: str) -> InferenceEndpoint:
        try:
            return self._routes[model_id]
        except KeyError:
            raise RoutingError(f"no inference endpoint published for model {model_id}") from None

    def snapshot(self) -> Mapping[str, InferenceEndpoint]:
        """The current routes; a later publish/withdraw does not change it."""
        return self._routes


# Process-wide table, one writer (services.model_service), read by backends.
table = RoutingTable()
