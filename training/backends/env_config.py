"""
Typed, explicit backend configuration (Constitution P3).

Each backend declares its knobs as a Pydantic model: one field per
environment variable, with type and default. Values come from the
environment, then from `BackendConfig.backend_overrides` (which wins), and
the effective configuration is logged at startup with each value's source.
An empty environment variable counts as unset.
"""
import os
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr


T = TypeVar("T", bound="EnvConfig")


class EnvConfig(BaseModel):
    """Base for per-backend config models. Subclasses set ENV: field -> variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    ENV: ClassVar[Dict[str, str]] = {}

    _sources: Dict[str, str] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_env(cls: Type[T], overrides: Optional[Dict[str, Any]] = None,
                 environ: Optional[Dict[str, str]] = None) -> T:
        env = os.environ if environ is None else environ
        values: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        for field, var in cls.ENV.items():
            raw = env.get(var, "")
            if raw.strip() != "":
                values[field] = raw
                sources[field] = f"env {var}"
        for key, val in (overrides or {}).items():
            values[key] = val
            sources[key] = "override"
        cfg = cls.model_validate(values)
        cfg._sources = sources
        return cfg

    def source_of(self, field: str) -> str:
        return self._sources.get(field, "default")

    def describe(self) -> str:
        """One line per field: name=value (source)."""
        return "\n".join(
            f"  {name}={getattr(self, name)!r} ({self.source_of(name)})"
            for name in type(self).model_fields
        )
