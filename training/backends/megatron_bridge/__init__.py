"""Megatron-Bridge backend — Evo2 StripedHyena classification (Megatron/NeMo2)."""
from .backend import MegatronBridgeBackend, MegatronBridgeHandle
from .builder import MegatronBridgeArgumentBuilder
from .converter import NeMo2ClassificationDataConverter

__all__ = [
    "MegatronBridgeBackend",
    "MegatronBridgeHandle",
    "MegatronBridgeArgumentBuilder",
    "NeMo2ClassificationDataConverter",
]
