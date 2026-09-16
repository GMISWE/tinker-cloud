from dataclasses import dataclass, field
from typing import Dict, Set, Any
import asyncio


@dataclass
class TrainingRuntimeState:
    """Holds mutable runtime structures for the training API."""

    training_clients: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    training_runs_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    poll_tracking: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    background_tasks: Set[asyncio.Task] = field(default_factory=set)
