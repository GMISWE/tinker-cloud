"""A pass's GPU memory: what a rank reads of its own process and device, and how a backend
reports the ranks' readings in the pass's result.

A rank restarts the allocator's peaks when a pass starts and reads when it ends. The backend then
gives every request of the pass each reading as `<name>:max`, the largest over the ranks.
"""
import contextlib
from typing import Dict, Iterator, List

import torch


def restart_peaks() -> None:
    """Start a pass's window: the allocator's two peaks count from here."""
    torch.cuda.reset_peak_memory_stats()


def reading() -> Dict[str, int]:
    """This process's GPU memory in bytes. The two peaks are the allocator's highest levels since
    `restart_peaks`, the model included: tensors, and tensors with the blocks the allocator keeps.
    `gpu_device_used_bytes` is the device's total less its free memory now: every process on the
    GPU, with what CUDA and its libraries hold outside the allocator."""
    free, total = torch.cuda.mem_get_info()
    return {"gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            "gpu_device_used_bytes": total - free}


@contextlib.contextmanager
def pass_reading() -> Iterator[Dict[str, int]]:
    """For a rank that runs the whole pass in one call: the pass runs inside this block, and the
    yielded dict holds `reading()` when the block has ended. A rank whose pass and read are two
    calls uses `restart_peaks` and `reading`."""
    restart_peaks()
    memory: Dict[str, int] = {}
    yield memory
    memory.update(reading())


def largest_over_ranks(readings: List[Dict[str, int]]) -> Dict[str, int]:
    """The ranks' readings as a pass's metrics: `<name>:max` for each name."""
    return {f"{name}:max": max(r[name] for r in readings) for name in readings[0]}
