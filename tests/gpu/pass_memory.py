"""Shared by the suites that read a pass's GPU memory metrics through the SDK."""
import torch

ALLOCATED, RESERVED, DEVICE = "gpu_allocated_peak_bytes:max", "gpu_reserved_peak_bytes:max", "gpu_device_used_bytes:max"


def rows(count: int, length: int, generator: torch.Generator):
    """`count` cross-entropy rows of `length` random tokens, each position weighted 1/length."""
    from tinker import types

    out = []
    for _ in range(count):
        tokens = torch.randint(100, 20000, (length,), generator=generator)
        out.append(types.Datum(model_input=types.ModelInput.from_ints(tokens.tolist()), loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(tokens.roll(-1)),
            "weights": types.TensorData.from_torch(torch.full((length,), 1.0 / length))}))
    return out


def check_readings_nest(passes: dict) -> None:
    """Print every pass's three readings and hold their order: tensors within the allocator's blocks
    within the device's used memory."""
    for name, metrics in passes.items():
        print(f"\n[pass memory] {name}: " + ", ".join(f"{k} {metrics[k] / 2**20:.0f} MiB" for k in (ALLOCATED, RESERVED, DEVICE)))
        assert 0 < metrics[ALLOCATED] <= metrics[RESERVED] <= metrics[DEVICE]
