"""Device helpers for the scripts and benchmarks: pick a device, time on it.

The library itself never needs these -- `sinkslot_solve` & co. dispatch on
their inputs' device. They exist so the experiment scripts (gradient_flow/,
color_transfer/, the bench harness) can run on CUDA, Apple-silicon MPS or CPU
without each one hardcoding `torch.cuda.*`.
"""

from __future__ import annotations

import torch

from .._backends import resolve_backend


def default_device(dtype=torch.float32) -> str:
    """The fastest device present that can hold `dtype`: "cuda", else "mps"
    (not for float64, which MPS lacks), else "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and dtype != torch.float64:
        return "mps"
    return "cpu"


def fp64_device(device) -> torch.device:
    """Where float64 work alongside `device` should run: `device` itself,
    except MPS (no float64), which falls back to CPU."""
    device = torch.device(device)
    return torch.device("cpu") if device.type == "mps" else device


def to(t: torch.Tensor, device, dtype) -> torch.Tensor:
    """`t` on `device` in `dtype`, in two steps with the cast done on CPU.

    On PyTorch 2.14.1 a single `t.to(device="cpu", dtype=torch.float64)` (or
    `t.to(cpu_float64_tensor)`) from an MPS tensor returns reinterpreted bytes
    rather than converted values -- 1/1024 comes back as 0 or 2e-314, with no
    error. Splitting it fixes that, but the order matters: a CPU float64 tensor
    must be cast before it moves to MPS (which cannot hold float64), and an MPS
    tensor must move to CPU before it is cast to float64. Casting on whichever
    side is CPU satisfies both.
    """
    if t.device.type == "cpu":
        return t.to(dtype).to(device)
    return t.to(device).to(dtype)


def synchronize(device) -> None:
    """Wait for queued work on `device`; a no-op on CPU."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device) -> None:
    """Release the caching allocator's unused blocks on `device`; a no-op on CPU."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def reset_peak_memory(device) -> None:
    """Start a fresh peak-memory window on `device` (CUDA only; see `peak_memory`)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device) -> float:
    """Peak allocated bytes on `device` since `reset_peak_memory`, or NaN where
    PyTorch keeps no peak counter (MPS, CPU) -- reported as missing rather
    than substituted with a current-usage figure that would read as a peak."""
    if torch.device(device).type == "cuda":
        return float(torch.cuda.max_memory_allocated(device))
    return float("nan")


def describe(device) -> str:
    """One line naming the device and the SinkSLOT backend `backend="auto"` uses on it."""
    backend = resolve_backend("auto", device)
    if backend == "torch":
        return (f"{device} (SinkSLOT on the pure-torch fallback -- much slower than "
                f"the fused kernels the reported numbers use, same algorithm)")
    return f"{device} (SinkSLOT on the fused {backend} kernels)"
