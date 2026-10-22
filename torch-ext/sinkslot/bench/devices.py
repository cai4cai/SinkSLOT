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

    On PyTorch 2.14.1 copying an MPS tensor into a CPU float64 tensor in one
    step never writes the destination, and raises nothing: `cpu_f64.copy_(t)`
    leaves it unchanged, and `t.to(device="cpu", dtype=torch.float64)` (or
    `t.to(cpu_float64_tensor)`) returns uninitialised memory -- usually zeros,
    sometimes arbitrary values. Every source dtype is affected; other
    destination dtypes, and the reverse direction, are not. This is PyTorch
    issue pytorch/pytorch#197715 (a 2.14 regression); the fix,
    pytorch/pytorch#197722, makes the one-step call raise rather than convert,
    so the two-step form below stays necessary after it ships. Two steps
    avoid the bug, but their order matters: a CPU float64 tensor
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


def empty_caches() -> None:
    """`empty_cache` on every GPU device type present, for code that has no
    device at hand (e.g. after an out-of-memory error in a metrics helper)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
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


# Solver families with no implementation off CUDA, keyed by the method-name
# prefix every script uses for them, with the reason shown when one is skipped.
# The one place this lives: bench_forward and color_transfer/ all read it.
CUDA_ONLY = {
    "flash": "FlashSinkhorn is Triton-only",
    "geomloss_online": "GeomLoss online is KeOps, which has no MPS support",
    "geomloss_multiscale": "GeomLoss multiscale is KeOps, which has no MPS support",
    "ott": "OTT-JAX runs on JAX's own devices, not torch's MPS",
}


def cuda_only_reason(method: str) -> str | None:
    """Why `method` can only run on CUDA, or None if it runs anywhere."""
    return next((why for prefix, why in CUDA_ONLY.items() if method.startswith(prefix)), None)


def methods_for_device(device, requested, all_methods) -> list:
    """The methods to run on `device`: `requested`, or when None every method in
    `all_methods` that `device` can run (printing what was skipped). Raises
    `SystemExit` if a method requested explicitly needs CUDA and `device` isn't."""
    if torch.device(device).type == "cuda":
        return list(all_methods if requested is None else requested)
    if requested is None:
        skipped = [m for m in all_methods if cuda_only_reason(m)]
        if skipped:
            print(f"Skipping CUDA-only methods on {device}: {', '.join(skipped)}")
        return [m for m in all_methods if not cuda_only_reason(m)]
    bad = {m: cuda_only_reason(m) for m in requested if cuda_only_reason(m)}
    if bad:
        raise SystemExit("; ".join(f"{m} needs CUDA ({why})" for m, why in bad.items())
                         + f"; not available on {device}")
    return list(requested)


def describe(device) -> str:
    """One line naming the device and the SinkSLOT backend `backend="auto"` uses on it."""
    backend = resolve_backend("auto", device)
    if backend == "torch":
        return (f"{device} (SinkSLOT on the pure-torch fallback -- much slower than "
                f"the fused kernels the reported numbers use, same algorithm)")
    return f"{device} (SinkSLOT on the fused {backend} kernels)"
