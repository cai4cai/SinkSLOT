"""Fused-kernel backends: which one runs on which device, and how to reach it.

Each fused backend is named after its kernel language, not its device --
"triton" runs on `cuda` tensors, "metal" (Metal shaders compiled at runtime
with `torch.mps.compile_shader`) on `mps` tensors -- so `backend` and
`device` never mean the same thing. "torch" is the pure-torch fallback
(`sinkslot_alternating_torch` & co.) and runs anywhere.

Every fused backend is a module (`_triton.py`, `_metal.py`) exposing the same
two entry points, so the solver loops and the cost are written once against
`fused_ops(name)` and never test for a particular backend:

    seg_lse(indptr, colidx, lam, phi, psi, out, base, block, num_warps)
        out[i] = LSE_k(lam[k] + phi[i] + psi[colidx[k]]) over CSR row i,
        or base[i] - LSE when `base` is not None. Writes into `out`.
    sqeuclidean_cost(x, y, rows, cols)
        float32 ||x[rows[k]] - y[cols[k]]||^2, one fused pass.

Adding a backend is one entry in `DEVICE_BACKENDS` and `_AVAILABLE`, one
branch in `fused_ops`, and a module implementing the two functions above.
"""

from __future__ import annotations

import functools

import torch

try:
    import triton  # noqa: F401  -- probed only; _triton.py imports it for real
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# compile_shader arrived after the torch>=2.5 floor in pyproject.toml, so the
# Metal backend is gated on it rather than raising the floor for everyone.
_HAS_METAL = hasattr(torch, "mps") and hasattr(torch.mps, "compile_shader") \
    and torch.backends.mps.is_available()

# Fused backend for each device type; backend="auto" picks from this table.
DEVICE_BACKENDS = {"cuda": "triton", "mps": "metal"}
FUSED_BACKENDS = tuple(DEVICE_BACKENDS.values())
BACKENDS = ("auto", "torch", *FUSED_BACKENDS)

_AVAILABLE = {"triton": _HAS_TRITON, "metal": _HAS_METAL}
_REQUIRES = {
    "triton": "Triton installed",
    "metal": "a PyTorch build with torch.mps.compile_shader",
}


def fused_backend(device) -> str | None:
    """The fused backend that runs on `device`, or None if it has none here
    (CPU, or the backend's toolchain isn't installed)."""
    name = DEVICE_BACKENDS.get(torch.device(device).type)
    return name if name is not None and _AVAILABLE[name] else None


def resolve_backend(backend: str, device) -> str:
    """`backend` ("auto" / "torch" / a fused name) -> the one that will run.

    "auto" is the device's fused backend when available, "torch" otherwise.
    A fused name is returned as-is if it can run on `device`, and raises
    `ValueError` if not, rather than silently falling back to torch.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        return fused_backend(device) or "torch"
    if backend != "torch" and fused_backend(device) != backend:
        device_type = next(d for d, b in DEVICE_BACKENDS.items() if b == backend)
        raise ValueError(
            f"backend={backend!r} requires {_REQUIRES[backend]} and X on "
            f"{device_type.upper()}; use backend='torch' otherwise"
        )
    return backend


@functools.cache
def fused_ops(name: str):
    """The kernel module of fused backend `name`, imported on first use."""
    if name == "triton":
        from . import _triton
        return _triton
    if name == "metal":
        from . import _metal
        return _metal
    raise ValueError(f"{name!r} is not a fused backend; expected one of {FUSED_BACKENDS}")
