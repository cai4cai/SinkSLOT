"""Triton fused kernels (CUDA): segmented online LSE and sparse sq-euclidean cost.

One of the fused backends dispatched by `_backends.fused_ops`; `_metal.py` is
the Apple-silicon counterpart with the same two entry points. Imported only
once a Triton backend has actually been selected, so it can import triton
unconditionally.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _seg_lse_online_kernel(
    indptr, colidx, lam, phi, psi, out, base, SUB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass LSE with a running max -- FlashAttention's online softmax.

    The two-pass kernel reads each row twice: once to find the max, once to
    accumulate exp(v - max). This keeps a running (m, s) pair and rescales the
    accumulator whenever the max grows:

        m' = max(m, max(block));   s' = s * exp(m - m') + sum(exp(block - m'))

    so every element is touched once. Halves the loads of colidx, lam and the
    gathered psi, which is the whole cost here -- the arithmetic is negligible.

    With SUB the kernel stores `base[i] - LSE` rather than the LSE itself. The
    caller always wants `log_a - LSE`, and at small n the loop is bound by kernel
    launches rather than bandwidth (67 GB/s at n=1000 against 443 at n=10000), so
    folding that subtraction in removes two of the four launches per iteration.

    Dtype: `m`/`s` (the running max/sum) take their dtype from `lam`/`phi`/`psi`
    at the call site, not from a hardcoded declaration here -- unlike
    `_cost_kernel`'s accumulator below, there's no `tl.zeros(..., dtype=...)`
    pinning them. In practice every caller in this repo passes fp32 tensors
    (see `seg_lse_online`), so this always runs fp32, but that's a property of
    the callers, not a guarantee this kernel enforces.
    """
    i = tl.program_id(0)
    start = tl.load(indptr + i)
    end = tl.load(indptr + i + 1)
    p = tl.load(phi + i)

    m = -float("inf")
    s = 0.0
    for off in range(start, end, BLOCK):
        k = off + tl.arange(0, BLOCK)
        valid = k < end
        c = tl.load(colidx + k, mask=valid, other=0).to(tl.int32)
        v = tl.load(lam + k, mask=valid, other=0.0)
        q = tl.load(psi + c, mask=valid, other=0.0)
        x = tl.where(valid, v + p + q, -float("inf"))
        m_new = tl.maximum(m, tl.max(x))
        s = s * tl.exp(m - m_new) + tl.sum(tl.where(valid, tl.exp(x - m_new), 0.0))
        m = m_new

    lse = tl.where(s > 0.0, m + tl.log(s), -float("inf"))
    if SUB:
        lse = tl.load(base + i) - lse
    tl.store(out + i, lse)


def seg_lse(indptr, colidx, lam, phi, psi, n, out, base, block, num_warps):
    """Launch `_seg_lse_online_kernel`, one program per row, rows 0..n-1."""
    _seg_lse_online_kernel[(n,)](
        indptr, colidx, lam, phi, psi, out, base if base is not None else out,
        SUB=base is not None, BLOCK=block, num_warps=num_warps or 4,
    )
    return out


@triton.jit
def _cost_kernel(X, Y, ROWS, COLS, OUT, NNZ, D: tl.constexpr, BLOCK: tl.constexpr):
    """out[k] = ||x[rows[k]] - y[cols[k]]||^2, one pass, no (nnz, d) temporaries.

    Dtype: `acc` is a fixed float32 accumulator, independent of whatever dtype
    X/Y actually are -- this kernel is written for, and only tested against,
    float32 input. See `sparse_sqeuclidean_cost`'s dtype check, which is the
    enforcement point (this kernel itself has no way to check or error).
    """
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = k < NNZ
    r = tl.load(ROWS + k, mask=mask, other=0).to(tl.int64)
    c = tl.load(COLS + k, mask=mask, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in tl.static_range(D):
        xv = tl.load(X + r * D + j, mask=mask, other=0.0)
        yv = tl.load(Y + c * D + j, mask=mask, other=0.0)
        dv = xv - yv
        acc += dv * dv
    tl.store(OUT + k, acc, mask=mask)


def sqeuclidean_cost(x, y, rows, cols, block=1024):
    """Launch `_cost_kernel` over the nnz support entries."""
    nnz, d = rows.numel(), x.shape[1]
    out = torch.empty(nnz, device=x.device, dtype=torch.float32)
    _cost_kernel[(triton.cdiv(nnz, block),)](
        x.contiguous(), y.contiguous(), rows, cols, out, nnz,
        D=d, BLOCK=block, num_warps=4,
    )
    return out
