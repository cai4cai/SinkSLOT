"""Metal fused kernels (Apple silicon): segmented online LSE and sparse sq-euclidean cost.

The Apple-silicon counterpart of `_triton.py`, with the same two entry points
(see `_backends.py`). The kernels ship as Metal source and are compiled once
per process with `torch.mps.compile_shader`, so there is no native build step
and nothing beyond what PyTorch's MPS backend already needs.

Numerics follow the Triton kernels: float32 throughout, and the precise
`exp`/`log` rather than Metal's fast-math defaults, since the stopping rules
compare potential changes against tolerances near float32 resolution.
"""

from __future__ import annotations

import functools

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Merge two online-LSE states (m, s): the sum of exp over both, relative to the larger max.
inline void lse_merge(thread float& m, thread float& s, float m2, float s2) {
    float mn = max(m, m2);
    if (mn == -INFINITY) return;
    s = s * precise::exp(m - mn) + s2 * precise::exp(m2 - mn);
    m = mn;
}

// One threadgroup per CSR row: a strided loop keeps a per-thread online (m, s) pair,
// simd_max/simd_sum combine a simdgroup, and threadgroup memory combines simdgroups.
// Same rescale as the Triton kernel, s' = s * exp(m - m') + exp(x - m'), but applied per
// element: exactly one exp per element either way.
template <typename IT>
kernel void seg_lse(
    device const int*   indptr [[buffer(0)]],
    device const IT*    colidx [[buffer(1)]],
    device const float* lam    [[buffer(2)]],
    device const float* phi    [[buffer(3)]],
    device const float* psi    [[buffer(4)]],
    device float*       out    [[buffer(5)]],
    device const float* base   [[buffer(6)]],
    constant long&      sub    [[buffer(7)]],   // store base[i] - LSE rather than LSE
    uint2 group  [[threadgroup_position_in_grid]],
    uint2 tid2   [[thread_position_in_threadgroup]],
    uint2 tpg    [[threads_per_threadgroup]],
    uint  lane   [[thread_index_in_simdgroup]],
    uint  simd   [[simdgroup_index_in_threadgroup]],
    uint  nsimd  [[simdgroups_per_threadgroup]])
{
    // Rows run along y so that n is bounded by the grid's per-dimension limit, not n * group size
    const uint row = group.y, tid = tid2.x;
    const int start = indptr[row], end = indptr[row + 1];
    const float p = phi[row];

    float m = -INFINITY, s = 0.0f;
    for (int k = start + (int)tid; k < end; k += (int)tpg.x) {
        float x = lam[k] + p + psi[(int)colidx[k]];
        if (x > m) {
            s = s * precise::exp(m - x) + 1.0f;
            m = x;
        } else if (x > -INFINITY) {
            s += precise::exp(x - m);
        }
    }

    float gm = simd_max(m);
    float gs = simd_sum(m == -INFINITY ? 0.0f : s * precise::exp(m - gm));

    threadgroup float tm[32], ts[32];
    if (nsimd > 1) {
        if (lane == 0) { tm[simd] = gm; ts[simd] = gs; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd != 0) return;
        m = lane < nsimd ? tm[lane] : -INFINITY;
        s = lane < nsimd ? ts[lane] : 0.0f;
        gm = simd_max(m);
        gs = simd_sum(m == -INFINITY ? 0.0f : s * precise::exp(m - gm));
    }
    if (tid != 0) return;

    float lse = gs > 0.0f ? gm + precise::log(gs) : -INFINITY;
    out[row] = sub ? base[row] - lse : lse;
}

template [[host_name("seg_lse_i16")]] kernel void seg_lse<short>(
    device const int*, device const short*, device const float*, device const float*,
    device const float*, device float*, device const float*, constant long&,
    uint2, uint2, uint2, uint, uint, uint);
template [[host_name("seg_lse_i32")]] kernel void seg_lse<int>(
    device const int*, device const int*, device const float*, device const float*,
    device const float*, device float*, device const float*, constant long&,
    uint2, uint2, uint2, uint, uint, uint);

// out[k] = ||x[rows[k]] - y[cols[k]]||^2 with a float32 accumulator, one thread per nonzero.
kernel void sqeuclidean_cost(
    device const float* x    [[buffer(0)]],
    device const float* y    [[buffer(1)]],
    device const long*  rows [[buffer(2)]],
    device const long*  cols [[buffer(3)]],
    device float*       out  [[buffer(4)]],
    constant long&      nnz  [[buffer(5)]],
    constant long&      d    [[buffer(6)]],
    uint k [[thread_position_in_grid]])
{
    if ((long)k >= nnz) return;
    device const float* xr = x + rows[k] * d;
    device const float* yc = y + cols[k] * d;
    float acc = 0.0f;
    for (long j = 0; j < d; ++j) {
        float dv = xr[j] - yc[j];
        acc += dv * dv;
    }
    out[k] = acc;
}
"""

_SEG_LSE = {torch.int16: "seg_lse_i16", torch.int32: "seg_lse_i32"}

# Threadgroup-size cap for `seg_lse`: the Metal counterpart of `launch_cfg`'s
# fixed num_warps=2. Measured on an M-series GPU (alternating loop, d=3,
# n=1k..50k, L=16..256), one 32-64-thread group per row beats a group as wide
# as the row tile at every size: 2.7x at n=50k, L=16 and still 1.05x at L=256.
# Narrow groups leave more rows resident per core, and the kernel is waiting
# on memory anyway, so the extra elements per thread cost nothing.
_MAX_GROUP = 64


@functools.cache
def _library():
    return torch.mps.compile_shader(_SOURCE)


def seg_lse(indptr, colidx, lam, phi, psi, out, base, block, num_warps=None):
    """Launch `seg_lse` with one `block`-thread threadgroup per row of `out`.

    `block` is `launch_cfg`'s row-tile width, used here as the threadgroup
    size up to `_MAX_GROUP`; `num_warps` is a Triton knob with no Metal
    counterpart and is ignored.
    """
    if colidx.dtype not in _SEG_LSE:
        raise ValueError(f"colidx must be int16 or int32, got {colidx.dtype}")
    kernel = getattr(_library(), _SEG_LSE[colidx.dtype])
    group = min(block, _MAX_GROUP, kernel.max_threads_per_threadgroup)
    kernel(
        indptr, colidx, lam, phi, psi, out, base if base is not None else out,
        int(base is not None), threads=(group, out.numel()), group_size=(group, 1),
    )
    return out


def sqeuclidean_cost(x, y, rows, cols):
    """Launch `sqeuclidean_cost`, one thread per support entry."""
    nnz, d = rows.numel(), x.shape[1]
    out = torch.empty(nnz, device=x.device, dtype=torch.float32)
    if nnz:
        _library().sqeuclidean_cost(
            x.contiguous(), y.contiguous(), rows.long(), cols.long(), out, nnz, d,
            threads=nnz,
        )
    return out
