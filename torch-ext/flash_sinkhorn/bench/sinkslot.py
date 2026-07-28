"""SinkSLOT: fused-Triton sparse Sinkhorn on the unsmoothed sliced plan (gamma=0).

Ported from cai4cai/SLOT (bench/solvers/{sot,sinkslot,sinkslot_triton}.py), the
implementation accompanying the SLOT paper. SinkSLOT is SROT with the smoothing
weight set to zero: the reference plan P^SOT stays sparse, the Gibbs kernel
K = P^SOT (*) exp(-C/eps) inherits its zero pattern, and every Sinkhorn iteration
is restricted to that support -- O(L(N+M)) instead of O(NM). Because P^SOT lies
in Gamma(a, b) by construction, no row or column is ever empty.

This is the v5 variant: v1's alternating scheme with both reductions fused into
Triton kernels (online-softmax segmented LSE, CSR for the row half-step and CSC
for the column half-step). Mathematically identical to a plain torch segmented
LSE; the point is throughput, not a different algorithm. The CUDA-graph capture
of the upstream production path is intentionally omitted here.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl


def sot_directions(d: int, L: int, seed: int) -> np.ndarray:
    """Unit directions, matching build_sot_plan's RNG usage exactly."""
    rng = np.random.default_rng(seed)
    thetas = rng.standard_normal((L, d))
    norms = np.maximum(np.linalg.norm(thetas, axis=1, keepdims=True), 1e-300)
    return thetas / norms


def _ot_1d_coo(px: torch.Tensor, py: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """1-D optimal plan as (rows, cols, vals); at most n+m-1 nonzeros.

    North-west corner on the sorted order: the two cumulative-weight vectors cut
    [0,1] into segments, and each segment is a single (i, j) pair carrying its
    own length as mass.
    """
    ix = torch.argsort(px)
    iy = torch.argsort(py)
    ca = torch.cumsum(a[ix], 0)
    cb = torch.cumsum(b[iy], 0)

    bounds = torch.cat([ca, cb]).sort().values
    prev = torch.cat([bounds.new_zeros(1), bounds[:-1]])
    mass = bounds - prev
    keep = mass > 0
    mass, mid = mass[keep], (0.5 * (prev + bounds))[keep]

    i = torch.searchsorted(ca.contiguous(), mid).clamp_(max=ca.numel() - 1)
    j = torch.searchsorted(cb.contiguous(), mid).clamp_(max=cb.numel() - 1)
    return ix[i], iy[j], mass


def _ot_1d_coo_batched(PX: torch.Tensor, PY: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Vectorised `_ot_1d_coo` over C slices at once.

    PX: (n, C), PY: (m, C). Returns flat COO for all columns concatenated:
    (rows, cols, vals), each 1-D. Identical result to C sequential `_ot_1d_coo`
    calls; the Python loop over slices is removed.

    The breakpoints are the union of the two cumulative-weight vectors ca and cb.
    Both are already sorted (cumsums of positive weights), so the union is a MERGE
    of two sorted sequences, not a sort: each element's position in the merged
    order is its own index plus the count of the other sequence below it, from one
    `searchsorted`. That replaces an O((n+m) log(n+m)) sort per column with two
    binary searches, which dominates the plan build at large n.
    """
    n, C = PX.shape
    m = PY.shape[0]
    ix = torch.argsort(PX, dim=0)                     # (n, C)
    iy = torch.argsort(PY, dim=0)                     # (m, C)
    ca = torch.cumsum(a[ix], dim=0)                   # (n, C), sorted asc per col
    cb = torch.cumsum(b[iy], dim=0)                   # (m, C), sorted asc per col

    # Merge ca and cb into sorted `bounds` (n+m, C) via rank-scatter. searchsorted
    # is batched over columns with the (C, len) layout, then transposed back.
    caT, cbT = ca.T.contiguous(), cb.T.contiguous()   # (C, n), (C, m)
    # rank of ca[i] = i + #{cb < ca[i]}; rank of cb[j] = j + #{ca <= cb[j]}.
    # The asymmetric side (right for one, left for the other) breaks ties so the
    # two rank sets are a permutation of 0..n+m-1 with no collisions.
    rank_a = (torch.arange(n, device=PX.device)[None, :]
              + torch.searchsorted(cbT, caT, right=True)).T       # (n, C)
    rank_b = (torch.arange(m, device=PX.device)[None, :]
              + torch.searchsorted(caT, cbT, right=False)).T      # (m, C)

    bounds = PX.new_empty(n + m, C)
    bounds.scatter_(0, rank_a, ca)
    bounds.scatter_(0, rank_b, cb)

    prev = torch.cat([bounds.new_zeros(1, C), bounds[:-1]], dim=0)
    mass = bounds - prev
    mid = 0.5 * (prev + bounds)

    i = torch.searchsorted(caT, mid.T.contiguous()).clamp_(max=n - 1).T   # (n+m, C)
    j = torch.searchsorted(cbT, mid.T.contiguous()).clamp_(max=m - 1).T   # (n+m, C)
    R = torch.gather(ix, 0, i)
    Cc = torch.gather(iy, 0, j)

    keep = mass > 0
    return R[keep], Cc[keep], mass[keep]


def _ot_1d_coo_batched_cuda(PX: torch.Tensor, PY: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """CUDA-optimised `_ot_1d_coo_batched`: same plan, transposed layout, fp64 scan.

    This is the SinkSLOT-CUDA setup path. Identical construction to the naive
    version above, restructured for the GPU on two measured axes:

    * Layout. Everything runs in the TRANSPOSED (C, len) layout. Profiled at
      n=65536, L=200, the two cumsums were 42.4 ms of the naive function's
      55.8 ms -- ~5 GB/s, because `dim=0` on an (n, C) tensor is the strided scan
      path. The same scan along the contiguous last dim of (C, n) is 110x faster,
      and working in (C, ...) throughout removes the `mid.T` copies (105 MB each,
      twice) and leaves the gather indices contiguous.

    * Precision. With normalised weights ca runs 0->1, so an fp32 scan over n
      terms carries ~sqrt(n)*eps ~ 3e-5 of accumulated error while a typical
      segment mass is ~1/(n+m) ~ 7.6e-6. The rounding exceeds the masses it
      defines, so `mass > 0` -- the support -- depends on the scan's blocking.
      Against an fp64 reference plan the naive fp32 dim=0 scan disagreed on 1.27%
      of the support at n=16384 and 3.81% at n=32768; fp64 accumulation is exact
      to ~1e-16 relative, so the plan becomes layout- and blocking-independent.

    Net: 49.5x on the dominant stage AND a strictly more accurate plan. Because
    the support differs from the naive fp32 scan, SinkSLOT-CUDA keeps its own
    reference-cache namespace (see bench_forward.py).
    """
    n, C = PX.shape
    m = PY.shape[0]
    PXt, PYt = PX.T.contiguous(), PY.T.contiguous()    # (C, n), (C, m)
    ix = torch.argsort(PXt, dim=-1)                    # (C, n)
    iy = torch.argsort(PYt, dim=-1)                    # (C, m)
    ca = torch.cumsum(a[ix].double(), dim=-1).float()  # (C, n), sorted asc per row
    cb = torch.cumsum(b[iy].double(), dim=-1).float()  # (C, m), sorted asc per row

    # Merge ca and cb into sorted `bounds` (C, n+m) via rank-scatter.
    # rank of ca[i] = i + #{cb < ca[i]}; rank of cb[j] = j + #{ca <= cb[j]}.
    # The asymmetric side (right for one, left for the other) breaks ties so the
    # two rank sets are a permutation of 0..n+m-1 with no collisions.
    rank_a = (torch.arange(n, device=PX.device)[None, :]
              + torch.searchsorted(cb, ca, right=True))            # (C, n)
    rank_b = (torch.arange(m, device=PX.device)[None, :]
              + torch.searchsorted(ca, cb, right=False))           # (C, m)

    bounds = ca.new_empty(C, n + m)
    bounds.scatter_(1, rank_a, ca)
    bounds.scatter_(1, rank_b, cb)

    prev = torch.cat([bounds.new_zeros(C, 1), bounds[:, :-1]], dim=1)
    mass = bounds - prev
    mid = 0.5 * (prev + bounds)

    i = torch.searchsorted(ca, mid).clamp_(max=n - 1)   # (C, n+m), contiguous
    j = torch.searchsorted(cb, mid).clamp_(max=m - 1)   # (C, n+m), contiguous
    R = torch.gather(ix, 1, i)
    Cc = torch.gather(iy, 1, j)

    # Take the mask's compaction index once and gather with it rather than masking
    # R/Cc/mass separately (which runs the compaction scan three times): 3.27 ->
    # 1.24 ms at n=65536, L=200, bit-identical output. Flattens slice-major, but
    # the caller coalesces by key immediately so emission order is immaterial.
    sel = (mass > 0).reshape(-1).nonzero(as_tuple=False).squeeze(1)
    return R.reshape(-1)[sel], Cc.reshape(-1)[sel], mass.reshape(-1)[sel]


def sot_plan_coo(
    X: torch.Tensor, Y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    L: int, seed: int, chunk: int = None, ot1d=_ot_1d_coo_batched,
):
    """Unsmoothed SOT plan as COO: (rows, cols, vals), nnz <= L(N+M).

    Same construction as sot_plan_dense but never allocates N x M. The gamma
    blend is deliberately absent -- gamma * (a (x) b) is rank-one and separable,
    so the caller folds it into the potentials rather than materialising it.

    Built in one shot by default (chunk = L): all L directions are projected, one
    batched merge produces the raw entries, and a single coalesce sums them. The
    per-chunk coalesce of the original design re-sorted the accumulating support
    once per chunk, which dominated the build; a single coalesce over the full raw
    set is far cheaper, and the transient (~70 MiB at n=10000, L=100) is negligible
    at the sizes benchmarked here. `chunk` stays a memory valve for pathological n.

    The flat key `row * m + col` is int32 whenever n*m fits, which halves both
    the key array and the sort workspace inside `unique`.

    `ot1d` selects the per-slice 1-D OT builder: the naive `_ot_1d_coo_batched`
    (the SinkSLOT baseline) or `_ot_1d_coo_batched_cuda` (the SinkSLOT-CUDA fp64
    path). Only the plan differs; the coalesce is identical.
    """
    n, d = X.shape
    m = Y.shape[0]
    thetas = torch.as_tensor(sot_directions(d, L, seed), dtype=X.dtype, device=X.device)
    key_dtype = torch.int32 if n * m < 2**31 else torch.int64

    def coalesce(flat, vals):
        """Group-sum by key, without materialising an int64 inverse.

        `torch.unique(return_inverse=True)` hands back an int64 index per entry,
        which at this size is the single largest array in the build. `unique`
        returns its keys sorted, so `searchsorted` recovers the same map in
        int32 for one extra pass -- 1.56x off the peak, measured.
        """
        uniq = torch.unique(flat)
        inv = torch.searchsorted(uniq, flat, out_int32=True)
        return uniq, vals.new_zeros(uniq.numel()).index_add_(0, inv, vals)

    # Chunk size barely moves the peak -- that is set by the final merge against
    # the accumulated support, not by the chunk -- so this is picked to keep the
    # per-chunk projection small rather than to tune the merge.
    # One shot: project all L directions, one batched merge, one coalesce. The
    # upstream per-chunk coalesce re-sorted the accumulating support once per
    # chunk (~7x at L=100), which dominated the build; a single coalesce over the
    # full raw set is ~5x faster and, at the sizes benchmarked here, the transient
    # (~70 MiB at n=10000, L=100) is negligible. `chunk` remains a memory valve
    # for pathological n, defaulting to one shot.
    chunk = L if chunk is None else chunk
    run_f = run_v = None
    for start in range(0, L, chunk):
        th = thetas[start:start + chunk]
        PX, PY = X @ th.T, Y @ th.T
        r, c, v = ot1d(PX, PY, a, b)
        del PX, PY
        f = (r * m + c).to(key_dtype)
        if run_f is not None:
            f, v = torch.cat([run_f, f]), torch.cat([run_v, v])
        run_f, run_v = f, v
    run_f, run_v = coalesce(run_f, run_v)

    # Divided once at the end rather than per slice: same value, one pass, and
    # it keeps the running accumulator on the same scale as `_ot_1d_coo` returns.
    return (run_f // m).long(), (run_f % m).long(), run_v / L


def to_csr(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, n: int,
           narrow_key: bool = False):
    """COO -> CSR. Returns (indptr, colidx, vals_permuted, perm).

    `perm` is kept so caller-side per-entry arrays (cost, plan values) can be
    reordered into the same layout, and results mapped back.

    The sort is STABLE on purpose. `sot_plan_coo` returns its entries ordered by
    the flat index `row * m + col`, so they already arrive sorted by column
    within each row; an unstable sort is free to scramble that, and the inner
    loop's cost is the gather `psi[colidx[k]]`, whose locality is exactly that
    ordering. For the CSC build (axes swapped) the same argument makes rows
    ascending within each column.

    `narrow_key` (SinkSLOT-CUDA path) casts the CSC sort key to int32 when `n`
    fits: the key is a row index bounded by `n`, so a 32-bit radix pass suffices
    and the int64 sort was scanning 64 bits for nothing -- 4.96 -> 2.24 ms at
    nnz=26M, same permutation (stability makes it exact, not merely equivalent).
    """
    # `sot_plan_coo` coalesces on the flat key `row * m + col` and returns it
    # sorted, so for the CSR build `rows` is already non-decreasing and the
    # permutation is the identity. Detecting that skips an int64 argsort of nnz
    # (7.9 MiB at nnz=989k) and lets the values be aliased rather than gathered.
    # The CSC build (axes swapped) still needs the real sort.
    if bool(torch.all(rows[1:] >= rows[:-1])):
        perm, r = None, rows
    else:
        key = rows.to(torch.int32) if (narrow_key and n < 2 ** 31) else rows
        perm = torch.argsort(key, stable=True)
        r = rows[perm]
    counts = torch.bincount(r, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=rows.device)
    indptr[1:] = torch.cumsum(counts, 0)

    # int16 indices whenever the gathered axis fits. At large n the loop is
    # bandwidth-bound, and the two streamed arrays are the index (4 B) and lam
    # (4 B); lam cannot be narrowed -- an fp16 mantissa puts ~5e-3 absolute error
    # on a log-domain value of order 10, against a 1e-6 marginal threshold -- but
    # the index can, which takes 8 B/nnz to 6 B.
    idx = cols if perm is None else cols[perm]
    return indptr, idx.to(torch.int16 if int(idx.max()) < 2**15 else torch.int32), \
        (vals if perm is None else vals[perm]), perm


@triton.jit
def _cost_kernel(X, Y, ROWS, COLS, OUT, NNZ, D: tl.constexpr, BLOCK: tl.constexpr):
    """out[k] = ||x[rows[k]] - y[cols[k]]||^2, one pass, no (nnz, d) temporaries."""
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


def sparse_sqeuclidean_cost(x, y, rows, cols, block: int = 1024):
    """Squared-euclidean cost on the sparse support, fused (SinkSLOT-CUDA path).

    The obvious `(x[rows] - y[cols]).square().sum(1)` materialises three (nnz, d)
    intermediates to produce one float per entry -- 832 MB each at nnz=26M, d=8,
    against 104 MB of output. Measured 9-10x faster fused across n=16k..65k, and
    it is the difference between the cost stage being 27.8 ms and 3.1 ms at
    n=65536, L=200.

    Agrees with the torch expression to ~3e-7 relative (fp32 reassociation of the
    d-term sum); the resulting shift in the dual objective is below 2e-7.
    """
    nnz, d = rows.numel(), x.shape[1]
    out = torch.empty(nnz, device=x.device, dtype=torch.float32)
    _cost_kernel[(triton.cdiv(nnz, block),)](
        x.contiguous(), y.contiguous(), rows, cols, out, nnz,
        D=d, BLOCK=block, num_warps=4,
    )
    return out


@triton.jit
def _seg_lse_kernel(
    indptr, colidx, lam, phi, psi, shift, out,
    USE_SHIFT: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[i] = LSE_{k in row i} ( lam[k] + phi[i] + psi[col[k]] ).

    With USE_SHIFT the caller supplies the per-row shift (the v2/v3 propagated
    bound); otherwise the row max is computed in a first pass, which is what
    makes this also serve as the v1 stable-LSE path.
    """
    i = tl.program_id(0)
    start = tl.load(indptr + i)
    end = tl.load(indptr + i + 1)
    p = tl.load(phi + i)

    if USE_SHIFT:
        s = tl.load(shift + i)
    else:
        s = -float("inf")
        for off in range(start, end, BLOCK):
            k = off + tl.arange(0, BLOCK)
            m = k < end
            c = tl.load(colidx + k, mask=m, other=0).to(tl.int32)
            v = tl.load(lam + k, mask=m, other=-float("inf"))
            q = tl.load(psi + c, mask=m, other=0.0)
            s = tl.maximum(s, tl.max(tl.where(m, v + p + q, -float("inf"))))

    acc = 0.0
    for off in range(start, end, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < end
        c = tl.load(colidx + k, mask=m, other=0).to(tl.int32)
        v = tl.load(lam + k, mask=m, other=0.0)
        q = tl.load(psi + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, tl.exp(v + p + q - s), 0.0))

    tl.store(out + i, tl.where(acc > 0.0, s + tl.log(acc), -float("inf")))


def seg_lse(indptr, colidx, lam, phi, psi, n, shift=None, block=128):
    """Fused segmented LSE over M = lam + phi[row] + psi[col], never forming M."""
    out = torch.empty(n, dtype=lam.dtype, device=lam.device)
    dummy = shift if shift is not None else out
    _seg_lse_kernel[(n,)](
        indptr, colidx, lam, phi, psi, dummy, out,
        USE_SHIFT=shift is not None, BLOCK=block, num_warps=4,
    )
    return out


@triton.jit
def _plan_kernel(indptr, colidx, lam, phi, psi, out, BLOCK: tl.constexpr):
    """out[k] = exp(lam[k] + phi[i] + psi[col[k]]) -- the final plan values."""
    i = tl.program_id(0)
    start = tl.load(indptr + i)
    end = tl.load(indptr + i + 1)
    p = tl.load(phi + i)
    for off in range(start, end, BLOCK):
        k = off + tl.arange(0, BLOCK)
        m = k < end
        c = tl.load(colidx + k, mask=m, other=0).to(tl.int32)
        v = tl.load(lam + k, mask=m, other=0.0)
        q = tl.load(psi + c, mask=m, other=0.0)
        tl.store(out + k, tl.exp(v + p + q), mask=m)


def plan_values(indptr, colidx, lam, phi, psi, nnz, n, block=128):
    out = torch.empty(nnz, dtype=lam.dtype, device=lam.device)
    _plan_kernel[(n,)](indptr, colidx, lam, phi, psi, out, BLOCK=block, num_warps=4)
    return out


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


def launch_cfg(nnz: int, n: int) -> tuple[int, int]:
    """(BLOCK, num_warps) for a mean row length of nnz/n.

    One program owns one row, so BLOCK is the row-tile width. A fixed 128 is
    wrong at both ends: with L=16 the mean row holds ~15 entries and most of
    each tile is masked-off waste, while with L=512 the row needs several
    sequential tiles that a wider one would retire at once. Rounding the mean
    row length to a power of two and matching one warp per 32 lanes tracks both.

    Bounds are 32 (a warp) and 1024 (register pressure past that costs more in
    occupancy than it saves in trips).

    num_warps is 2, NOT one warp per 32 lanes. Matching warps to BLOCK is the
    obvious rule and it is 1.25--1.33x slower here across every size measured:
    one program owns one row, so a narrow program leaves more of them resident
    per SM, and the extra elements per thread cost nothing because the kernel is
    waiting on memory anyway. Only past BLOCK=512 does the register pressure of
    16 elements per thread start to bite.
    """
    mean_row = max(1, nnz // max(1, n))
    block = 1 << max(5, min(10, (mean_row - 1).bit_length()))
    return block, 2 if block <= 512 else 4


def seg_lse_online(indptr, colidx, lam, phi, psi, n, block=None, num_warps=None,
                   base=None, out=None):
    """Segmented LSE; `base - LSE` when `base` is given, into `out` if supplied.

    `out` lets the caller reuse a buffer across iterations instead of allocating
    an n-vector per half-step -- another launch's worth of work the allocator
    would otherwise do inside the loop.
    """
    if block is None:
        block, num_warps = launch_cfg(colidx.numel(), n)
    if out is None:
        out = torch.empty(n, dtype=lam.dtype, device=lam.device)
    _seg_lse_online_kernel[(n,)](
        indptr, colidx, lam, phi, psi, out, base if base is not None else out,
        SUB=base is not None, BLOCK=block, num_warps=num_warps or 4,
    )
    return out


# --------------------------------------------------------------------------
# v5 loop and entry point
# --------------------------------------------------------------------------


def _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
            n_iters, stop=None, eps=None):
    """v5: alternating fused half-steps over prebuilt CSR/CSC.

    Potentials are absorbed (phi = f/eps, psi = g/eps): lam already carries
    log P^SOT - C/eps, so seg_lse_online with base=log_a returns log_a - LSE
    directly, which is the next phi. The excluded potential is zeroed on its own
    axis (z_n, z_m).

    Returns (phi, psi, iters_run, converged, final_viol). With `stop` None or
    stop.mode == "fixed" it runs exactly n_iters (converged/final_viol None).
    With stop.mode in {"marginal", "potential"} it runs up to stop.max_iter and
    stops on the max (L-infinity) marginal violation: after the column half-step
    the column marginals are exactly b, so only the row marginal deviates, and
    r = a * exp(phi - phi_next) where phi_next is the next row LSE -- one extra
    LSE, no O(nnz) work. (potential mode falls back to the marginal check here;
    the u/v-change rule is Spar-Sink's and is applied there.) This is max, not a
    total-variation sum: matches the SLOT repo's actual working "marg_viol" rule
    (bench/solvers/sinkslot.py's `_violation`/`_run_v5` there) -- a sum over n
    terms against a fixed absolute tol is unreachable at n=10,000 regardless of
    convergence, which is why an earlier version of this function (sum-based)
    looked like marginal-violation stopping didn't work here either.

    stop.mode == "potential_linf" reproduces FlashSinkhorn's own native rule:
    stop once the dual potentials stop moving, max(|Δf|, |Δg|) < stop.tol since
    the last check, with no extra LSE call (unlike marginal mode). Since phi, psi
    are absorbed (phi=f/eps), the raw phi/psi change is rescaled by `eps`
    (required in this mode) so stop.tol compares on the same physical scale as
    FlashSinkhorn's and SROT's unabsorbed f, g.
    """
    r_blk, r_w = launch_cfg(r_idx.numel(), n)
    c_blk, c_w = launch_cfg(c_idx.numel(), m)
    phi, psi = torch.zeros_like(log_a), torch.zeros_like(log_b)
    z_n, z_m = torch.zeros_like(log_a), torch.zeros_like(log_b)

    mode = getattr(stop, "mode", "fixed") if stop is not None else "fixed"

    if mode == "fixed":
        for _ in range(n_iters):
            seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi)
            seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi)
        return phi, psi, n_iters, None, None

    if mode == "potential_linf":
        if eps is None:
            raise ValueError("eps is required for stop.mode == 'potential_linf'")
        prev_phi, prev_psi = phi.clone(), psi.clone()
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi)
            seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi)
            it += 1
            if it % stop.check_every == 0:
                change = eps * max((phi - prev_phi).abs().max().item(),
                                    (psi - prev_psi).abs().max().item())
                if change < stop.tol:
                    converged = True
                    break
                prev_phi.copy_(phi)
                prev_psi.copy_(psi)
        return phi, psi, it, converged, change

    a = log_a.exp()
    phi_next = torch.empty_like(log_a)
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi)
        seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi)
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi_next)
            row_marg = a * (phi - phi_next).exp()          # col marginal is exactly b
            # max (L-infinity), matching SLOT's actual _run_v5 exactly:
            # `if float((a * ((phi - phi_new).exp() - 1.0).abs()).max()) <= threshold`.
            # Not gated on mass_tol either -- SLOT's own working rule doesn't
            # check mass separately; still returned for the diagnostic column.
            viol = float((row_marg - a).abs().max())
            mass = float(row_marg.sum())
            if viol <= stop.tol:
                converged = True
                break
    return phi, psi, it, converged, viol


def build_support(x, y, a, b, eps, L, seed=0):
    """(rows, cols, lam) for the sinkslot kernel: lam = log P^SOT - C/eps.

    Built on the sparse SOT support only, so this is O(L(N+M)), not O(NM).
    """
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=seed)
    cost = (x[rows] - y[cols]).square().sum(1)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    return rows, cols, lam


def run_sinkslot(x, y, a, b, eps, L, n_iters, seed=0):
    """Fixed-iteration SinkSLOT (v5). Returns (phi, psi, rows, cols, lam).

    phi, psi are the ABSORBED potentials (f/eps, g/eps); the dual objective is
    eps * (a . phi + b . psi).
    """
    n, m = a.numel(), b.numel()
    rows, cols, lam = build_support(x, y, a, b, eps, L, seed=seed)
    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
    phi, psi, _, _, _ = _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                a.log(), b.log(), n, m, n_iters)
    return phi, psi, rows, cols, lam
