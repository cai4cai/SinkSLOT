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


def sot_plan_coo(
    X: torch.Tensor, Y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    L: int, seed: int,
):
    """Unsmoothed SOT plan as COO: (rows, cols, vals), nnz <= L(N+M).

    Same construction as sot_plan_dense but never allocates N x M. The gamma
    blend is deliberately absent -- gamma * (a (x) b) is rank-one and separable,
    so the caller folds it into the potentials rather than materialising it.

    Coalescing is done in CHUNKS rather than over all L slices at once. Building
    the whole L(N+M) list first is what sets the peak of the entire solve: at
    n=m=10000, L=100 it is ~2M entries against a coalesced support of 989k, and
    measured end to end the construction peaked at 82.3 MiB while the Sinkhorn
    loop that follows adds 1.7 MiB. Folding each chunk into the running support
    keeps the transient proportional to the chunk, not to L.

    The flat key `row * m + col` is int32 whenever n*m fits, which halves both
    the key array and the sort workspace inside `unique`.
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
    chunk = max(1, min(L, 16))
    run_f = run_v = None
    for start in range(0, L, chunk):
        # Project only this chunk's directions. The full n x L projection is
        # 8 MiB at n=10000, L=100, and every slice is used exactly once.
        th = thetas[start:start + chunk]
        PX, PY = X @ th.T, Y @ th.T
        fs, vs = [], []
        for ell in range(PX.shape[1]):
            r, c, v = _ot_1d_coo(PX[:, ell], PY[:, ell], a, b)
            fs.append((r * m + c).to(key_dtype))
            vs.append(v)
        del PX, PY
        f, v = torch.cat(fs), torch.cat(vs)
        del fs, vs
        if run_f is not None:
            f, v = torch.cat([run_f, f]), torch.cat([run_v, v])
            run_f = run_v = None
        run_f, run_v = coalesce(f, v)
        del f, v

    # Divided once at the end rather than per slice: same value, one pass, and
    # it keeps the running accumulator on the same scale as `_ot_1d_coo` returns.
    return (run_f // m).long(), (run_f % m).long(), run_v / L


def to_csr(rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, n: int):
    """COO -> CSR. Returns (indptr, colidx, vals_permuted, perm).

    `perm` is kept so caller-side per-entry arrays (cost, plan values) can be
    reordered into the same layout, and results mapped back.

    The sort is STABLE on purpose. `sot_plan_coo` returns its entries ordered by
    the flat index `row * m + col`, so they already arrive sorted by column
    within each row; an unstable sort is free to scramble that, and the inner
    loop's cost is the gather `psi[colidx[k]]`, whose locality is exactly that
    ordering. For the CSC build (axes swapped) the same argument makes rows
    ascending within each column.
    """
    # `sot_plan_coo` coalesces on the flat key `row * m + col` and returns it
    # sorted, so for the CSR build `rows` is already non-decreasing and the
    # permutation is the identity. Detecting that skips an int64 argsort of nnz
    # (7.9 MiB at nnz=989k) and lets the values be aliased rather than gathered.
    # The CSC build (axes swapped) still needs the real sort.
    if bool(torch.all(rows[1:] >= rows[:-1])):
        perm, r = None, rows
    else:
        perm = torch.argsort(rows, stable=True)
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


def _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iters):
    """Fixed-iteration v5: alternating fused half-steps over prebuilt CSR/CSC.

    Potentials are absorbed (phi = f/eps, psi = g/eps): lam already carries
    log P^SOT - C/eps, so seg_lse_online with base=log_a returns log_a - LSE
    directly, which is the next phi. The excluded potential is zeroed on its own
    axis (z_n, z_m). No stopping rule -- the benchmark compares equal work.
    """
    r_blk, r_w = launch_cfg(r_idx.numel(), n)
    c_blk, c_w = launch_cfg(c_idx.numel(), m)
    phi, psi = torch.zeros_like(log_a), torch.zeros_like(log_b)
    z_n, z_m = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(n_iters):
        seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi)
        seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi)
    return phi, psi


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
    phi, psi = _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                       a.log(), b.log(), n, m, n_iters)
    return phi, psi, rows, cols, lam
