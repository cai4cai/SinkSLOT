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

`sinkslot_solve` is the device-agnostic entry point: the Triton kernels above
when Triton is installed and the input is CUDA, a pure-torch fallback
(`_run_v5_torch`) otherwise -- same algorithm, cross-checked in
testing/test_sinkslot_bench.py, so a CPU machine or a machine without Triton
installed can still run SinkSLOT, just without the fused-kernel throughput.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def sot_directions(d: int, L: int, seed: int) -> torch.Tensor:
    """Unit directions: standard-normal draws, L2-normalised.

    Same construction as vendor/sinkhorn_methods.py's build_sot_plan, but on
    torch's own RNG rather than numpy's, so the two no longer agree bit-for-bit
    on a shared seed (different generator, same distribution).

    Dtype: there's no tensor input to take a dtype from (d, L, seed are plain
    Python ints), so this always returns float64, on purpose, for an accurate
    normalisation regardless of what dtype the caller works in. `sot_plan_coo`
    (the only caller) immediately casts the result down to `X`'s dtype -- if
    you add a new caller, do the same, or you'll end up carrying float64
    directions through an otherwise float32 pipeline.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    thetas = torch.randn((L, d), generator=gen, dtype=torch.float64)
    norms = thetas.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return thetas / norms


def _ot_1d_coo_batched(PX: torch.Tensor, PY: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """1-D optimal plans as (rows, cols, vals), C slices at once.

    North-west corner on the sorted order: the two cumulative-weight vectors cut
    [0,1] into segments, and each segment is a single (i, j) pair carrying its
    own length as mass. At most n+m-1 nonzeros per slice.

    PX: (n, C), PY: (m, C). Returns flat COO for all columns concatenated:
    (rows, cols, vals), each 1-D.

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

    Dtype: the `.double()` upcast below is internal and fixed, not driven by
    the caller's dtype -- `a`, `b`, `PX`, `PY` can be float32 or float64 on the
    way in, the cumsum always runs in float64, and `ca`/`cb` (and everything
    returned) are always float32 on the way out. Passing float64 inputs does
    not get you a float64 plan; it only feeds float64 values into a scan that
    was going to run in float64 either way.
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

    Dtype: with `ot1d=_ot_1d_coo_batched_cuda`, the returned `vals` are always
    float32, whatever dtype `X`/`Y`/`a`/`b` are (see that function's own
    docstring) -- passing float64 in does not get you a float64 plan. With the
    naive `_ot_1d_coo_batched` there's no such ceiling; the output follows the
    input dtype. Either way, there is no way to get a float64-precision plan
    on a CUDA tensor: `gradient_flow/finite_diff.py`'s `build_support`, which
    needs float64 for everything downstream, doesn't try -- it explicitly
    passes `.float()` inputs to whichever builder the device selects, accepts
    the float32-quality plan either way, and only upcasts the *returned*
    tensor to float64 afterward so the rest of its pipeline has a consistent
    dtype. That's a container cast, not recovered precision.
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
    # it keeps the running accumulator on the per-slice mass scale throughout.
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


if _HAS_TRITON:
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


def sparse_sqeuclidean_cost(x, y, rows, cols, block: int = 1024, use_triton=None):
    """Squared-euclidean cost on the sparse support.

    Fused Triton kernel when `x` is CUDA and Triton is installed (the
    SinkSLOT-CUDA path). Otherwise falls back to the obvious torch expression,
    which is what the fused kernel is optimised away from in the first place --
    same formula, just three (nnz, d) intermediates instead of one fused pass.
    Only the fused path was ever measured for speed; the fallback exists so the
    function (and everything built on it) also works without a CUDA GPU or
    without Triton installed at all, per #10.

    `use_triton`: None (default) auto-detects from `_HAS_TRITON and x.is_cuda`.
    Pass explicitly to override -- e.g. `False` to force the torch fallback on
    a CUDA tensor even though Triton is available, which `sinkslot_solve`'s own
    `backend="torch"` needs (auto-detection alone can't express "torch path,
    but on GPU": that combination is CUDA-true, Triton-available-true, and
    auto-detection always picks Triton there). `True` raises if Triton isn't
    actually usable, rather than silently falling back.

    Fused-path notes: materialising the (nnz, d) intermediates costs 832 MB each
    at nnz=26M, d=8, against 104 MB of output. Measured 9-10x faster fused across
    n=16k..65k, and it is the difference between the cost stage being 27.8 ms and
    3.1 ms at n=65536, L=200. Agrees with the torch expression to ~3e-7 relative
    (fp32 reassociation of the d-term sum); the resulting shift in the dual
    objective is below 2e-7.

    Dtype: the fused path always returns float32, matching `_cost_kernel`'s
    fixed float32 accumulator -- not inferred from `x`/`y`, and checked below
    rather than left implicit, since Triton has no way to promote a mismatched
    accumulator and would otherwise return quietly-wrong numbers. The torch
    fallback (`use_triton=False`) has no such restriction: it follows `x`/`y`'s
    own dtype, since it's the plain `(x[rows]-y[cols]).square().sum(1)`
    expression with nothing fixed-precision underneath.
    """
    if use_triton is None:
        use_triton = _HAS_TRITON and x.is_cuda
    elif use_triton and not _HAS_TRITON:
        raise ValueError("use_triton=True but Triton isn't importable")
    if not use_triton:
        return (x[rows] - y[cols]).square().sum(1)
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError(
            f"sparse_sqeuclidean_cost's Triton path is float32-only (the "
            f"kernel's accumulator is a fixed float32); got x.dtype={x.dtype}, "
            f"y.dtype={y.dtype}. Cast to float32 first, or pass use_triton=False."
        )
    nnz, d = rows.numel(), x.shape[1]
    out = torch.empty(nnz, device=x.device, dtype=torch.float32)
    _cost_kernel[(triton.cdiv(nnz, block),)](
        x.contiguous(), y.contiguous(), rows, cols, out, nnz,
        D=d, BLOCK=block, num_warps=4,
    )
    return out


if _HAS_TRITON:
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
        `_cost_kernel`'s accumulator above, there's no `tl.zeros(..., dtype=...)`
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

    Triton-only (the fused CSR kernel). For a device- or Triton-agnostic solve,
    use `sinkslot_solve` or `_run_v5_torch`, which use `_seg_lse_coo` instead.
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "seg_lse_online needs Triton (pip install triton); "
            "use sinkslot_solve() for a solve that works without it"
        )
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


_STOP_MODES = ("fixed", "marginal", "potential", "potential_linf")


def _resolve_stop_mode(stop):
    """Shared by `_run_v5` and `_run_v5_torch`: resolve `stop.mode` (or "fixed"
    if `stop` is None) and validate it against `_STOP_MODES` upfront, so the
    four valid modes stay one source of truth instead of two independently
    -maintained checks. They drifted out of sync once already -- `_run_v5_torch`
    validated from the start, `_run_v5` didn't, so a typo'd mode used to behave
    differently depending on which backend happened to run it (fixed alongside
    this helper, not by it: the old inline check ran only after the "fixed"
    and "potential_linf" branches had already been ruled out, but for an
    invalid mode neither of those branches matches anyway, so validating here
    instead, before any branch runs, raises in the exact same cases as before).
    """
    mode = getattr(stop, "mode", "fixed") if stop is not None else "fixed"
    if mode not in _STOP_MODES:
        raise ValueError(
            f"unknown stop.mode {mode!r}; expected one of "
            f"'fixed', 'marginal', 'potential', 'potential_linf'"
        )
    return mode


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

    mode = _resolve_stop_mode(stop)

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

    # mode in {"marginal", "potential"} -- the only two left after _resolve_stop_mode.
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
            # phi_next is NOT a redundant recompute of phi: phi used the psi from
            # the previous iteration, phi_next uses the psi just updated two lines
            # up, so it's what phi would be after one more row half-step -- the
            # one-step-ahead value the row-marginal-violation formula below needs.
            # Storing the previous phi instead would answer a different question
            # (how much phi itself moved), not the row marginal's actual deviation
            # from a. See the docstring above for the r = a*exp(phi - phi_next)
            # identity this relies on.
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


# --------------------------------------------------------------------------
# Pure-torch fallback (#10) -- no Triton, works on CPU or a non-CUDA device.
# --------------------------------------------------------------------------


def _seg_lse_coo(vals, idx, size):
    """Segmented log-sum-exp over COO-style per-entry indices.

    out[i] = logsumexp_{k: idx[k] == i} vals[k]; -inf for an empty group. Same
    scatter_reduce/index_add pattern as `test_run_v5_matches_plain_torch_
    segmented_lse` in testing/test_sinkslot_bench.py, which validates this
    matches `_seg_lse_online_kernel`'s fp32 output on the real Triton path.
    """
    mx = vals.new_full((size,), float("-inf")).scatter_reduce(
        0, idx, vals, reduce="amax", include_self=True)
    acc = vals.new_zeros(size).index_add_(0, idx, (vals - mx[idx]).exp())
    return torch.where(acc > 0, mx + acc.clamp_min(torch.finfo(vals.dtype).tiny).log(),
                        torch.full_like(mx, float("-inf")))


def _run_v5_torch(rows, cols, lam, log_a, log_b, n, m, n_iters, stop=None, eps=None):
    """Pure-torch counterpart to `_run_v5` -- same four `stop.mode` semantics, no
    Triton, no CSR/CSC (operates directly on the COO `sot_plan_coo` returns,
    since `index_add_`/`scatter_reduce_` don't need sorted input the way the
    Triton kernel's one-program-per-row parallelism does).

    See `_run_v5`'s docstring for what each mode does; the logic here mirrors it
    exactly, substituting `_seg_lse_coo(lam + other[idx], self_idx, size)` for
    `seg_lse_online(...)`. Cross-checked against `_run_v5` in
    testing/test_sinkslot_bench.py.
    """
    phi, psi = torch.zeros_like(log_a), torch.zeros_like(log_b)
    mode = _resolve_stop_mode(stop)

    if mode == "fixed":
        for _ in range(n_iters):
            phi = log_a - _seg_lse_coo(lam + psi[cols], rows, n)
            psi = log_b - _seg_lse_coo(lam + phi[rows], cols, m)
        return phi, psi, n_iters, None, None

    if mode == "potential_linf":
        if eps is None:
            raise ValueError("eps is required for stop.mode == 'potential_linf'")
        prev_phi, prev_psi = phi.clone(), psi.clone()
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            phi = log_a - _seg_lse_coo(lam + psi[cols], rows, n)
            psi = log_b - _seg_lse_coo(lam + phi[rows], cols, m)
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

    # mode in {"marginal", "potential"} -- the only two left after _resolve_stop_mode.
    a = log_a.exp()
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        phi = log_a - _seg_lse_coo(lam + psi[cols], rows, n)
        psi = log_b - _seg_lse_coo(lam + phi[rows], cols, m)
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            # One-step-ahead phi using the psi just updated above -- see _run_v5's
            # matching comment for why this isn't a redundant recompute.
            phi_next = log_a - _seg_lse_coo(lam + psi[cols], rows, n)
            row_marg = a * (phi - phi_next).exp()
            viol = float((row_marg - a).abs().max())
            if viol <= stop.tol:
                converged = True
                break
    return phi, psi, it, converged, viol


def sinkslot_solve(X, Y, a, b, eps, L, seed, n_iters, stop=None, chunk=None,
                    backend="auto"):
    """SinkSLOT end to end: build the sliced plan, then solve -- on any device.

    Dispatches on X's device and whether Triton is importable: the fused Triton
    kernels when both hold, the pure-torch path (`_run_v5_torch`, `_ot_1d_coo_
    batched`) otherwise. Same algorithm and stopping semantics either way --
    the two are cross-checked in testing/test_sinkslot_bench.py -- so this is
    the one call that works whether or not the caller has a CUDA GPU or Triton
    installed (#10). The pure-torch path is meaningfully slower (no fused
    kernels, no CSR launch-config tuning): use it for correctness/portability,
    not for reproducing the paper's own throughput numbers, which are all
    measured on the Triton path.

    `backend`: "auto" (default) picks Triton when it's importable and X is
    CUDA, torch otherwise -- covers the CPU and the "no Triton installed"
    cases. "triton" forces it, raising if unavailable. "torch" forces the
    pure-torch cost/solve path regardless of device -- the only way to run
    that path ON a CUDA tensor, since "auto" always prefers Triton there when
    it's available; useful for testing the torch path's CUDA numerics
    specifically, or as a workaround if Triton itself is ever the problem.
    The sliced-plan builder (`_ot_1d_coo_batched[_cuda]`) is chosen by device
    either way, independent of `backend`: it's plain torch regardless, and the
    CUDA-layout variant is still the better choice on a CUDA tensor even when
    the cost/solve stage is forced to torch.

    Returns (phi, psi, rows, cols, S, iters_run, converged, final_viol): phi,
    psi absorbed (phi=f/eps, psi=g/eps); rows/cols/S the sliced support, needed
    by callers building the transport plan or its gradient.
    """
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton', or 'torch', got {backend!r}")
    if backend == "triton":
        if not (_HAS_TRITON and X.is_cuda):
            raise ValueError("backend='triton' requires Triton installed and X on CUDA")
        use_triton = True
    elif backend == "torch":
        use_triton = False
    else:
        use_triton = _HAS_TRITON and X.is_cuda

    n, m = X.shape[0], Y.shape[0]
    ot1d = _ot_1d_coo_batched_cuda if X.is_cuda else _ot_1d_coo_batched
    rows, cols, S = sot_plan_coo(X, Y, a, b, L=L, seed=seed, chunk=chunk, ot1d=ot1d)
    cost = sparse_sqeuclidean_cost(X, Y, rows, cols, use_triton=use_triton)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    if use_triton:
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)
        phi, psi, it, converged, viol = _run_v5(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
            n_iters, stop, eps)
    else:
        phi, psi, it, converged, viol = _run_v5_torch(
            rows, cols, lam, log_a, log_b, n, m, n_iters, stop, eps)

    return phi, psi, rows, cols, S, it, converged, viol


# --------------------------------------------------------------------------
# Gradient (envelope theorem)
# --------------------------------------------------------------------------


def plan_barycentric_sparse(T_vals, rows, cols, x, y):
    """Barycentric projections (Tx, Ty) of a sparse plan given as (rows, cols, T_vals).

    Normalizes by the plan's own achieved marginals (scatter-summed from
    T_vals), not the target a, b -- matters when the solve hasn't fully
    converged. Also lives (independently) in flash_sinkhorn/bench/bench_forward.py,
    which this doesn't import from since that pulls in the whole benchmark harness.
    """
    n, d = x.shape
    m = y.shape[0]
    tiny = torch.finfo(T_vals.dtype).tiny
    r = torch.zeros(n, device=x.device, dtype=T_vals.dtype).index_add_(0, rows, T_vals)
    c = torch.zeros(m, device=y.device, dtype=T_vals.dtype).index_add_(0, cols, T_vals)
    Tx = torch.zeros(n, d, device=x.device, dtype=T_vals.dtype).index_add_(
        0, rows, T_vals.unsqueeze(1) * y[cols])
    Ty = torch.zeros(m, d, device=y.device, dtype=T_vals.dtype).index_add_(
        0, cols, T_vals.unsqueeze(1) * x[rows])
    Tx = Tx / r.clamp_min(tiny).unsqueeze(1)
    Ty = Ty / c.clamp_min(tiny).unsqueeze(1)
    return Tx, Ty


def slot_grad(X, Y, a, eps, L, seed, n_iters, backend="auto"):
    """grad_X SLOT_eps(X, Y) by the envelope theorem (Feydy et al. 2019's trick):

        grad_X SLOT_eps(X, Y) = 2 * diag(a) * (X - T_eps(X))

    where T_eps(X) is the barycentric projection of the converged sparse plan --
    no need to backprop through the Sinkhorn loop itself.

    Built on `sinkslot_solve`, so it works on CPU or CUDA-without-Triton via
    the same `backend` override ("auto" / "triton" / "torch", see
    `sinkslot_solve`'s own docstring) -- not CUDA/Triton-only anymore. Same
    fp32 caveat as `sparse_sqeuclidean_cost`'s Triton path when `backend`
    resolves to Triton; the torch path follows X/Y/a's own dtype.
    """
    phi, psi, rows, cols, S, _, _, _ = sinkslot_solve(
        X, Y, a, a, eps, L, seed, n_iters, backend=backend)
    cost = sparse_sqeuclidean_cost(
        X, Y, rows, cols,
        use_triton=(backend == "triton") if backend != "auto" else None,
    )
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps

    T_vals = (phi[rows] + psi[cols] + lam).exp()
    Tx, _ = plan_barycentric_sparse(T_vals, rows, cols, X, Y)
    return 2.0 * a[:, None] * (X - Tx)


# --------------------------------------------------------------------------
# Hessian-vector product (#18)
# --------------------------------------------------------------------------


def slot_hvp(X, Y, a, eps, L, seed, n_iters, v, tau2=3e-7, solve_tol=1e-11,
             max_cg_iter=8000):
    """Hessian-vector product H(X) @ v of grad_X SLOT_eps, by implicit
    differentiation of the entropic-OT fixed point on the sliced-OT support --
    the second-order counterpart to `slot_grad`'s envelope-theorem gradient.

    Derivation. At the converged potentials (phi, psi absorbed, as in
    `_run_v5`), the row/column marginal constraints hold identically along any
    curve X(t) = X + t*v with phi(t), psi(t) implicitly defined by those same
    constraints. Differentiating both constraints at t=0 gives a linear system
    in u := dphi/dt, w := dpsi/dt (n and m entries respectively):

        [ diag(a)      P    ] [u]   [rhs_u]
        [   P^T    diag(a)  ] [w] = [rhs_w]

    where P is the sparse plan (rows, cols, T_vals) and rhs_{u,w} come from
    scattering -P_ij * d(lam_ij)/dt (a per-entry quantity that's explicit in
    v, no implicit solve needed for it) by row and by column. This matrix is
    symmetric PSD -- for any (u, w), (u,w)^T M (u,w) = sum_ij P_ij*(u_i+w_j)^2
    >= 0, using P's row/col sums being a itself -- with a 1-D null space (the
    usual (+c,-c) potential-shift ambiguity), so `tau2` regularizes it, the
    same role FlashSinkhorn's own dense HVP (hvp.py) has its `tau2` play on
    its analogous Schur complement -- though NOT the same value: FlashSinkhorn's
    default (1e-5) is tuned for its dense Schur complement, a different matrix
    with different conditioning, and reused verbatim here gave ~23% error
    against a finite-difference check (see below) where this module's default
    (3e-7) gives ~2%. Given u, w, the HVP follows from differentiating
    T_eps(X) = diag(1/a) @ P @ Y (dr/dt = 0 along the constraint curve, so only
    dP/dt contributes):

        H(X)v = 2*diag(a)*v - 2*diag(1/a)-weighted scatter of
                P_ij*(u_i + w_j + d(lam_ij)/dt) * Y_j

    Solved via `torchsparsegradutils.sparse_generic_solve` on the sparse (n+m)
    system directly (MINRES, since M is symmetric but only PSD, not PD) --
    per #18, rather than materialising the dense n x n Schur complement
    P @ diag(1/a) @ P^T the way the dense case can afford to.

    This is new code, grounded in standard implicit-differentiation-of-Sinkhorn
    theory (same structure as OTT-JAX's and FlashSinkhorn's own Hessians), not
    the paper's own derivation -- cross-check the notation there before relying
    on this for anything reported. Validated empirically instead: against a
    FROZEN-SUPPORT central finite difference of `slot_grad` (testing/test_hvp.py)
    -- frozen support, not a naive slot_grad(X+h*v) vs slot_grad(X-h*v), because
    each such call rebuilds the sliced-OT support from scratch, and
    gradient_flow/finite_diff.py already shows (empirically, in this repo) that
    doing so at any h small enough to resolve second-order structure picks up
    O(1/h) rank-flip jump artifacts unrelated to the real signal. On a frozen
    support, matches to ~2% relative error at the tuned defaults, stable across
    3 orders of magnitude in h (i.e. not a finite-difference truncation
    artifact) -- see the module's own history for the tau2 sweep this default
    was picked from. n_iters affects accuracy the same way it affects
    slot_grad's: too few inner Sinkhorn iterations leaves phi, psi (and hence
    the linear system's own P, rhs) short of the true fixed point.

    X, Y, a, v expected fp32 (same Triton-kernel constraint as slot_grad).
    `a` is used as both marginals, matching slot_grad's own
    `sot_plan_coo(X, Y, a, a, ...)` call (X, Y assumed equal-size, uniform-a,
    as in every caller in this repo).
    """
    try:
        import torchsparsegradutils as tsgu
        from torchsparsegradutils.utils import minres, MINRESSettings
    except ImportError as e:
        raise ImportError(
            "slot_hvp needs torchsparsegradutils (pip install torchsparsegradutils)"
        ) from e
    import functools

    n, d = X.shape
    m = Y.shape[0]
    rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=seed, ot1d=_ot_1d_coo_batched_cuda)
    cost = sparse_sqeuclidean_cost(X, Y, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps
    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)

    log_a = a.log()
    phi, psi, _, _, _ = _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                 log_a, log_a, n, m, n_iters)
    T_vals = (phi[rows] + psi[cols] + lam).exp()

    # Explicit part: d(lam_ij)/dt for this v, only X depends on t.
    dlam_dt = -(2.0 / eps) * ((X[rows] - Y[cols]) * v[rows]).sum(1)

    rhs_u = torch.zeros(n, device=X.device, dtype=T_vals.dtype).index_add_(
        0, rows, -(T_vals * dlam_dt))
    rhs_w = torch.zeros(m, device=X.device, dtype=T_vals.dtype).index_add_(
        0, cols, -(T_vals * dlam_dt))

    idx_u = torch.arange(n, device=X.device)
    idx_w = torch.arange(m, device=X.device) + n
    row_off, col_off = rows, cols + n
    I = torch.cat([idx_u, idx_w, row_off, col_off])
    J = torch.cat([idx_u, idx_w, col_off, row_off])
    V = torch.cat([a + tau2, a + tau2, T_vals, T_vals])
    Mmat = torch.sparse_coo_tensor(torch.stack([I, J]), V, (n + m, n + m)).coalesce()
    rhs = torch.cat([rhs_u, rhs_w]).unsqueeze(1)

    solve = functools.partial(
        minres, settings=MINRESSettings(minres_tolerance=solve_tol,
                                         max_cg_iterations=max_cg_iter))
    sol = tsgu.sparse_generic_solve(Mmat, rhs, solve=solve, transpose_solve=solve).squeeze(1)
    u, w = sol[:n], sol[n:]

    weight = T_vals * (u[rows] + w[cols] + dlam_dt)
    dT = torch.zeros(n, d, device=X.device, dtype=X.dtype).index_add_(
        0, rows, weight.unsqueeze(1) * Y[cols])
    return 2.0 * a[:, None] * v - 2.0 * dT
