"""SinkSLOT: sliced-OT support construction and per-entry cost.

This is the paper's own contribution -- a distinct algorithm from FlashSinkhorn
(sparse, built on a sliced-OT reference plan, rather than FlashSinkhorn's dense
fused kernel over the full cost matrix).

This module holds the pieces with no FlashSinkhorn equivalent to mirror: the
sliced-OT support construction (`sot_coo`, `_ot_1d_coo_batched[_cuda]`),
CSR conversion (`to_csr`), and the sparse cost (`sparse_sqeuclidean_cost`).
The Sinkhorn solve loops live in `sinkhorn_solvers.py`, the envelope-theorem
gradient in `gradient.py`, the Hessian-vector product in `hvp.py` -- split out
to mirror flash_sinkhorn's own {sinkhorn_solvers,implicit_grad,hvp}.py layout.

Package name: sinkslot
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def get_random_projections(d: int, L: int, seed: int) -> torch.Tensor:
    """L random directions, drawn uniformly from the unit sphere in R^d.

    Standard-normal draws, L2-normalised -- the standard way to sample
    uniformly on the sphere.

    Same construction as vendor/sinkhorn_methods.py's build_sot_plan, but on
    torch's own RNG rather than numpy's, so the two no longer agree bit-for-bit
    on a shared seed (different generator, same distribution).

    Dtype: there's no tensor input to take a dtype from (d, L, seed are plain
    Python ints), so this always returns float64, on purpose, for an accurate
    normalisation regardless of what dtype the caller works in. `sot_coo`
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


def sot_coo(
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
    thetas = torch.as_tensor(get_random_projections(d, L, seed), dtype=X.dtype, device=X.device)
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

    The sort is STABLE on purpose. `sot_coo` returns its entries ordered by
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
    # `sot_coo` coalesces on the flat key `row * m + col` and returns it
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
