"""SinkSLOT's Sinkhorn solver loops and device-agnostic entry point.

This is the v5 variant: v1's alternating scheme with both reductions fused into
Triton kernels (online-softmax segmented LSE, CSR for the row half-step and CSC
for the column half-step). Mathematically identical to a plain torch segmented
LSE; the point is throughput, not a different algorithm. The CUDA-graph capture
of the upstream production path is intentionally omitted here.

`sinkslot_solve` is the device-agnostic entry point: `sinkslot_alternating_triton`
above when Triton is installed and the input is CUDA, the pure-torch fallback
(`sinkslot_alternating_torch`) otherwise -- same algorithm, cross-checked in
testing/test_sinkslot_bench.py, so a CPU machine or a machine without Triton
installed can still run SinkSLOT, just without the fused-kernel throughput.

Split out from solver.py to mirror flash_sinkhorn's own sinkhorn_solvers.py:
that module's `sinkhorn_flashstyle_alternating`/`_symmetric` are the FlashSinkhorn
equivalent of the two loops below, and neither module underscore-prefixes its
solver loops even though they're not the top-level entry point (a plain,
descriptive name plus leaving it out of `__init__.py`'s exports is how both
packages mark "internal, but not hidden behind Python's underscore convention"
-- see `_resolve_stop_mode`'s own docstring for the #14 bug this fixes: the
problem was the package-level re-export, not the lack of a leading underscore).
"""

from __future__ import annotations

import torch

from .solver import (
    _HAS_TRITON,
    _ot_1d_coo_batched,
    _ot_1d_coo_batched_cuda,
    sot_plan_coo,
    to_csr,
    sparse_sqeuclidean_cost,
)

if _HAS_TRITON:
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
    use `sinkslot_solve` or `sinkslot_alternating_torch`, which use
    `_seg_lse_coo` instead.
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
    """Shared by `sinkslot_alternating_triton` and `sinkslot_alternating_torch`:
    resolve `stop.mode` (or "fixed" if `stop` is None) and validate it against
    `_STOP_MODES` upfront, so the four valid modes stay one source of truth
    instead of two independently-maintained checks. They drifted out of sync
    once already -- the torch path validated from the start, the Triton path
    didn't, so a typo'd mode used to behave differently depending on which
    backend happened to run it (fixed alongside this helper, not by it: the
    old inline check ran only after the "fixed" and "potential_linf" branches
    had already been ruled out, but for an invalid mode neither of those
    branches matches anyway, so validating here instead, before any branch
    runs, raises in the exact same cases as before).
    """
    mode = getattr(stop, "mode", "fixed") if stop is not None else "fixed"
    if mode not in _STOP_MODES:
        raise ValueError(
            f"unknown stop.mode {mode!r}; expected one of "
            f"'fixed', 'marginal', 'potential', 'potential_linf'"
        )
    return mode


def sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
            n_iters, stop=None, eps=None):
    """Alternating (Gauss-Seidel) Sinkhorn over prebuilt CSR/CSC, fused Triton half-steps.

    Potentials are absorbed (phi = f/eps, psi = g/eps): lam already carries
    log P^SOT - C/eps, so seg_lse_online with base=log_a returns log_a - LSE
    directly, which is the next phi. The excluded potential is zeroed on its own
    axis (z_n, z_m).

    Returns (phi, psi, iters_run, converged, final_viol). With `stop` None or
    stop.mode == "fixed" it runs exactly n_iters (converged/final_viol None).
    With stop.mode in {"marginal", "potential"} it runs up to stop.max_iter and
    stops once a proxy for the max (L-infinity) row/col marginal violation
    drops below stop.tol (see the check's own comment below -- it's not the
    exact violation, traded for one fewer LSE call per side per check).
    (potential mode falls back
    to the same check here; the u/v-change rule is Spar-Sink's and is applied
    there.) This is max, not a total-variation sum: matches the SLOT repo's
    actual working "marg_viol" rule (bench/solvers/sinkslot.py's
    `_violation`/`_run_v5` there) -- a sum over n terms against a fixed
    absolute tol is unreachable at n=10,000 regardless of convergence, which
    is why an earlier version of this function (sum-based) looked like
    marginal-violation stopping didn't work here either.

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
                torch.utils.swap_tensors(prev_phi, phi)
                torch.utils.swap_tensors(prev_psi, psi)
        return phi, psi, it, converged, change

    # mode in {"marginal", "potential"} -- the only two left after _resolve_stop_mode.
    #
    # Check both row and column marginal violation, without an extra
    # seg_lse_online call.
    # swap_tensors below moves phi's data into phi_old in place (no memcpy);
    # phi's old buffer becomes garbage, which is fine since seg_lse_online
    # immediately overwrites it via out=phi.
    a, b = log_a.exp(), log_b.exp()
    phi_old, psi_old = phi.clone(), psi.clone()
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        torch.utils.swap_tensors(phi_old, phi)
        seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi)
        torch.utils.swap_tensors(psi_old, psi)
        seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi)
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            row_marg = a * (phi_old - phi).exp()
            col_marg = b * (psi_old - psi).exp()
            # One sync (float() at the end), not two: the max itself runs on
            # device first.
            viol = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if viol <= stop.tol:
                converged = True
                break
    return phi, psi, it, converged, viol


def sinkslot_symmetric_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
            n_iters, stop=None, eps=None, alpha=0.5):
    """Symmetric (Jacobi) Sinkhorn over prebuilt CSR/CSC, fused Triton half-steps (#34).

    Both half-steps read the SAME (phi, psi) pair -- unlike
    `sinkslot_alternating_triton`'s Gauss-Seidel scheme, where the column
    half-step already sees the row half-step's fresh update -- so the two
    candidates are computed into separate buffers, then blended:

        phi_new = (1 - alpha) * phi + alpha * phi_cand
        psi_new = (1 - alpha) * psi + alpha * psi_cand

    matching FlashSinkhorn's own `sinkhorn_flashstyle_symmetric` update rule,
    simplified to a single fixed `alpha` applied every iteration rather than
    that function's own alpha=1.0-then-0.5 schedule tied to its epsilon
    scaling -- SinkSLOT runs at a single fixed eps, so there's no schedule to
    tie it to. `alpha=0.5` (the default, matching FlashSinkhorn's own
    steady-state value) is genuine Jacobi averaging; `alpha=1.0` recovers an
    unaveraged simultaneous update (both candidates fully replace their old
    values, still order-independent unlike Gauss-Seidel).

    Returns (phi, psi, iters_run, converged, final_viol), same contract as
    `sinkslot_alternating_triton`. With stop.mode in {"marginal", "potential"}:
    unlike alternating, NEITHER marginal is exact after a blended update --
    the column marginal being exact right after the column half-step is a
    Gauss-Seidel property this scheme doesn't have (both `phi` and `psi` move
    on every iteration, and by less than a full half-step whenever
    alpha < 1) -- so both marginals need a fresh check: two extra half-step
    calls per check, reusing `phi_cand`/`psi_cand`'s buffers since they're
    stale by then. FlashSinkhorn's own implementation avoids the extra calls
    by reconstructing the pre-blend candidate algebraically
    (`cand = 2*new - old`), a shortcut that only holds at alpha=0.5 exactly;
    not used here since `alpha` is a general parameter, not hardcoded.
    """
    r_blk, r_w = launch_cfg(r_idx.numel(), n)
    c_blk, c_w = launch_cfg(c_idx.numel(), m)
    phi, psi = torch.zeros_like(log_a), torch.zeros_like(log_b)
    phi_cand, psi_cand = torch.empty_like(log_a), torch.empty_like(log_b)
    z_n, z_m = torch.zeros_like(log_a), torch.zeros_like(log_b)

    mode = _resolve_stop_mode(stop)

    # A shared closure here, unlike sinkslot_alternating_triton's inlined
    # seg_lse_online calls, because the two situations aren't the same
    # shape: half_steps() is called 5 times below with an IDENTICAL "compute
    # both candidates" pattern every time (the marginal branch reuses it
    # verbatim for its post-blend fresh check too, see below), whereas
    # alternating's "marginal" branch only ever needs ONE of its two
    # half-steps recomputed (phi_next), never both -- there's no
    # equally-clean repeated unit to name there. This is also why an
    # earlier, cross-backend closure merge of _run_v5/_run_v5_torch was
    # reverted (#22's history): that one needed three closures to avoid
    # aliasing phi/phi_next, and the indirection cost more than the
    # duplicated lines were worth. No such aliasing risk here.
    def half_steps():
        # Both read the CURRENT phi/psi (unmutated at this point in every
        # call site below); writes go into phi_cand/psi_cand, never phi/psi
        # directly, so this is safe to call before OR after the blend that
        # follows it -- it never observes its own output.
        seg_lse_online(r_ptr, r_idx, r_lam, z_n, psi, n, r_blk, r_w, base=log_a, out=phi_cand)
        seg_lse_online(c_ptr, c_idx, c_lam, z_m, phi, m, c_blk, c_w, base=log_b, out=psi_cand)

    if mode == "fixed":
        for _ in range(n_iters):
            half_steps()
            phi.mul_(1.0 - alpha).add_(phi_cand, alpha=alpha)
            psi.mul_(1.0 - alpha).add_(psi_cand, alpha=alpha)
        return phi, psi, n_iters, None, None

    if mode == "potential_linf":
        if eps is None:
            raise ValueError("eps is required for stop.mode == 'potential_linf'")
        prev_phi, prev_psi = phi.clone(), psi.clone()
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            half_steps()
            phi.mul_(1.0 - alpha).add_(phi_cand, alpha=alpha)
            psi.mul_(1.0 - alpha).add_(psi_cand, alpha=alpha)
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
    b = log_b.exp()
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        half_steps()
        phi.mul_(1.0 - alpha).add_(phi_cand, alpha=alpha)
        psi.mul_(1.0 - alpha).add_(psi_cand, alpha=alpha)
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            # Fresh half-steps from the just-updated (phi, psi) -- see the
            # docstring above for why neither marginal is exact here the way
            # alternating's column marginal is. Overwrites phi_cand/psi_cand,
            # already stale from the blend two lines up.
            half_steps()
            row_marg = a * (phi - phi_cand).exp()
            col_marg = b * (psi - psi_cand).exp()
            viol = max(float((row_marg - a).abs().max()), float((col_marg - b).abs().max()))
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


def sinkslot_alternating_torch(rows, cols, lam, log_a, log_b, n, m, n_iters, stop=None, eps=None):
    """Pure-torch counterpart to `sinkslot_alternating_triton` -- same four
    `stop.mode` semantics, no Triton, no CSR/CSC (operates directly on the COO
    `sot_plan_coo` returns, since `index_add_`/`scatter_reduce_` don't need
    sorted input the way the Triton kernel's one-program-per-row parallelism
    does).

    See `sinkslot_alternating_triton`'s docstring for what each mode does; the
    logic here mirrors it exactly, substituting
    `_seg_lse_coo(lam + other[idx], self_idx, size)` for `seg_lse_online(...)`.
    Cross-checked against `sinkslot_alternating_triton` in
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
        prev_phi, prev_psi = phi, psi
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
                prev_phi = phi
                prev_psi = psi
        return phi, psi, it, converged, change

    # mode in {"marginal", "potential"} -- the only two left after _resolve_stop_mode.
    # Check both row and column marginal violation, without an extra
    # _seg_lse_coo call. No .clone() needed: phi/psi are reassigned to a new
    # tensor each iteration, never mutated in place, so `phi_old = phi`
    # before overwriting is already a cheap reference.
    a, b = log_a.exp(), log_b.exp()
    phi_old, psi_old = phi, psi
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        phi_old = phi
        psi_old = psi
        phi = log_a - _seg_lse_coo(lam + psi[cols], rows, n)
        psi = log_b - _seg_lse_coo(lam + phi[rows], cols, m)
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            row_marg = a * (phi_old - phi).exp()
            col_marg = b * (psi_old - psi).exp()
            viol = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if viol <= stop.tol:
                converged = True
                break
    return phi, psi, it, converged, viol


def sinkslot_symmetric_torch(rows, cols, lam, log_a, log_b, n, m, n_iters, stop=None,
                              eps=None, alpha=0.5):
    """Pure-torch counterpart to `sinkslot_symmetric_triton` (#34) -- see its
    docstring for the update rule and stop-mode semantics; the logic here
    mirrors it exactly, substituting `_seg_lse_coo(...)` for
    `seg_lse_online(...)`, matching `sinkslot_alternating_torch`'s own
    relationship to `sinkslot_alternating_triton`.
    """
    phi, psi = torch.zeros_like(log_a), torch.zeros_like(log_b)
    mode = _resolve_stop_mode(stop)

    # Shared closure for the same reason sinkslot_symmetric_triton's own
    # half_steps() is -- see that function's docstring.
    def half_steps(phi_cur, psi_cur):
        phi_cand = log_a - _seg_lse_coo(lam + psi_cur[cols], rows, n)
        psi_cand = log_b - _seg_lse_coo(lam + phi_cur[rows], cols, m)
        return phi_cand, psi_cand

    if mode == "fixed":
        for _ in range(n_iters):
            phi_cand, psi_cand = half_steps(phi, psi)
            phi = (1.0 - alpha) * phi + alpha * phi_cand
            psi = (1.0 - alpha) * psi + alpha * psi_cand
        return phi, psi, n_iters, None, None

    if mode == "potential_linf":
        if eps is None:
            raise ValueError("eps is required for stop.mode == 'potential_linf'")
        prev_phi, prev_psi = phi.clone(), psi.clone()
        it = 0
        converged = False
        change = float("inf")
        while it < stop.max_iter:
            phi_cand, psi_cand = half_steps(phi, psi)
            phi = (1.0 - alpha) * phi + alpha * phi_cand
            psi = (1.0 - alpha) * psi + alpha * psi_cand
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
    b = log_b.exp()
    it = 0
    converged = False
    viol = float("inf")
    while it < stop.max_iter:
        phi_cand, psi_cand = half_steps(phi, psi)
        phi = (1.0 - alpha) * phi + alpha * phi_cand
        psi = (1.0 - alpha) * psi + alpha * psi_cand
        it += 1
        if it % stop.check_every == 0 or it == stop.max_iter:
            # Fresh half-steps from the just-updated (phi, psi) -- see
            # sinkslot_symmetric_triton's docstring for why neither marginal
            # is exact here the way alternating's column marginal is.
            phi_fresh, psi_fresh = half_steps(phi, psi)
            row_marg = a * (phi - phi_fresh).exp()
            col_marg = b * (psi - psi_fresh).exp()
            viol = max(float((row_marg - a).abs().max()), float((col_marg - b).abs().max()))
            if viol <= stop.tol:
                converged = True
                break
    return phi, psi, it, converged, viol


def sinkslot_solve(X, Y, a, b, eps, L, seed, n_iters, stop=None, chunk=None,
                    backend="auto", variant="alternating", alpha=0.5):
    """SinkSLOT end to end: build the sliced plan, then solve -- on any device.

    Dispatches on X's device and whether Triton is importable: the fused Triton
    kernels when both hold, the pure-torch path (`sinkslot_alternating_torch`,
    `_ot_1d_coo_batched`) otherwise. Same algorithm and stopping semantics
    either way -- the two are cross-checked in testing/test_sinkslot_bench.py --
    so this is the one call that works whether or not the caller has a CUDA GPU
    or Triton installed (#10). The pure-torch path is meaningfully slower (no
    fused kernels, no CSR launch-config tuning): use it for
    correctness/portability, not for reproducing the paper's own throughput
    numbers, which are all measured on the Triton path.

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

    `variant`: "alternating" (default, unchanged behavior) picks the
    Gauss-Seidel loop (`sinkslot_alternating_triton`/`_torch`); "symmetric"
    (#34) picks the Jacobi loop (`sinkslot_symmetric_triton`/`_torch`) instead,
    which updates phi and psi simultaneously from the same prior state rather
    than sequentially -- see that function's own docstring for the update
    rule and why its marginal-violation check costs more than alternating's.
    `alpha` is the symmetric update's blend weight (`phi_new = (1-alpha)*phi
    + alpha*phi_cand`); unused when `variant="alternating"`.

    Returns (phi, psi, rows, cols, S, iters_run, converged, final_viol): phi,
    psi absorbed (phi=f/eps, psi=g/eps); rows/cols/S the sliced support, needed
    by callers building the transport plan or its gradient.
    """
    if backend not in ("auto", "triton", "torch"):
        raise ValueError(f"backend must be 'auto', 'triton', or 'torch', got {backend!r}")
    if variant not in ("alternating", "symmetric"):
        raise ValueError(f"variant must be 'alternating' or 'symmetric', got {variant!r}")
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
        if variant == "symmetric":
            phi, psi, it, converged, viol = sinkslot_symmetric_triton(
                r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
                n_iters, stop, eps, alpha)
        else:
            phi, psi, it, converged, viol = sinkslot_alternating_triton(
                r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m,
                n_iters, stop, eps)
    else:
        if variant == "symmetric":
            phi, psi, it, converged, viol = sinkslot_symmetric_torch(
                rows, cols, lam, log_a, log_b, n, m, n_iters, stop, eps, alpha)
        else:
            phi, psi, it, converged, viol = sinkslot_alternating_torch(
                rows, cols, lam, log_a, log_b, n, m, n_iters, stop, eps)

    return phi, psi, rows, cols, S, it, converged, viol
