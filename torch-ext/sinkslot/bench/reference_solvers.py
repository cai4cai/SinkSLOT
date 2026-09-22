"""Reusable wrappers for running the two reference baselines (FlashSinkhorn,
GeomLoss) with a genuine, native-style early-stopping check -- not
color-transfer-specific, meant for any experiment that needs a fair
head-to-head against these baselines (the scalability/speedup benchmarks in
particular).

Both baselines are called with the SAME potential-change stopping rule,
checked in the SAME way: max(|delta f|, |delta g|) < threshold, evaluated
every `check_every` iterations inside a single continuous loop (never a
restart-based post-hoc check). This is FlashSinkhorn's own native mechanism
(confirmed directly against the upstream ot-triton-lab/flash-sinkhorn
source, not the cai4cai fork -- upstream's sinkhorn_flashstyle_alternating/
_symmetric already expose `threshold`/`check_every` natively, just with no
`stop_mode` choice: potential-change is the only rule it knows). GeomLoss's
low-level `sinkhorn_loop` has no such hook at all, so
`geomloss_online_native` below reimplements its own update math directly
(single-scale, debias=False, fixed eps, matching
main_pixel.py's/bench_forward.py's own SqDist(X,Y) convention exactly) with
the check added in.

Verified that SinkSLOT's own "potential" stop mode computes the
mathematically IDENTICAL quantity, once you account for its phi/psi being
eps-absorbed (phi=f/eps): eps*max(|delta phi|, |delta psi|) matched
FlashSinkhorn's max(|delta f|, |delta g|) to float32 noise (~1e-6) in a
controlled test where both were forced to use the same dense a⊗b reference
measure. The three methods' potential-change checks are the same rule; only
their solved problems differ (SinkSLOT's sparse P^SOT vs the dense a⊗b these
two baselines use), so a shared --tol is a fair, apples-to-apples criterion
across all three under stop_mode="potential" -- unlike the earlier concern
(documented history in color_transfer/main_pixel.py) that turned out to be
a diagnostic-script bug, not a real formula mismatch.

SinkSLOT's own "potential" mode has a separate, confirmed correctness bug
(torch.utils.swap_tensors-based staleness on any non-converging check) --
safe only when check_every is set large enough that convergence genuinely
happens within the first check (e.g. check_every == max_iter, so at most one
check ever fires). Not addressed here since it is a SinkSLOT-side fix; see
torch-ext/sinkslot/sinkhorn_solvers.py and color_transfer/trajectory.py's own
docstring for the full trace.
"""

from typing import Optional, Tuple

import torch


def flashsinkhorn_native_run(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
    symmetric: bool = False, allow_tf32: bool = False, report_change: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool]]:
    """Thin wrapper around upstream FlashSinkhorn's OWN native early-stopping
    mechanism (threshold/check_every), no fork/patch required -- unlike this
    repo's earlier stop_mode="marginal" usage, which depends on a cai4cai-fork
    -specific addition not present upstream. Returns (f, g, n_iters_used,
    converged[, last_change]); converged is None if threshold is None (no
    stopping check at all, matching sinkhorn_flashstyle_alternating's own
    contract in that case).

    symmetric=True uses sinkhorn_flashstyle_symmetric (damped-Jacobi updates,
    matching GeomLoss's own update scheme -- see geomloss_online_native's
    docstring) instead of the default alternating (Gauss-Seidel) solver.

    report_change=True additionally returns the last computed
    max(|delta f|, |delta g|) value, for reporting/plotting (e.g. alongside
    SinkSLOT's own returned change scalar). Neither
    sinkhorn_flashstyle_alternating nor _symmetric exposes this value
    directly (only n_iters_used), so it costs two extra, cheap,
    threshold=None calls at n_iters_used-check_every and n_iters_used to
    reconstruct it post-hoc -- exact, not an approximation, since both calls
    are deterministic given the same inputs.
    """
    if symmetric:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric
        f, g, n_iters_used = sinkhorn_flashstyle_symmetric(
            sc, tc, sw, tw, use_epsilon_scaling=False, eps=eps, n_iters=max_iter,
            threshold=threshold, check_every=check_every, allow_tf32=allow_tf32,
            return_n_iters=True,
        )
        solve = lambda n: sinkhorn_flashstyle_symmetric(
            sc, tc, sw, tw, use_epsilon_scaling=False, eps=eps, n_iters=n,
            threshold=None, allow_tf32=allow_tf32)
    else:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
        f, g, n_iters_used = sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=max_iter,
            threshold=threshold, check_every=check_every, allow_tf32=allow_tf32,
            return_n_iters=True,
        )
        solve = lambda n: sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=n, threshold=None, allow_tf32=allow_tf32)
    converged = None if threshold is None else n_iters_used < max_iter

    if not report_change:
        return f, g, n_iters_used, converged

    if n_iters_used >= check_every:
        f_prev, g_prev = solve(n_iters_used - check_every)
        f_last, g_last = solve(n_iters_used)
        last_change = max((f_last - f_prev).abs().max().item(), (g_last - g_prev).abs().max().item())
    else:
        last_change = float("inf")
    return f, g, n_iters_used, converged, last_change


def geomloss_online_native(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool]]:
    """GeomLoss online (KeOps), reimplementing sinkhorn_loop's own update math
    directly (single-scale, debias=False, fixed eps, last_extrapolation=False
    -- matching main_pixel.py's/bench_forward.py's own SqDist(X,Y)=||x-y||^2
    convention exactly) so a genuine early-stopping check can be added, since
    GeomLoss's low-level sinkhorn_loop exposes no per-iteration hook at all
    (only a fixed-length eps_list, checked in full every call). Returns
    (f, g, n_iters_used, converged); converged is None if threshold is None.

    Verified this reproduces sinkhorn_loop's own output bit-for-bit at a
    fixed n_iters with no early stop triggered (threshold=None) -- see
    torch-ext/sinkslot/bench/tests or the color-transfer PR's own
    verification notes for the exact check.

    Update rule: GeomLoss's sinkhorn_loop always uses symmetric (damped
    Jacobi) updates -- both f and g recomputed from the PREVIOUS iteration's
    values simultaneously, then blended 50% old / 50% new -- never
    Gauss-Seidel/alternating (no such option exists in GeomLoss's API at any
    level). This is mathematically the SAME update scheme as FlashSinkhorn's
    OWN "symmetric" variant (sinkhorn_flashstyle_symmetric), so
    flashsinkhorn_native_run(..., symmetric=True) is the apples-to-apples
    FlashSinkhorn comparison point for this function, not the default
    alternating solver (which converges at a genuinely different rate -- a
    confirmed algorithmic difference, not a bug; see
    color_transfer/trajectory.py's own docstring for the measured
    comparison).
    """
    from functools import partial

    from geomloss._legacy.sinkhorn_divergence import log_weights
    from geomloss._legacy.sinkhorn_samples import lse_genred, softmin_online

    d = sc.shape[1]
    a_log, b_log = log_weights(sw), log_weights(tw)
    softmin = partial(softmin_online, log_conv=lse_genred("SqDist(X,Y)", d))
    C_xy = (sc, tc.detach())
    C_yx = (tc, sc.detach())

    g_ab = softmin(eps, C_yx, a_log)
    f_ba = softmin(eps, C_xy, b_log)

    n_iters_used = max_iter
    converged = False if threshold is not None else None
    last_change = float("inf")
    for i in range(max_iter):
        ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)
        gt_ab = softmin(eps, C_yx, a_log + f_ba / eps)
        f_new, g_new = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)

        if threshold is not None and (i + 1) % check_every == 0:
            change = max((f_new - f_ba).abs().max().item(), (g_new - g_ab).abs().max().item())
            last_change = change
            f_ba, g_ab = f_new, g_new
            if change < threshold:
                n_iters_used = i + 1
                converged = True
                break
        else:
            f_ba, g_ab = f_new, g_new

    return f_ba.squeeze(0), g_ab.squeeze(0), n_iters_used, converged, last_change
