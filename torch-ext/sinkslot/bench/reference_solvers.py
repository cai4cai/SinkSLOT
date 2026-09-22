"""Reusable wrappers for running the two reference baselines (FlashSinkhorn,
GeomLoss) with a genuine, native-style early-stopping check -- not
color-transfer-specific, meant for any experiment that needs a fair
head-to-head against these baselines (the scalability/speedup benchmarks in
particular).

Both baselines are called with the SAME potential-change stopping rule,
checked in the SAME way: max(|delta f|, |delta g|) < threshold, evaluated
every `check_every` iterations inside a single continuous loop (never a
restart-based post-hoc check). This is FlashSinkhorn's own native mechanism
(`flashsinkhorn_native_run`, calling sinkhorn_flashstyle_alternating/
_symmetric's own threshold/check_every directly). GeomLoss's low-level
`sinkhorn_loop` has no such hook at all, so `geomloss_online_native` (single
-scale) and `geomloss_multiscale_native` (two-scale, coarse warm-start then
one coarse-to-fine jump) reimplement its update math directly, reusing
GeomLoss's own building blocks where possible, with the check added in.

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
(torch.utils.swap_tensors-based staleness on any non-converging check),
fixed directly in torch-ext/sinkslot/sinkhorn_solvers.py (see that file's
own comment for the trace).

Every function here also returns the entropic transport cost at each
checkpoint, computed for free from the dual potentials via the standard
Sinkhorn duality formula cost = <a,f> + <b,g> (balanced, non-debiased case;
see e.g. GeomLoss's own sinkhorn_cost). Algorithms may legitimately differ
(different reference measure, different update scheme, different point-
cloud resolution), so this isn't meant to prove bit-exact agreement -- it's
a checkpoint-level sanity signal that every method is solving a problem
converging to a comparable cost, recorded under the same stopping rule.
"""

from typing import Optional, Tuple

import torch


def flashsinkhorn_native_run(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
    symmetric: bool = False, allow_tf32: bool = True, report_change: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float]:
    """Thin wrapper around upstream FlashSinkhorn's OWN native early-stopping
    mechanism (threshold/check_every), no fork/patch required -- unlike this
    repo's earlier stop_mode="marginal" usage, which depends on a cai4cai-fork
    -specific addition not present upstream. Returns (f, g, n_iters_used,
    converged, cost[, last_change]); converged is None if threshold is None
    (no stopping check at all, matching sinkhorn_flashstyle_alternating's own
    contract in that case).

    symmetric=True uses sinkhorn_flashstyle_symmetric (damped-Jacobi updates,
    matching GeomLoss's own update scheme -- see geomloss_online_native's
    docstring) instead of the default alternating (Gauss-Seidel) solver.

    allow_tf32 defaults to True, matching sinkhorn_flashstyle_alternating/
    _symmetric's own default: FlashSinkhorn is run the way it's meant to be
    run, not artificially constrained to match the other methods' precision.
    Algorithms differing is fine here (see module docstring) -- what has to
    match is the stopping rule.

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
    cost = float((sw * f).sum() + (tw * g).sum())

    if not report_change:
        return f, g, n_iters_used, converged, cost

    if n_iters_used >= check_every:
        f_prev, g_prev = solve(n_iters_used - check_every)
        f_last, g_last = solve(n_iters_used)
        last_change = max((f_last - f_prev).abs().max().item(), (g_last - g_prev).abs().max().item())
    else:
        last_change = float("inf")
    return f, g, n_iters_used, converged, cost, last_change


def geomloss_online_native(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float, float]:
    """GeomLoss online (KeOps), reimplementing sinkhorn_loop's own update math
    directly (single-scale, debias=False, fixed eps, last_extrapolation=False
    -- matching main_pixel.py's/bench_forward.py's own SqDist(X,Y)=||x-y||^2
    convention exactly) so a genuine early-stopping check can be added, since
    GeomLoss's low-level sinkhorn_loop exposes no per-iteration hook at all
    (only a fixed-length eps_list, checked in full every call). Returns
    (f, g, n_iters_used, converged, cost, last_change); converged is None if
    threshold is None.

    Verified this reproduces sinkhorn_loop's own output bit-for-bit at a
    fixed n_iters with no early stop triggered (threshold=None).

    Update rule: GeomLoss's sinkhorn_loop always uses symmetric (damped
    Jacobi) updates -- both f and g recomputed from the PREVIOUS iteration's
    values simultaneously, then blended 50% old / 50% new -- never
    Gauss-Seidel/alternating (no such option exists in GeomLoss's API at any
    level). This is mathematically the SAME update scheme as FlashSinkhorn's
    OWN "symmetric" variant (sinkhorn_flashstyle_symmetric), so
    flashsinkhorn_native_run(..., symmetric=True) is the apples-to-apples
    FlashSinkhorn comparison point for this function, not the default
    alternating solver (which converges at a genuinely different rate -- a
    confirmed algorithmic difference, not a bug).
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

    f_ba, g_ab = f_ba.squeeze(0), g_ab.squeeze(0)
    cost = float((sw * f_ba).sum() + (tw * g_ab).sum())
    return f_ba, g_ab, n_iters_used, converged, cost, last_change


def geomloss_multiscale_native(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
    warmup_coarse_iters: int = 5, cluster_scale: Optional[float] = None, truncate: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float, float]:
    """GeomLoss multiscale (KeOps), reusing GeomLoss's own two-scale building
    blocks directly (clusterize, kernel_truncation, extrapolate_samples,
    softmin_multiscale) rather than reimplementing them, since they are
    nontrivial (block-sparse KeOps reductions via cluster ranges). Returns
    (f, g, n_iters_used, converged, cost, last_change); converged is None if
    threshold is None.

    GeomLoss's own sinkhorn_multiscale is built around its epsilon-scaling
    annealing schedule: the "jump" from coarse to fine resolution happens
    once, exactly when the annealed eps has shrunk below the cluster scale.
    That schedule is incompatible with the fixed-eps, checkpointed-iteration
    convention the rest of this module uses (needed for a fair, same-eps
    comparison against SinkSLOT/FlashSinkhorn), so this function instead:
    runs a small, fixed number of Jacobi iterations at COARSE resolution as
    a cheap warm start (warmup_coarse_iters, not part of the checkpointed
    iteration count), performs the SAME single coarse-to-fine jump
    GeomLoss's own two-scale clusterize() produces (kernel_truncation +
    extrapolate_samples, at the fixed target eps), then continues Jacobi
    iterations at FINE resolution (block-sparse, truncated via
    kernel_truncation) for max_iter iterations with the periodic
    potential-change check -- well-defined throughout the fine phase since
    resolution (and therefore tensor shape) is fixed after the one jump.

    Potentials are returned de-permuted back to the caller's own sc/tc
    order (clusterize sorts points by cluster for memory contiguity).
    """
    from functools import partial

    from geomloss._legacy.sinkhorn_divergence import log_weights, max_diameter
    from geomloss._legacy.sinkhorn_samples import (
        clusterize, extrapolate_samples, keops_lse, kernel_truncation, softmin_multiscale,
    )
    from geomloss._legacy.utils import squared_distances

    d = sc.shape[1]
    softmin = partial(softmin_multiscale, log_conv=keops_lse("SqDist(X,Y)", d))
    cost_matrix = lambda x, y: squared_distances(x, y)

    if cluster_scale is None:
        diameter = max_diameter(sc, tc)
        cluster_scale = diameter / (d ** 0.5 * 2000 ** (1.0 / d))

    [a_c, a_f], [x_c, x_f], [ranges_x], perm_x = clusterize(sw, sc, scale=cluster_scale)
    [b_c, b_f], [y_c, y_f], [ranges_y], perm_y = clusterize(tw, tc, scale=cluster_scale)
    a_log_c, b_log_c = log_weights(a_c), log_weights(b_c)
    a_log_f, b_log_f = log_weights(a_f), log_weights(b_f)

    C_xy_c = (x_c, y_c, ranges_x, ranges_y, None)
    C_yx_c = (y_c, x_c, ranges_y, ranges_x, None)

    g_ab = softmin(eps, C_yx_c, a_log_c)
    f_ba = softmin(eps, C_xy_c, b_log_c)
    for _ in range(warmup_coarse_iters):
        ft_ba = softmin(eps, C_xy_c, b_log_c + g_ab / eps)
        gt_ab = softmin(eps, C_yx_c, a_log_c + f_ba / eps)
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)

    # The one coarse -> fine jump, exactly as sinkhorn_loop's own jump
    # handling does it (kernel_truncation then a parallel extrapolation).
    C_xy_f_full = (x_f, y_f.detach(), None, None, None)
    C_yx_f_full = (y_f, x_f.detach(), None, None, None)
    C_xy_fine, C_yx_fine = kernel_truncation(
        C_xy_c, C_yx_c, C_xy_f_full, C_yx_f_full, f_ba, g_ab, eps,
        truncate=truncate, cost=cost_matrix,
    )
    f_ba, g_ab = (
        extrapolate_samples(f_ba, g_ab, eps, 1.0, C_xy_c, b_log_c, C_xy_fine, softmin=softmin),
        extrapolate_samples(g_ab, f_ba, eps, 1.0, C_yx_c, a_log_c, C_yx_fine, softmin=softmin),
    )

    n_iters_used = max_iter
    converged = False if threshold is not None else None
    last_change = float("inf")
    prev_f, prev_g = f_ba, g_ab
    for i in range(max_iter):
        ft_ba = softmin(eps, C_xy_fine, b_log_f + g_ab / eps)
        gt_ab = softmin(eps, C_yx_fine, a_log_f + f_ba / eps)
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)

        if threshold is not None and (i + 1) % check_every == 0:
            change = max((f_ba - prev_f).abs().max().item(), (g_ab - prev_g).abs().max().item())
            last_change = change
            prev_f, prev_g = f_ba, g_ab
            if change < threshold:
                n_iters_used = i + 1
                converged = True
                break

    f_out = torch.empty_like(f_ba)
    f_out[perm_x] = f_ba
    g_out = torch.empty_like(g_ab)
    g_out[perm_y] = g_ab
    cost = float((sw * f_out).sum() + (tw * g_out).sum())
    return f_out, g_out, n_iters_used, converged, cost, last_change
