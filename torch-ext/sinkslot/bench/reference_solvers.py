"""Reusable early-stopping wrappers for FlashSinkhorn and GeomLoss, checked
under the same potential-change rule as SinkSLOT's "potential" mode
(verified identical once phi=f/eps is accounted for; see
color_transfer/trajectory_potential.py). Not color-transfer-specific.

flashsinkhorn_native_run calls sinkhorn_flashstyle_alternating/_symmetric's
own threshold/check_every directly, with defaults mirroring SamplesLoss's
own wherever the low-level function accepts them. GeomLoss's sinkhorn_loop
has no such hook at all, so geomloss_online_native and
geomloss_multiscale_native reimplement its update math (each verified
bit-exact against sinkhorn_loop's own output) with the check added in.

Every function also returns cost = <a,f> + <b,g> at each checkpoint, free
from the dual potentials already in hand. Not expected to match bit-for-bit
across methods (different reference measures/resolutions/update schemes) --
just a sanity signal that every method converges to a comparable value.
"""

import time
from typing import Optional, Tuple

import torch


def flashsinkhorn_native_run(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 10,
    symmetric: bool = True, use_epsilon_scaling: bool = False,
    allow_tf32: bool = True, report_change: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float]:
    """Defaults mirror SamplesLoss's own (backend="symmetric",
    inner_iterations=10, allow_tf32=True), except use_epsilon_scaling: the
    library's own annealing schedule has a fixed natural length set by
    diameter/blur/scaling, independent of max_iter, and can complete (fall
    through the loop normally) without ever passing a threshold check --
    n_iters_used < max_iter then falsely looks like early convergence.
    Confirmed: a run reporting converged=True this way had
    potential_change=0.88, nowhere near threshold. So threshold-based early
    stopping needs use_epsilon_scaling=False (fixed eps, no natural early
    exit) to mean what it says; pass True explicitly only when you just want
    FlashSinkhorn's fastest solve and don't need to trust converged/
    last_change. Only applies when symmetric=True (alternating has no such
    option); when True, eps maps to blur=eps**0.5, since the library ignores
    a plain eps= once scaling is on. last_extrapolation=False (symmetric
    only): the library's default appends one extra, unblended alpha=1.0 step
    after the loop completes normally (not on early threshold break) --
    dropped to match geomloss_online_native's own convention, which has no
    such step; this is what a persistent ~2e-3 potential gap between the two
    at fixed eps traced back to.

    Returns (f, g, n_iters_used, converged, cost[, last_change]); converged
    is None if threshold is None. report_change=True adds two extra
    threshold=None calls to reconstruct the last change value, since neither
    solver exposes it directly (only n_iters_used).
    """
    if symmetric:
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric
        eps_kwargs = {"blur": eps ** 0.5} if use_epsilon_scaling else {"eps": eps}
        f, g, n_iters_used = sinkhorn_flashstyle_symmetric(
            sc, tc, sw, tw, use_epsilon_scaling=use_epsilon_scaling, n_iters=max_iter,
            threshold=threshold, check_every=check_every, allow_tf32=allow_tf32,
            last_extrapolation=False, return_n_iters=True, **eps_kwargs,
        )
        solve = lambda n: sinkhorn_flashstyle_symmetric(
            sc, tc, sw, tw, use_epsilon_scaling=use_epsilon_scaling, n_iters=n,
            threshold=None, allow_tf32=allow_tf32, last_extrapolation=False, **eps_kwargs)
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


def flashsinkhorn_samplesloss_run(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 10,
    symmetric: bool = True, allow_tf32: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float]:
    """One-shot convergence run via FlashSinkhorn's own SamplesLoss(potentials=
    True), drop-in alternative to flashsinkhorn_native_run (same call
    signature and return shape) that goes through the library's own
    SamplesLoss wrapper instead of calling sinkhorn_flashstyle_alternating/
    _symmetric directly. Fixed eps (use_epsilon_scaling=False), for the same
    reason as flashsinkhorn_native_run's own default. debias=False (we want
    the raw entropic OT plan/potentials for barycentric projection, not a
    symmetrized divergence, which would need two extra Sinkhorn solves).
    normalize=False (our eps is calibrated in true RGB-space squared-
    Euclidean units; any internal rescaling would silently change what eps
    means, breaking the controlled comparison across methods).
    last_extrapolation=False, for the same reason as flashsinkhorn_native_run's
    own default (only applies to backend="symmetric"; SamplesLoss accepts
    the kwarg regardless of backend and ignores it for "alternating").

    Verified empirically to match flashsinkhorn_native_run bit-for-bit
    (same cost, same n_iters_used) on both backends, on the currently
    installed flash_sinkhorn==0.4.0 (PyPI) -- the earlier concern that
    SamplesLoss's backend="alternating" never threaded early-stopping
    through was fixed upstream in this version.

    Returns (f, g, n_iters_used, converged, cost).
    """
    from flash_sinkhorn.samples_loss import SamplesLoss

    loss = SamplesLoss(
        loss="sinkhorn", backend="symmetric" if symmetric else "alternating",
        potentials=True, return_n_iters=True, use_epsilon_scaling=False,
        eps=eps, n_iters=max_iter, threshold=threshold, inner_iterations=check_every,
        allow_tf32=allow_tf32, debias=False, normalize=False, last_extrapolation=False,
    )
    f, g, n_iters_used = loss(sw, sc, tw, tc)
    n_iters_used = int(n_iters_used)
    converged = None if threshold is None else n_iters_used < max_iter
    cost = float((sw * f).sum() + (tw * g).sum())
    return f, g, n_iters_used, converged, cost


def plan_diagnostics_dense(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, f: torch.Tensor, g: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float]:
    """Post-hoc marginal violation AND true transport cost <C,P> for the
    dense a(x)b reference measure (FlashSinkhorn, GeomLoss), P = a*b*exp(
    (f+g-C)/eps), computed directly from the converged potentials in row
    blocks so the full N x M plan is never materialized. Ground truth, not
    a solver's own internal shortcut.

    <C,P> (unlike the dual value <a,f>+<b,g> = <C,P> + eps*KL(P|a(x)b)) is
    directly comparable across methods regardless of reference measure,
    since it only depends on the achieved plan and the true cost matrix --
    the dual value conflates transport quality with each method's own
    entropy term against its own reference, which is not the same
    quantity when the reference itself differs (e.g. SinkSLOT's sparse
    P^SOT vs this dense a(x)b).

    Returns (max L-infinity marginal violation, <C,P>).
    """
    n = sc.shape[0]
    row_sum = torch.empty_like(sw)
    col_sum = torch.zeros_like(tw)
    transport_cost = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        p_block = sw[start:end, None] * tw[None, :] * (
            (f[start:end, None] + g[None, :] - cost_block) / eps
        ).exp()
        row_sum[start:end] = p_block.sum(dim=1)
        col_sum += p_block.sum(dim=0)
        transport_cost += float((cost_block * p_block).sum())
    viol = float(torch.maximum((row_sum - sw).abs().max(), (col_sum - tw).abs().max()))
    return viol, transport_cost


def sinkslot_plan_diagnostics(
    phi: torch.Tensor, psi: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
    S: torch.Tensor, cost: torch.Tensor, eps: float, sw: torch.Tensor, tw: torch.Tensor,
) -> Tuple[float, float]:
    """Post-hoc marginal violation AND true transport cost <C,P> for
    SinkSLOT's sparse P^SOT support, P_ij = S_ij*exp(phi_i+psi_j-C_ij/eps)
    on (rows,cols) only (zero elsewhere by construction). Ground truth from
    the converged potentials, not the solver's own internal shortcut check.
    See plan_diagnostics_dense's docstring for why <C,P> (not the dual
    value) is the quantity comparable against the other methods.

    Returns (max L-infinity marginal violation, <C,P>).
    """
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    p = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    row_sum = torch.zeros_like(sw).index_add_(0, rows, p)
    col_sum = torch.zeros_like(tw).index_add_(0, cols, p)
    viol = float(torch.maximum((row_sum - sw).abs().max(), (col_sum - tw).abs().max()))
    transport_cost = float((cost * p).sum())
    return viol, transport_cost


def plan_diagnostics_dense_l1(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, f: torch.Tensor, g: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float, float]:
    """L1 marginal violations (row, column, total mass) for the dense a(x)b
    reference measure, entirely in log-space via logsumexp -- unlike
    plan_diagnostics_dense's direct exp((f+g-C)/eps) then sum, this never
    exponentiates the raw (f+g-C)/eps term before reducing over the M (or N)
    axis, avoiding over/underflow at small eps.

    P_ij = a_i b_j exp((f_i+g_j-C_ij)/eps), so:
      row_sum_i = a_i * exp(f_i/eps + LSE_j[log(b_j) + (g_j-C_ij)/eps])
      col_sum_j = b_j * exp(g_j/eps + LSE_i[log(a_i) + (f_i-C_ij)/eps])
    Column LSE is combined across row-blocks via logsumexp's own
    composability (LSE(LSE(block1), LSE(block2), ...) == LSE of everything),
    so the full N x M plan is still never materialized.

    Returns (row_l1, col_l1, mass_l1):
      row_l1  = sum_i |row_sum_i - a_i|
      col_l1  = sum_j |col_sum_j - b_j|
      mass_l1 = |1 - sum_i row_sum_i|
    """
    n = sc.shape[0]
    log_sw = sw.clamp_min(torch.finfo(sw.dtype).tiny).log()
    log_tw = tw.clamp_min(torch.finfo(tw.dtype).tiny).log()
    row_sum = torch.empty_like(sw)
    col_lse_blocks = []
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        row_lse = torch.logsumexp(log_tw[None, :] + (g[None, :] - cost_block) / eps, dim=1)
        row_sum[start:end] = sw[start:end] * (f[start:end] / eps + row_lse).exp()
        col_lse_blocks.append(torch.logsumexp(
            log_sw[start:end, None] + (f[start:end, None] - cost_block) / eps, dim=0))
    col_lse = torch.logsumexp(torch.stack(col_lse_blocks, dim=0), dim=0)
    col_sum = tw * (g / eps + col_lse).exp()

    row_l1 = float((row_sum - sw).abs().sum())
    col_l1 = float((col_sum - tw).abs().sum())
    mass_l1 = float((1.0 - row_sum.sum()).abs())
    return row_l1, col_l1, mass_l1


def sinkslot_plan_diagnostics_l1(
    phi: torch.Tensor, psi: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
    S: torch.Tensor, cost: torch.Tensor, eps: float, sw: torch.Tensor, tw: torch.Tensor,
) -> Tuple[float, float, float]:
    """L1 marginal violations (row, column, total mass) for SinkSLOT's sparse
    P^SOT support, via a manual index-based logsumexp over (rows,cols) (torch
    has no built-in scatter/index logsumexp): max-shift per group first
    (index_reduce_ "amax", groups with no support entries correctly stay at
    -inf and later exponentiate to 0, not NaN), then index_add the shifted
    exponentials.

    Returns (row_l1, col_l1, mass_l1), same definitions as
    plan_diagnostics_dense_l1.
    """
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    log_p = phi[rows] + psi[cols] + log_S - cost / eps

    neg_inf = torch.finfo(log_p.dtype).min
    row_max = torch.full_like(sw, neg_inf).index_reduce_(0, rows, log_p, "amax", include_self=True)
    col_max = torch.full_like(tw, neg_inf).index_reduce_(0, cols, log_p, "amax", include_self=True)
    row_sumexp = torch.zeros_like(sw).index_add_(0, rows, (log_p - row_max[rows]).exp())
    col_sumexp = torch.zeros_like(tw).index_add_(0, cols, (log_p - col_max[cols]).exp())
    row_sum = row_max.exp() * row_sumexp
    col_sum = col_max.exp() * col_sumexp

    row_l1 = float((row_sum - sw).abs().sum())
    col_l1 = float((col_sum - tw).abs().sum())
    mass_l1 = float((1.0 - row_sum.sum()).abs())
    return row_l1, col_l1, mass_l1


def geomloss_online_native(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
    stop_mode: str = "potential_linf",
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float, float]:
    """GeomLoss online (KeOps): reimplements sinkhorn_loop's own update math
    at a fixed eps (single-scale, debias=False), since sinkhorn_loop has no
    early-stop hook of its own. Verified bit-exact against sinkhorn_loop
    itself at threshold=None. Always symmetric (damped-Jacobi) updates --
    same scheme as flashsinkhorn_native_run(symmetric=True); GeomLoss has no
    alternating/Gauss-Seidel option at any level.

    stop_mode="potential_linf" (default): max(|df|, |dg|) < threshold, same
    rule as flashsinkhorn_native_run's own default and SinkSLOT's "potential"
    mode. stop_mode="marginal": max row/col violation of the dense a(x)b
    plan, |P_i. - a_i| / |P_.j - b_j| <= threshold, matching FlashSinkhorn's/
    SinkSLOT's own "marginal" stop mode. Derived for free from ft_ba/gt_ab
    (already computed every iteration for the update itself), no extra
    softmin call: P_i. = a_i*exp((f_ba_i - ft_ba_i)/eps) since ft_ba is
    exactly the fresh row target -eps*logsumexp_j[...] that f_ba would equal
    at a marginal-exact fixed point, same "no extra LSE call" trick
    sinkslot_alternating_triton's and sinkhorn_flashstyle_alternating's own
    marginal checks use.

    Returns (f, g, n_iters_used, converged, cost, last_change).
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

        if threshold is not None and (i + 1) % check_every == 0:
            if stop_mode == "marginal":
                row_marg = sw * ((f_ba - ft_ba).squeeze(0) / eps).exp()
                col_marg = tw * ((g_ab - gt_ab).squeeze(0) / eps).exp()
                change = max((row_marg - sw).abs().max().item(), (col_marg - tw).abs().max().item())
            else:
                change = max((ft_ba - f_ba).abs().max().item(), (gt_ab - g_ab).abs().max().item())
            last_change = change
            f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)
            if change < threshold:
                n_iters_used = i + 1
                converged = True
                break
        else:
            f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)

    f_ba, g_ab = f_ba.squeeze(0), g_ab.squeeze(0)
    cost = float((sw * f_ba).sum() + (tw * g_ab).sum())
    return f_ba, g_ab, n_iters_used, converged, cost, last_change


def geomloss_multiscale_native(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
    warmup_coarse_iters: int = 5, cluster_scale: Optional[float] = None, truncate: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float, float]:
    """GeomLoss multiscale (KeOps): a coarse warm-start, one coarse-to-fine
    jump, then fine-resolution updates with the same check -- reusing
    GeomLoss's own clusterize/kernel_truncation/extrapolate_samples/
    softmin_multiscale rather than reimplementing them. Runs at a fixed eps:
    GeomLoss's own sinkhorn_multiscale anneals instead (incompatible with
    the fixed-eps convention here), so warmup_coarse_iters plus the one
    jump stand in for its annealing schedule. Verified bit-exact against a
    direct sinkhorn_loop call built with the same two-scale inputs.

    Returns (f, g, n_iters_used, converged, cost, last_change), de-permuted
    back to sc/tc's own order (clusterize sorts points by cluster).
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

    # Coarse -> fine jump (kernel_truncation + parallel extrapolation).
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


def plan_diagnostics_dense_rounded(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, f: torch.Tensor, g: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float]:
    """Marginal violation and true transport cost of the ROUNDED plan (issue
    #57: Altschuler, Niles-Weed & Rigollet, NeurIPS 2017, Algorithm 2), for
    the dense a(x)b reference measure. Never materializes the N x M plan --
    N, M run up to ~2.5e5 here, so a dense (N, M) matrix would be hundreds of
    GB; this streams the same row-shrink / column-shrink / rank-one-deficit
    algorithm in row blocks instead.

    Let P_ij = a_i b_j exp((f_i+g_j-C_ij)/eps) (the unrounded plan). The
    reference algorithm is:
        x_i = min(a_i / row_sum(P)_i, 1);      G1 = P * x[:,None]
        y_j = min(b_j / col_sum(G1)_j, 1);     G2 = G1 * y[None,:]
        err_a = a - row_sum(G2); err_b = b - col_sum(G2)
        G_hat = G2 + outer(err_a, err_b) / err_a.sum()
    Since G2 = P * x_i * y_j is just P rescaled by per-row/per-column
    scalars (no cross-term accumulation), every quantity above is computable
    from row-blocked passes over P without ever storing G1, G2 or G_hat:

      Pass 1: row_sum(P) -> x
      Pass 2: col_sum(G1) = sum_i x_i P_ij -> y
      Pass 3: row_sum(G2) (per block, direct) and col_sum(G2) (accumulated)
              -> err_a, err_b
      Pass 4: true cost of G2 (needs P_block again) plus true cost of the
              rank-one term sum_ij C_ij err_a_i err_b_j / sum(err_a) (needs
              only C_block, not P_block, weighted by the now-known err
              vectors) -- both accumulated in the same block loop.

    The rank-one term's own row/column sums are known in closed form (no
    extra pass needed): summing it over j gives err_a_i * sum(err_b) /
    sum(err_a), and over i gives err_b_j exactly (using sum(err_a) in the
    denominator both times, matching the reference implementation). So
    row_sum(G_hat) = a - err_a * (1 - sum(err_b)/sum(err_a)) and
    col_sum(G_hat) = b exactly, up to floating point -- both computed here
    directly from Pass 3's outputs, not by re-deriving the marginals of a
    materialized G_hat.

    Returns (max L-infinity marginal violation, <C, G_hat>).
    """
    n = sc.shape[0]
    tiny = torch.finfo(sw.dtype).tiny

    # Pass 1: row_sum(P) -> x
    row_sum_p = torch.empty_like(sw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        p_block = sw[start:end, None] * tw[None, :] * (
            (f[start:end, None] + g[None, :] - cost_block) / eps
        ).exp()
        row_sum_p[start:end] = p_block.sum(dim=1)
    x = (sw / row_sum_p.clamp_min(tiny)).clamp(max=1.0)

    # Pass 2: col_sum(G1) = sum_i x_i P_ij -> y
    col_sum_g1 = torch.zeros_like(tw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        p_block = sw[start:end, None] * tw[None, :] * (
            (f[start:end, None] + g[None, :] - cost_block) / eps
        ).exp()
        col_sum_g1 += (x[start:end, None] * p_block).sum(dim=0)
    y = (tw / col_sum_g1.clamp_min(tiny)).clamp(max=1.0)

    # Pass 3: row_sum(G2) (direct per block) + col_sum(G2) (accumulated) -> err_a, err_b
    row_sum_g2 = torch.empty_like(sw)
    col_sum_g2 = torch.zeros_like(tw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        p_block = sw[start:end, None] * tw[None, :] * (
            (f[start:end, None] + g[None, :] - cost_block) / eps
        ).exp()
        g2_block = x[start:end, None] * y[None, :] * p_block
        row_sum_g2[start:end] = g2_block.sum(dim=1)
        col_sum_g2 += g2_block.sum(dim=0)
    err_a = sw - row_sum_g2
    err_b = tw - col_sum_g2
    sum_err_a = float(err_a.sum())
    sum_err_b = float(err_b.sum())

    # Closed-form marginals of G_hat (no extra pass): the rank-one term's row
    # sum is err_a_i * sum(err_b)/sum(err_a); its column sum is err_b_j exactly.
    row_sum_ghat = row_sum_g2 + err_a * (sum_err_b / max(sum_err_a, tiny))
    col_sum_ghat = col_sum_g2 + err_b
    viol = float(torch.maximum((row_sum_ghat - sw).abs().max(), (col_sum_ghat - tw).abs().max()))

    # Pass 4: true cost of G2, plus true cost of the rank-one deficit term.
    true_cost = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        p_block = sw[start:end, None] * tw[None, :] * (
            (f[start:end, None] + g[None, :] - cost_block) / eps
        ).exp()
        g2_block = x[start:end, None] * y[None, :] * p_block
        true_cost += float((cost_block * g2_block).sum())
        rank_one_block = err_a[start:end, None] * err_b[None, :] / max(sum_err_a, tiny)
        true_cost += float((cost_block * rank_one_block).sum())

    return viol, true_cost


def sinkslot_plan_diagnostics_rounded(
    sc: torch.Tensor, tc: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor,
    rows: torch.Tensor, cols: torch.Tensor, S: torch.Tensor, cost: torch.Tensor,
    eps: float, sw: torch.Tensor, tw: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float]:
    """Marginal violation and true transport cost of the ROUNDED plan (same
    algorithm as plan_diagnostics_dense_rounded, see its docstring) for
    SinkSLOT's sparse P^SOT support.

    Steps 1-2 (row/column shrink) stay sparse: they only ever touch the
    given (rows, cols) support, via the same index_add pattern
    sinkslot_plan_diagnostics uses. Step 3's rank-one deficit term,
    err_a_i * err_b_j / sum(err_a), is generically DENSE (nonzero for
    essentially every (i,j) pair, not just the original sparse support) --
    that's the one place this can't stay sparse. Its marginals are still
    known in closed form (no extra pass needed, same as the dense case).
    Its cost contribution sum_ij C_ij err_a_i err_b_j / sum(err_a) does need
    a dense row-blocked pass over the true cost matrix (sc, tc directly,
    not the sparse `cost` argument, which only covers the original support)
    -- the one genuinely expensive part of rounding a sparse plan.

    Returns (max L-infinity marginal violation, <C, G_hat>).
    """
    n, m = sw.shape[0], tw.shape[0]
    tiny = torch.finfo(sw.dtype).tiny

    log_S = S.clamp_min(tiny).log()
    p = (phi[rows] + psi[cols] + log_S - cost / eps).exp()

    # Steps 1-2: row/column shrink, sparse.
    row_sum_p = torch.zeros_like(sw).index_add_(0, rows, p)
    x = (sw / row_sum_p.clamp_min(tiny)).clamp(max=1.0)
    p1 = p * x[rows]
    col_sum_g1 = torch.zeros_like(tw).index_add_(0, cols, p1)
    y = (tw / col_sum_g1.clamp_min(tiny)).clamp(max=1.0)
    p2 = p1 * y[cols]

    row_sum_g2 = torch.zeros_like(sw).index_add_(0, rows, p2)
    col_sum_g2 = torch.zeros_like(tw).index_add_(0, cols, p2)
    err_a = sw - row_sum_g2
    err_b = tw - col_sum_g2
    sum_err_a = float(err_a.sum())
    sum_err_b = float(err_b.sum())

    row_sum_ghat = row_sum_g2 + err_a * (sum_err_b / max(sum_err_a, tiny))
    col_sum_ghat = col_sum_g2 + err_b
    viol = float(torch.maximum((row_sum_ghat - sw).abs().max(), (col_sum_ghat - tw).abs().max()))

    # Sparse part of the cost: <C, G2> on the original support.
    true_cost = float((cost * p2).sum())

    # Dense part: the rank-one deficit's cost, over the TRUE (dense) cost
    # matrix -- the deficit is not confined to the sparse support.
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = ((sc[start:end, None, :] - tc[None, :, :]) ** 2).sum(-1)
        rank_one_block = err_a[start:end, None] * err_b[None, :] / max(sum_err_a, tiny)
        true_cost += float((cost_block * rank_one_block).sum())

    return viol, true_cost
