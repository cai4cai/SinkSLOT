"""Reference (baseline) solvers ported from their authors' code, each with the
shared stop modes added in: GeomLoss online, SROT and Spar-Sink.

Stop modes (StopCfg.mode), shared across every benchmarked method:
  "fixed"      run exactly n_iters.
  "potential"  max(|df|, |dg|) < tol between consecutive checkpoints (every
               check_every iterations) -- FlashSinkhorn's native rule.
  "marginal"   L-infinity marginal violation <= tol at each checkpoint.
  "scaling"    Spar-Sink's own rule on the scaling vectors u, v (Spar-Sink only).
A mode a solver does not implement raises ValueError.

GeomLoss's sinkhorn_loop has no early-stop hook of its own, so
geomloss_online reimplements its update math (verified bit-exact against
sinkhorn_loop's own output) with the check added in. Returns
cost = <a,f> + <b,g> at convergence, free from the dual potentials already
in hand -- not expected to match other methods' own cost bit-for-bit
(different reference measures/update schemes), just a sanity signal.

Post-hoc plan diagnostics (marginal violation, true cost, rounding to the
transport polytope) live in plan_diagnostics.py instead: measurement, not
a solver.
"""

from typing import Optional, Tuple

import torch


def geomloss_online(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, max_iter: int, threshold: Optional[float] = None, check_every: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[bool], float, float]:
    """GeomLoss online (KeOps): reimplements sinkhorn_loop's own update math
    at a fixed eps (single-scale, debias=False), since sinkhorn_loop has no
    early-stop hook of its own. Verified bit-exact against sinkhorn_loop
    itself at threshold=None. Always symmetric (damped-Jacobi) updates;
    GeomLoss has no alternating/Gauss-Seidel option at any level.

    Stop rule: max(|df|, |dg|) < threshold between consecutive checkpoints
    (every check_every iterations), on the damped iterates -- the same rule
    as SinkSLOT's "potential" mode and FlashSinkhorn-symmetric's native check.

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
    prev_f, prev_g = f_ba, g_ab
    for i in range(max_iter):
        ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)
        gt_ab = softmin(eps, C_yx, a_log + f_ba / eps)
        f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)

        if threshold is not None and (i + 1) % check_every == 0:
            change = max((f_ba - prev_f).abs().max().item(), (g_ab - prev_g).abs().max().item())
            last_change = change
            prev_f, prev_g = f_ba, g_ab
            if change < threshold:
                n_iters_used = i + 1
                converged = True
                break

    f_ba, g_ab = f_ba.squeeze(0), g_ab.squeeze(0)
    cost = float((sw * f_ba).sum() + (tw * g_ab).sum())
    return f_ba, g_ab, n_iters_used, converged, cost, last_change


# =============================================================================
# SROT: Sliced-Regularized Optimal Transport
# =============================================================================
# Nguyen, "Sliced-Regularized Optimal Transport", arXiv:2604.23944.
# Port of build_sot_plan / sinkhorn_sot from https://github.com/khainb/SROT
# (lib/sinkhorn_methods.py). SROT replaces the entropic regularizer's reference
# measure: it penalizes KL(pi || pi_SOT), with pi_SOT the uniform average of L
# one-dimensional OT plans on random projections, instead of KL(pi || a (x) b).
# It is a different optimum from standard entropic OT, not a faster route to it.
#
# Deviations from the authors' code (same reference coupling and fixed point):
#   - 1-D OT plans via the north-west-corner rule on sorted cumulative masses
#     (torch) instead of ot.emd_1d; identical for a convex ground cost.
#   - Projection directions from SinkSLOT's seeded torch generator, not numpy.
#   - fp32 throughout, with log-domain updates instead of linear-domain fp64
#     u/v scalings.
#   - Stop modes as listed in the module docstring; theirs is the L-infinity
#     marginal rule ("marginal" here).


def build_sot_plan(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    *, slices: int, delta: float = 1e-8, seed: int = 0,
) -> torch.Tensor:
    """Uniform-average sliced-OT reference plan from `slices` random 1-D projections.

    Returns (1 - delta) * pi_SOT + delta * (a (x) b), as in the authors' code; the
    delta mix keeps every entry strictly positive when a, b > 0.

    Each slice projects both clouds onto a random unit direction and solves the 1-D
    OT problem by the north-west-corner rule on the sorted marginals: with cumulative
    masses ca and cb, the plan entry is max(0, min(ca_i, cb_j) - max(ca_{i-1}, cb_{j-1})).
    The plan is dense, O(n*m).

    Directions come from sinkslot.solver.get_random_projections, so SROT and
    SinkSLOT use the same L directions for the same seed. Everything else is
    computed in `x`'s dtype.
    """
    from sinkslot.solver import get_random_projections

    n, d = x.shape
    m = y.shape[0]
    thetas = get_random_projections(d, slices, seed).to(dtype=x.dtype, device=x.device)

    px_all = x @ thetas.T  # (n, L)
    py_all = y @ thetas.T  # (m, L)

    pi_sot = torch.zeros(n, m, device=x.device, dtype=x.dtype)
    for ell in range(slices):
        order_x = torch.argsort(px_all[:, ell])
        order_y = torch.argsort(py_all[:, ell])

        ca = torch.cumsum(a[order_x], dim=0)
        cb = torch.cumsum(b[order_y], dim=0)
        ca_prev = torch.cat([ca.new_zeros(1), ca[:-1]])
        cb_prev = torch.cat([cb.new_zeros(1), cb[:-1]])

        upper = torch.minimum(ca.unsqueeze(1), cb.unsqueeze(0))
        lower = torch.maximum(ca_prev.unsqueeze(1), cb_prev.unsqueeze(0))
        overlap = (upper - lower).clamp_min(0.0)

        pi_sot[order_x.unsqueeze(1), order_y.unsqueeze(0)] += overlap

    pi_sot /= slices
    if delta > 0.0:
        pi_sot = (1.0 - delta) * pi_sot + delta * torch.outer(a, b)
    return pi_sot


def _check_mode(stop, allowed: Tuple[str, ...], solver: str) -> str:
    mode = "fixed" if stop is None else stop.mode
    if mode not in allowed:
        raise ValueError(f"{solver} does not support stop mode {mode!r}; choices: {allowed}")
    return mode


def _srot_sinkhorn(
    cost: torch.Tensor, log_pi: torch.Tensor, log_a: torch.Tensor, log_b: torch.Tensor,
    eps: float, n_iters: int, stop=None,
):
    """Log-domain alternating Sinkhorn against the pi_SOT reference plan.

    Fixed point is pi = pi_SOT * exp((f (+) g - C)/eps) with marginals a, b:

        f_i = eps * [log a_i - logsumexp_j(log pi_ij + (g_j - C_ij)/eps)]
        g_j = eps * [log b_j - logsumexp_i(log pi_ij + (f_i - C_ij)/eps)]

    Stop modes: "fixed" (n_iters), "potential", "marginal" (see module docstring).
    After each g update the column marginal is exactly b, so "marginal" checks the
    row marginal a * exp((f_old - f)/eps) and the column one b * exp((g_old - g)/eps)
    without an extra logsumexp.

    Returns (f, g, iters_run, converged, last_check): last_check is the potential
    change ("potential"), the marginal violation ("marginal"), or None ("fixed").
    """
    mode = _check_mode(stop, ("fixed", "potential", "marginal"), "SROT")
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def _row_lse(gv):
        return torch.logsumexp(log_pi + (gv.unsqueeze(0) - cost) / eps, dim=1)

    def _col_lse(fv):
        return torch.logsumexp(log_pi + (fv.unsqueeze(1) - cost) / eps, dim=0)

    if mode == "fixed":
        for _ in range(n_iters):
            f = eps * (log_a - _row_lse(g))
            g = eps * (log_b - _col_lse(f))
        return f, g, n_iters, None, None

    a, b = log_a.exp(), log_b.exp()
    prev_f, prev_g = f, g
    it, converged, last = 0, False, float("inf")
    while it < stop.max_iter:
        f_old, g_old = f, g
        f = eps * (log_a - _row_lse(g))
        g = eps * (log_b - _col_lse(f))
        it += 1
        if it % stop.check_every != 0:
            continue
        if mode == "potential":
            last = max((f - prev_f).abs().max().item(), (g - prev_g).abs().max().item())
            prev_f, prev_g = f, g
            if last < stop.tol:
                converged = True
                break
        else:
            row_marg = a * ((f_old - f) / eps).exp()
            col_marg = b * ((g_old - g) / eps).exp()
            last = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if last <= stop.tol:
                converged = True
                break
    return f, g, it, converged, last


# =============================================================================
# Spar-Sink / Rand-Sink: importance-sparsified Sinkhorn
# =============================================================================
# Li, Yu, Li, Meng, "Importance Sparsification for Sinkhorn Algorithm", JMLR
# (arXiv:2306.06581). Port of spar_sinkhorn from
# https://github.com/Mengyu8042/Spar-Sink (code/all_funcs.py), Poisson sampling.
# Both methods sparsify the Sinkhorn kernel and iterate on the kept entries; they
# differ only in the sampling distribution, and approximate the same entropic
# problem as FlashSinkhorn/GeomLoss.
#
# Deviations from the authors' code (same sampling, K/q rescaling, empty-line
# handling and sparse iteration):
#   - Log-domain fp32 updates instead of linear-domain u = a / (K v), which
#     underflows (exp(-C/eps) = 0) at small eps.
#   - Stop modes as listed in the module docstring; "scaling" is theirs.
#   - Their kernel is exp(-C/eps) (entropy convention); the plan it yields is
#     the same, and we report the plan's own <T, C>, not their dual.


SPARSINK_METHODS = ("spar_sink", "rand_sink")


def build_sparse_kernel(
    cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float,
    *, method: str, sample_size: int, seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poisson-sample the Sinkhorn kernel; return (rows, cols, log_values).

    With inclusion probability q_ij = min(1, s * p_ij), keep entry (i, j) with
    probability q_ij and rescale it to K_ij / q_ij (unbiased for K). s bounds the
    expected nnz, so the realised count varies.

        spar_sink: p_ij ∝ sqrt(a_i b_j)   -- their importance probability
        rand_sink: p_ij ∝ 1               -- uniform over all entries

    Values are returned in log space: log(K_ij / q_ij) = -C_ij/eps - log q_ij.

    Sampling is chunked over rows so each torch.rand/nonzero call stays under the
    INT_MAX element limit; a generator advances identically either way, so the
    sample does not depend on the chunking.
    """
    if method not in SPARSINK_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Choices: {SPARSINK_METHODS}")

    n, m = cost.shape
    if method == "spar_sink":
        weights = torch.outer(a.sqrt(), b.sqrt())
    else:
        weights = torch.ones(n, m, device=cost.device, dtype=cost.dtype)
    probs = weights / weights.sum()
    q = (sample_size * probs).clamp_max(1.0)

    generator = torch.Generator(device=cost.device).manual_seed(seed)

    int32_max = 2**31 - 1
    if n * m <= int32_max:
        keep = torch.rand(n, m, generator=generator, device=cost.device, dtype=cost.dtype) < q
        rows, cols = keep.nonzero(as_tuple=True)
    else:
        chunk_rows = max(1, int32_max // m)
        row_chunks, col_chunks = [], []
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            keep_chunk = torch.rand(end - start, m, generator=generator, device=cost.device,
                                     dtype=cost.dtype) < q[start:end]
            r, c = keep_chunk.nonzero(as_tuple=True)
            row_chunks.append(r + start)
            col_chunks.append(c)
        rows = torch.cat(row_chunks)
        cols = torch.cat(col_chunks)

    log_values = -cost[rows, cols] / eps - q[rows, cols].log()
    return rows, cols, log_values


def _sparsink_sinkhorn(
    rows: torch.Tensor, cols: torch.Tensor, log_values: torch.Tensor,
    log_a: torch.Tensor, log_b: torch.Tensor, eps: float, n_iters: int, stop=None,
):
    """Log-domain alternating Sinkhorn over the sampled support only, O(nnz) per sweep.

    Each half-update is a segmented logsumexp over the kept entries, grouped by row
    (then by column): the log-domain form of their u = a/(Kv), v = b/(K^T u).

    Rows and columns with no sampled entry are dropped, as in the authors' code, and
    the rest keep their original a_i, b_j (no renormalization). Dropped lines carry
    potential 0 and are excluded from every stop check. When the kept a and b have
    different total mass, f and g shift by -/+ eps*log(sum b_kept / sum a_kept) per
    iteration while the plan stays fixed, so "potential" does not fire and the run
    ends at max_iter with converged=False.

    Stop modes: "fixed", "potential", "marginal" (see module docstring), and
    "scaling", their rule: 0.5 * (err_u + err_v) < stop.scaling_tol, with
    err_u = max|u - u_prev| / max(max|u|, max|u_prev|, 1) between consecutive
    iterations, u = exp(f/eps), v = exp(g/eps).

    Returns (f, g, empty, iters_run, converged, last_check), where empty counts the
    dropped rows plus columns.
    """
    mode = _check_mode(stop, ("fixed", "potential", "marginal", "scaling"), "Spar-Sink")
    n = log_a.shape[0]
    m = log_b.shape[0]
    neg_inf = torch.finfo(log_values.dtype).min
    row_kept = torch.zeros(n, dtype=torch.bool, device=log_a.device)
    row_kept[rows] = True
    col_kept = torch.zeros(m, dtype=torch.bool, device=log_b.device)
    col_kept[cols] = True
    empty = int((~row_kept).sum() + (~col_kept).sum())
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def segmented_lse(z: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        mx = torch.full((size,), neg_inf, device=z.device, dtype=z.dtype)
        mx = mx.scatter_reduce(0, index, z, reduce="amax", include_self=True)
        acc = torch.zeros(size, device=z.device, dtype=z.dtype)
        acc = acc.index_add(0, index, (z - mx[index]).exp())
        return mx + acc.clamp_min(torch.finfo(z.dtype).tiny).log()

    def _update_f(gv):
        fv = eps * (log_a - segmented_lse(log_values + gv[cols] / eps, rows, n))
        return torch.where(row_kept, fv, 0.0)

    def _update_g(fv):
        gv = eps * (log_b - segmented_lse(log_values + fv[rows] / eps, cols, m))
        return torch.where(col_kept, gv, 0.0)

    def _rel_change(p_old, p_new):
        u_old, u_new = (p_old / eps).exp(), (p_new / eps).exp()
        scale = max(u_new.abs().max().item(), u_old.abs().max().item(), 1.0)
        return (u_new - u_old).abs().max().item() / scale

    if mode == "fixed":
        for _ in range(n_iters):
            f = _update_f(g)
            g = _update_g(f)
        return f, g, empty, n_iters, None, None

    a, b = log_a.exp()[row_kept], log_b.exp()[col_kept]
    prev_f, prev_g = f, g
    it, converged, last = 0, False, float("inf")
    while it < stop.max_iter:
        f_old, g_old = f, g
        f = _update_f(g)
        g = _update_g(f)
        it += 1
        if it % stop.check_every != 0:
            continue
        if mode == "scaling":
            last = 0.5 * (_rel_change(f_old, f) + _rel_change(g_old, g))
            if last < stop.scaling_tol:
                converged = True
                break
            continue
        if mode == "potential":
            last = max((f - prev_f).abs().max().item(), (g - prev_g).abs().max().item())
            prev_f, prev_g = f, g
            if last < stop.tol:
                converged = True
                break
        else:
            row_marg = a * ((f_old - f)[row_kept] / eps).exp()
            col_marg = b * ((g_old - g)[col_kept] / eps).exp()
            last = float(torch.maximum((row_marg - a).abs().max(), (col_marg - b).abs().max()))
            if last <= stop.tol:
                converged = True
                break
    return f, g, empty, it, converged, last
