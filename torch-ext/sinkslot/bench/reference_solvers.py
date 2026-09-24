"""GeomLoss reference solver, checked under the same potential-change rule
as SinkSLOT's "potential" mode (verified identical once phi=f/eps is
accounted for; see color_transfer/main.py). Not color-transfer-specific.

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
