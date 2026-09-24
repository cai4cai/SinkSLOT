"""Post-hoc marginal violation and true transport cost <C,P>, computed
directly from converged potentials rather than a solver's own internal
shortcut. Dense (a(x)b reference, FlashSinkhorn/GeomLoss) and sparse
(SinkSLOT's P^SOT) variants, each in an unrounded and a rounded-to-the-
transport-polytope (issue #57: Altschuler, Weed, Rigollet, NeurIPS 2017
Algorithm 2) form. Dense functions never materialize the N x M plan (N, M
run up to ~2.5e5), streaming row blocks instead.

<C,P> (unlike the dual value <a,f>+<b,g> = <C,P> + eps*KL(P|reference)) is
directly comparable across methods regardless of reference measure, since
it only depends on the achieved plan and the true cost matrix -- the dual
value conflates transport quality with each method's own entropy term
against its own reference, not the same quantity when the reference
itself differs (e.g. SinkSLOT's sparse P^SOT vs the dense a(x)b).
"""

from typing import Tuple

import torch
from geomloss._legacy.utils import squared_distances


def _dense_block(sc, tc, sw, tw, eps, f, g, start, end):
    """cost_block, p_block = a*b*exp((f+g-C)/eps) for rows [start:end)."""
    cost_block = squared_distances(sc[start:end], tc)
    p_block = sw[start:end, None] * tw[None, :] * (
        (f[start:end, None] + g[None, :] - cost_block) / eps
    ).exp()
    return cost_block, p_block


def _rounding_marginals(row_sum_g2, col_sum_g2, sw, tw, tiny):
    """Given the row-shrunk-then-column-shrunk plan G2's own marginals,
    the rank-one deficit correction outer(err_a, err_b)/sum(err_a) has
    marginals known in closed form (row sum err_a_i*sum(err_b)/sum(err_a),
    col sum err_b_j exactly) -- no extra pass needed to get G_hat's own
    marginals or violation from this.

    Returns (err_a, sum_err_a, viol_lmax, viol_l1).
    """
    err_a = sw - row_sum_g2
    err_b = tw - col_sum_g2
    sum_err_a = float(err_a.sum())
    sum_err_b = float(err_b.sum())
    row_sum_ghat = row_sum_g2 + err_a * (sum_err_b / max(sum_err_a, tiny))
    col_sum_ghat = col_sum_g2 + err_b
    viol_lmax = float(torch.maximum((row_sum_ghat - sw).abs().max(), (col_sum_ghat - tw).abs().max()))
    viol_l1 = float((row_sum_ghat - sw).abs().sum() + (col_sum_ghat - tw).abs().sum())
    return err_a, sum_err_a, viol_lmax, viol_l1


def plan_diagnostics_dense(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, f: torch.Tensor, g: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float, float, float, float]:
    """Returns (max L-infinity marginal violation, row_l1, col_l1, mass_l1,
    <C,P>) for the dense a(x)b reference measure. row_l1+col_l1 is the
    combined L1 marginal violation; mass_l1 = |1 - total mass|.
    """
    n = sc.shape[0]
    row_sum = torch.empty_like(sw)
    col_sum = torch.zeros_like(tw)
    transport_cost = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block, p_block = _dense_block(sc, tc, sw, tw, eps, f, g, start, end)
        row_sum[start:end] = p_block.sum(dim=1)
        col_sum += p_block.sum(dim=0)
        transport_cost += float((cost_block * p_block).sum())
    viol_lmax = float(torch.maximum((row_sum - sw).abs().max(), (col_sum - tw).abs().max()))
    row_l1 = float((row_sum - sw).abs().sum())
    col_l1 = float((col_sum - tw).abs().sum())
    mass_l1 = float((1.0 - row_sum.sum()).abs())
    return viol_lmax, row_l1, col_l1, mass_l1, transport_cost


def sinkslot_plan_diagnostics(
    phi: torch.Tensor, psi: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
    S: torch.Tensor, cost: torch.Tensor, eps: float, sw: torch.Tensor, tw: torch.Tensor,
) -> Tuple[float, float, float, float, float]:
    """Returns (max L-infinity marginal violation, row_l1, col_l1, mass_l1,
    <C,P>) for SinkSLOT's sparse P^SOT support, P_ij =
    S_ij*exp(phi_i+psi_j-C_ij/eps) on (rows,cols) only (zero elsewhere by
    construction)."""
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    p = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    row_sum = torch.zeros_like(sw).index_add_(0, rows, p)
    col_sum = torch.zeros_like(tw).index_add_(0, cols, p)
    viol_lmax = float(torch.maximum((row_sum - sw).abs().max(), (col_sum - tw).abs().max()))
    row_l1 = float((row_sum - sw).abs().sum())
    col_l1 = float((col_sum - tw).abs().sum())
    mass_l1 = float((1.0 - row_sum.sum()).abs())
    transport_cost = float((cost * p).sum())
    return viol_lmax, row_l1, col_l1, mass_l1, transport_cost


def plan_diagnostics_dense_rounded(
    sc: torch.Tensor, tc: torch.Tensor, sw: torch.Tensor, tw: torch.Tensor,
    eps: float, f: torch.Tensor, g: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float, float]:
    """Marginal violation (L-infinity and L1) and true transport cost of
    the plan rounded onto the exact transport polytope, for the dense
    a(x)b reference measure. Never materializes the N x M plan -- streams
    the row-shrink / column-shrink / rank-one-deficit algorithm in row
    blocks instead:

        x_i = min(a_i / row_sum(P)_i, 1);      G1 = P * x[:,None]
        y_j = min(b_j / col_sum(G1)_j, 1);     G2 = G1 * y[None,:]
        G_hat = G2 + outer(err_a, err_b) / sum(err_a)   (err_a, err_b, see
                                                          _rounding_marginals)

    G2 = P * x_i * y_j is just P rescaled by per-row/per-column scalars (no
    cross-term accumulation), so every quantity above is a row-blocked pass
    over P: Pass 1 row_sum(P) -> x; Pass 2 col_sum(G1) -> y; Pass 3
    row_sum(G2) + col_sum(G2) -> err_a/err_b/violation (_rounding_marginals);
    Pass 4 true cost of G2 plus the rank-one deficit's cost (needs a dense
    pass over the true cost matrix even here, since err_a/err_b are
    generically nonzero everywhere, not just where P was).

    Returns (max L-infinity marginal violation, L1 marginal violation
    (row+col combined), <C, G_hat>).
    """
    n = sc.shape[0]
    tiny = torch.finfo(sw.dtype).tiny

    row_sum_p = torch.empty_like(sw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        _, p_block = _dense_block(sc, tc, sw, tw, eps, f, g, start, end)
        row_sum_p[start:end] = p_block.sum(dim=1)
    x = (sw / row_sum_p.clamp_min(tiny)).clamp(max=1.0)

    col_sum_g1 = torch.zeros_like(tw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        _, p_block = _dense_block(sc, tc, sw, tw, eps, f, g, start, end)
        col_sum_g1 += (x[start:end, None] * p_block).sum(dim=0)
    y = (tw / col_sum_g1.clamp_min(tiny)).clamp(max=1.0)

    row_sum_g2 = torch.empty_like(sw)
    col_sum_g2 = torch.zeros_like(tw)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        _, p_block = _dense_block(sc, tc, sw, tw, eps, f, g, start, end)
        g2_block = x[start:end, None] * y[None, :] * p_block
        row_sum_g2[start:end] = g2_block.sum(dim=1)
        col_sum_g2 += g2_block.sum(dim=0)
    err_a, sum_err_a, viol_lmax, viol_l1 = _rounding_marginals(row_sum_g2, col_sum_g2, sw, tw, tiny)
    err_b = tw - col_sum_g2

    true_cost = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block, p_block = _dense_block(sc, tc, sw, tw, eps, f, g, start, end)
        g2_block = x[start:end, None] * y[None, :] * p_block
        true_cost += float((cost_block * g2_block).sum())
        rank_one_block = err_a[start:end, None] * err_b[None, :] / max(sum_err_a, tiny)
        true_cost += float((cost_block * rank_one_block).sum())

    return viol_lmax, viol_l1, true_cost


def sinkslot_plan_diagnostics_rounded(
    sc: torch.Tensor, tc: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor,
    rows: torch.Tensor, cols: torch.Tensor, S: torch.Tensor, cost: torch.Tensor,
    eps: float, sw: torch.Tensor, tw: torch.Tensor, block_n: int = 4096,
) -> Tuple[float, float, float]:
    """Same algorithm as plan_diagnostics_dense_rounded (see its docstring)
    for SinkSLOT's sparse P^SOT support. Steps 1-2 (row/column shrink) stay
    sparse, touching only (rows, cols). The rank-one deficit term is
    generically dense (nonzero for essentially every (i,j), not just the
    original sparse support) -- its cost contribution needs a dense
    row-blocked pass over the true cost matrix (sc, tc directly, not the
    sparse `cost` argument), the one genuinely expensive part of rounding a
    sparse plan.

    Returns (max L-infinity marginal violation, L1 marginal violation
    (row+col combined), <C, G_hat>).
    """
    n = sw.shape[0]
    tiny = torch.finfo(sw.dtype).tiny

    log_S = S.clamp_min(tiny).log()
    p = (phi[rows] + psi[cols] + log_S - cost / eps).exp()

    row_sum_p = torch.zeros_like(sw).index_add_(0, rows, p)
    x = (sw / row_sum_p.clamp_min(tiny)).clamp(max=1.0)
    p1 = p * x[rows]
    col_sum_g1 = torch.zeros_like(tw).index_add_(0, cols, p1)
    y = (tw / col_sum_g1.clamp_min(tiny)).clamp(max=1.0)
    p2 = p1 * y[cols]

    row_sum_g2 = torch.zeros_like(sw).index_add_(0, rows, p2)
    col_sum_g2 = torch.zeros_like(tw).index_add_(0, cols, p2)
    err_a, sum_err_a, viol_lmax, viol_l1 = _rounding_marginals(row_sum_g2, col_sum_g2, sw, tw, tiny)
    err_b = tw - col_sum_g2

    true_cost = float((cost * p2).sum())
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        cost_block = squared_distances(sc[start:end], tc)
        rank_one_block = err_a[start:end, None] * err_b[None, :] / max(sum_err_a, tiny)
        true_cost += float((cost_block * rank_one_block).sum())

    return viol_lmax, viol_l1, true_cost
