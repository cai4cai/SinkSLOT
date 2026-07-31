"""SinkSLOT gradient via the envelope theorem, built from the native v5 solver.

grad_X SLOT_eps(X, Y) = 2 * diag(a) * (X - T_eps(X)), where T_eps(X) is the
barycentric projection of the converged sparse plan (Feydy et al. 2019's
envelope-theorem trick: no need to backprop through the Sinkhorn loop itself).

Uses exactly the pipeline flash_sinkhorn.bench.bench_forward.bench_sinkslotcuda
benchmarks: sot_plan_coo -> sparse_sqeuclidean_cost -> to_csr (row + col) ->
_run_v5 -> T_vals = (phi[rows]+psi[cols]+lam).exp() -> barycentric projection.
`plan_barycentric_sparse` and the minimal fixed-iteration stop config are
vendored here (rather than importing bench_forward.py, which pulls in the
whole benchmark harness) since both are a few self-contained lines.

Runs in fp32: sparse_sqeuclidean_cost's Triton kernel accumulates in fp32
regardless of input dtype (see sinkslot.py), so X/Y must be fp32 here even
though the other three methods (SOT/EOT/SROT, in run.py) run in float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from flash_sinkhorn.bench.sinkslot import (
    sot_plan_coo, to_csr, sparse_sqeuclidean_cost, _ot_1d_coo_batched_cuda, _run_v5,
)


@dataclass
class _FixedStop:
    mode: str = "fixed"


def plan_barycentric_sparse(T_vals, rows, cols, x, y):
    """Barycentric projections (Tx, Ty) of a sparse plan given as (rows, cols, T_vals).

    Normalizes by the plan's own achieved marginals (scatter-summed from
    T_vals), not the target a, b -- matters when the solve hasn't fully
    converged. Verbatim copy of bench_forward.py's function of the same name.
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


def slot_grad(X, Y, a, eps, L, seed, n_iters):
    """grad_X SLOT_eps(X, Y), X/Y/a expected fp32 (Triton requires it -- see module docstring)."""
    n, m = X.shape[0], Y.shape[0]
    rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=seed, ot1d=_ot_1d_coo_batched_cuda)
    cost = sparse_sqeuclidean_cost(X, Y, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps
    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)

    log_a = a.log()
    phi, psi, _, _, _ = _run_v5(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                 log_a, log_a, n, m, n_iters, _FixedStop())

    T_vals = (phi[rows] + psi[cols] + lam).exp()
    Tx, _ = plan_barycentric_sparse(T_vals, rows, cols, X, Y)
    return 2.0 * a[:, None] * (X - Tx)
