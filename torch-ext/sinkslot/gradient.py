"""grad_X SLOT_eps by the envelope theorem.

Split out from solver.py to mirror flash_sinkhorn's own file-per-concept
layout. NOT named implicit_grad.py to match flash_sinkhorn's implicit_grad.py:
`slot_grad` below uses the envelope theorem (Feydy et al. 2019), a direct
closed-form formula from the converged plan -- it does not differentiate the
Sinkhorn fixed point implicitly (no linear solve, no Jacobian). That's a
genuinely different technique from flash_sinkhorn's `implicit_grad_x`, which
does use the Implicit Function Theorem; giving this file or function that name
would misrepresent which method it runs. `hvp.py`'s `hvp_x_sqeuclid` is
SinkSLOT's actual implicit-differentiation counterpart (second-order, not
this file's first-order gradient).
"""

from __future__ import annotations

import torch

from .solver import sparse_sqeuclidean_cost
from .sinkhorn_solvers import sinkslot_solve


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
