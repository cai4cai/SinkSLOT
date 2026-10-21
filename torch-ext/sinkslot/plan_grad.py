"""Backward pass through SinkSLOT's transport plan by implicit differentiation.

`sparse_transport_plan(..., differentiable=True)` returns plan values that sit
in the autograd graph. Given the upstream gradient G = dL/dP on the support,
the backward solves the adjoint system at the Sinkhorn fixed point,

    H (alpha, beta) = (W 1, W^T 1),   W = G * P,
    H = [[diag(P 1), P], [P^T, diag(P^T 1)]],

with MINRES (H is symmetric positive semidefinite, singular along the gauge
direction (1, -1)), then returns

    dL/dC_ij = Gamma_ij = -(1/eps) P_ij (G_ij - alpha_i - beta_j),
    dL/dx_i  = 2 sum_j Gamma_ij (x_i - y_j),   dL/dy_j = 2 sum_i Gamma_ij (y_j - x_i),
    dL/da    = alpha,                          dL/db   = beta.

P^SOT and its support are treated as constants: their dependence on the
points is piecewise constant, but their values depend on (a, b), so alpha and
beta are the weight gradients with P^SOT held fixed. They are defined up to
the gauge (alpha + c, beta - c), which does not change the gradient along
perturbations with sum(da) = sum(db). Backward memory is O(|S|).
"""

from __future__ import annotations

import warnings

import torch

from .solver import sparse_sqeuclidean_cost


def _row_sum(vals, idx, size):
    return torch.zeros(size, *vals.shape[1:], device=vals.device, dtype=vals.dtype).index_add_(0, idx, vals)


class _SparsePlanFn(torch.autograd.Function):
    """Plan values on the support from solved potentials; implicit backward."""

    @staticmethod
    def forward(ctx, x, y, a, b, rows, cols, log_S, phi, psi, eps, adjoint_tol, adjoint_max_iter):
        cost = sparse_sqeuclidean_cost(x, y, rows, cols)
        vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
        ctx.save_for_backward(x, y, rows, cols, vals)
        ctx.eps, ctx.adjoint_tol, ctx.adjoint_max_iter = eps, adjoint_tol, adjoint_max_iter
        return vals

    @staticmethod
    def backward(ctx, grad_vals):
        x, y, rows, cols, p = ctx.saved_tensors
        alpha, beta, gamma = adjoint(p, grad_vals, rows, cols, x.shape[0], y.shape[0], ctx.eps,
                                     ctx.adjoint_tol, ctx.adjoint_max_iter)
        grad_x, grad_y = points_grad(gamma, rows, cols, x, y, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        grad_a = alpha if ctx.needs_input_grad[2] else None
        grad_b = beta if ctx.needs_input_grad[3] else None
        return grad_x, grad_y, grad_a, grad_b, None, None, None, None, None, None, None, None


def adjoint(p, grad_vals, rows, cols, n, m, eps, tol, max_iter):
    """(alpha, beta, Gamma) for plan values p and upstream gradient grad_vals on (rows, cols).

    Solves H (alpha, beta) = (W 1, W^T 1), W = grad_vals * p, by MINRES with the
    Jacobi preconditioner diag(P 1, P^T 1)^-1, and returns
    Gamma = dL/dC = -(1/eps) p (grad_vals - alpha_i - beta_j). Warns if the
    relative residual exceeds `tol`.
    """
    try:
        from torchsparsegradutils.utils import minres, MINRESSettings
    except ImportError as e:
        raise ImportError("sparse_transport_plan(differentiable=True) needs torchsparsegradutils "
                          "for its backward (pip install torchsparsegradutils)") from e

    r, c = _row_sum(p, rows, n), _row_sum(p, cols, m)
    W = grad_vals * p
    w = torch.cat([_row_sum(W, rows, n), _row_sum(W, cols, m)])

    def col(t, z):  # broadcast a length-(n or m or n+m) vector against z's trailing RHS dims
        return t.view(-1, *([1] * (z.dim() - 1)))

    def H(z):
        u, v = z[:n], z[n:]
        return torch.cat([col(r, z) * u + _row_sum(col(p, z) * v[cols], rows, n),
                          _row_sum(col(p, z) * u[rows], cols, m) + col(c, z) * v])

    diag = torch.cat([r, c]).clamp_min(torch.finfo(p.dtype).tiny)
    z = minres(H, w, preconditioner=lambda t: t / col(diag, t), max_iter=max_iter,
               settings=MINRESSettings(minres_tolerance=tol))
    residual = float((H(z) - w).norm() / w.norm().clamp_min(torch.finfo(w.dtype).tiny))
    if residual > tol:
        warnings.warn(f"sparse_transport_plan backward: adjoint relative residual {residual:.2e} "
                      f"> adjoint_tol {tol:.1e}", stacklevel=3)
    alpha, beta = z[:n], z[n:]
    gamma = -(p / eps) * (grad_vals - alpha[rows] - beta[cols])
    return alpha, beta, gamma


def points_grad(gamma, rows, cols, x, y, need_x=True, need_y=True):
    """dL/dx, dL/dy from Gamma = dL/dC on (rows, cols), for C_ij = ||x_i - y_j||^2."""
    n, m = x.shape[0], y.shape[0]
    grad_x = grad_y = None
    if need_x:
        grad_x = 2.0 * (_row_sum(gamma, rows, n)[:, None] * x - _row_sum(gamma[:, None] * y[cols], rows, n))
    if need_y:
        grad_y = 2.0 * (_row_sum(gamma, cols, m)[:, None] * y - _row_sum(gamma[:, None] * x[rows], cols, m))
    return grad_x, grad_y


def plan_values(x, y, a, b, rows, cols, log_S, phi, psi, eps, adjoint_tol=1e-6, adjoint_max_iter=1000):
    """Differentiable plan values exp(phi_i + psi_j + log_S_ij - C_ij/eps) on (rows, cols).

    `phi`, `psi` must be converged potentials for (x, y, a, b) on this support;
    they are not differentiated through directly (the backward differentiates
    the fixed point implicitly). `adjoint_tol`, `adjoint_max_iter`: MINRES
    relative tolerance and iteration cap for the adjoint solve.
    """
    return _SparsePlanFn.apply(x, y, a, b, rows, cols, log_S.detach(), phi.detach(), psi.detach(),
                               eps, adjoint_tol, adjoint_max_iter)
