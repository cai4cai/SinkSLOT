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
        cost = sparse_sqeuclidean_cost(x, y, rows, cols, use_triton=None if x.dtype == torch.float32 else False)
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


def adjoint(p, grad_vals, rows, cols, n, m, eps, tol, max_iter, w=None):
    """(alpha, beta, Gamma) for plan values p and upstream gradient grad_vals on (rows, cols).

    Solves H (alpha, beta) = w, w = (W 1, W^T 1) with W = grad_vals * p unless
    given, through its Jacobi-scaled Schur complement on the smaller side, and
    returns Gamma = dL/dC = -(1/eps) p (grad_vals - alpha_i - beta_j).

    With r = P 1, c = P^T 1 and Pt = diag(r)^-1/2 P diag(c)^-1/2, for m <= n
    MINRES solves (I - Pt^T Pt) beta' = c^-1/2 * w_t - Pt^T (r^-1/2 * w_s), then
    alpha' = r^-1/2 * w_s - Pt beta', alpha = r^-1/2 * alpha', beta = c^-1/2 * beta'
    (sides swapped for n < m). The operator's kernel, the gauge, is u = c^1/2:
    the right-hand side is projected orthogonally to u and u u^T / ||u||^2 is
    added to the operator, which makes it positive definite with condition
    number 1 / (1 - sigma_2^2). MINRES stops on its relative update size, so the
    solve is refined on the remaining residual up to 3 times until
    ||H z - w|| / ||w|| <= `tol`; it warns if that is not reached (float32
    bottoms out near 1e-6).
    """
    try:
        from torchsparsegradutils.utils import minres, MINRESSettings
    except ImportError as e:
        raise ImportError("sparse_transport_plan(differentiable=True) needs torchsparsegradutils "
                          "for its backward (pip install torchsparsegradutils)") from e

    r, c = _row_sum(p, rows, n), _row_sum(p, cols, m)
    if w is None:
        W = grad_vals * p
        w = torch.cat([_row_sum(W, rows, n), _row_sum(W, cols, m)])

    def col(t, z):  # broadcast a length-(n or m or n+m) vector against z's trailing RHS dims
        return t.view(-1, *([1] * (z.dim() - 1)))

    def H(z):
        u, v = z[:n], z[n:]
        return torch.cat([col(r, z) * u + _row_sum(col(p, z) * v[cols], rows, n),
                          _row_sum(col(p, z) * u[rows], cols, m) + col(c, z) * v])

    tiny = torch.finfo(p.dtype).tiny
    ds, dt = r.clamp_min(tiny).rsqrt(), c.clamp_min(tiny).rsqrt()
    pt = p * ds[rows] * dt[cols]  # Jacobi-scaled plan values
    if m <= n:  # unknowns on the target side
        s_idx, b_idx, d_small, d_big, n_small, n_big = cols, rows, dt, ds, m, n
    else:
        s_idx, b_idx, d_small, d_big, n_small, n_big = rows, cols, ds, dt, n, m
    u = d_small.reciprocal()  # kernel: (P^T 1)^1/2 or (P 1)^1/2
    u = u / u.norm()

    def to_big(x):  # Pt x, x on the small side
        return _row_sum(col(pt, x) * x[s_idx], b_idx, n_big)

    def to_small(y):  # Pt^T y, y on the big side
        return _row_sum(col(pt, y) * y[b_idx], s_idx, n_small)

    def schur(x):  # (I - Pt^T Pt + u u^T) x
        return x - to_small(to_big(x)) + col(u, x) * (col(u, x) * x).sum(0, keepdim=True)

    def solve(rhs):  # H z = rhs through the Schur complement
        w_s, w_t = rhs[:n], rhs[n:]
        w_small, w_big = (w_t, w_s) if m <= n else (w_s, w_t)
        w_small, w_big = d_small * w_small, d_big * w_big
        b = w_small - to_small(w_big)
        b = b - u * (u @ b)
        x_small = minres(schur, b, max_iter=max_iter, settings=MINRESSettings(minres_tolerance=tol))
        x_big = w_big - to_big(x_small)
        x_small, x_big = d_small * x_small, d_big * x_big
        return torch.cat([x_big, x_small]) if m <= n else torch.cat([x_small, x_big])

    w_norm = w.norm().clamp_min(tiny)
    z = torch.zeros_like(w)
    for _ in range(4):  # MINRES, then up to 3 refinements on the remaining residual
        res = w - H(z)
        residual = float(res.norm() / w_norm)
        if residual <= tol:
            break
        z = z + solve(res)
    residual = float((w - H(z)).norm() / w_norm)
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


def plan_values(x, y, a, b, rows, cols, log_S, phi, psi, eps, adjoint_tol=1e-5, adjoint_max_iter=1000):
    """Differentiable plan values exp(phi_i + psi_j + log_S_ij - C_ij/eps) on (rows, cols).

    `phi`, `psi` must be converged potentials for (x, y, a, b) on this support;
    they are not differentiated through directly (the backward differentiates
    the fixed point implicitly). `adjoint_tol`, `adjoint_max_iter`: MINRES
    relative tolerance and iteration cap for the adjoint solve.
    """
    return _SparsePlanFn.apply(x, y, a, b, rows, cols, log_S.detach(), phi.detach(), psi.detach(),
                               eps, adjoint_tol, adjoint_max_iter)


def _plan_matmul(p, rows, cols, n, B, chunk):
    """sum_j P_ij B_j by chunks of the support, as (n, k)."""
    out = torch.zeros(n, B.shape[1], device=B.device, dtype=B.dtype)
    for s in range(0, len(p), chunk):
        sl = slice(s, s + chunk)
        out.index_add_(0, rows[sl], p[sl, None] * B[cols[sl]])
    return out


def _sampled_dot(A, B, rows, cols, chunk):
    """<A_i, B_j> for each support entry (i, j)."""
    return torch.cat([(A[rows[s:s + chunk]] * B[cols[s:s + chunk]]).sum(1) for s in range(0, len(rows), chunk)])


class _SparsePlanMatmulFn(torch.autograd.Function):
    """O = P V from solved potentials; implicit backward without forming dL/dP."""

    @staticmethod
    def forward(ctx, x, y, a, b, V, rows, cols, log_S, phi, psi, eps, adjoint_tol, adjoint_max_iter, chunk):
        cost = sparse_sqeuclidean_cost(x, y, rows, cols, use_triton=None if x.dtype == torch.float32 else False)
        p = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
        O = _plan_matmul(p, rows, cols, x.shape[0], V, chunk)
        ctx.save_for_backward(x, y, V, rows, cols, p, O)
        ctx.eps, ctx.adjoint_tol, ctx.adjoint_max_iter, ctx.chunk = eps, adjoint_tol, adjoint_max_iter, chunk
        return O

    @staticmethod
    def backward(ctx, grad_O):
        x, y, V, rows, cols, p, O = ctx.saved_tensors
        n, m = x.shape[0], y.shape[0]
        grad_V = _plan_matmul(p, cols, rows, m, grad_O, ctx.chunk)  # P^T dL/dO
        # dL/dP_ij = <dL/dO_i, V_j>; its p-weighted row and column sums are
        # rowsum(dL/dO * O) and rowsum(V * P^T dL/dO).
        w = torch.cat([(grad_O * O).sum(1), (V * grad_V).sum(1)])
        G = _sampled_dot(grad_O, V, rows, cols, ctx.chunk)
        alpha, beta, gamma = adjoint(p, G, rows, cols, n, m, ctx.eps, ctx.adjoint_tol, ctx.adjoint_max_iter, w=w)
        grad_x, grad_y = points_grad(gamma, rows, cols, x, y, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        grad_a = alpha if ctx.needs_input_grad[2] else None
        grad_b = beta if ctx.needs_input_grad[3] else None
        grad_V = grad_V if ctx.needs_input_grad[4] else None
        return grad_x, grad_y, grad_a, grad_b, grad_V, *([None] * 9)


def plan_matmul(x, y, a, b, V, rows, cols, log_S, phi, psi, eps, adjoint_tol=1e-5, adjoint_max_iter=1000,
                chunk=2**20):
    """Differentiable P @ V for the plan of `plan_values`, with O(|S|) backward memory.

    Gradients flow to x, y, a, b (as in `plan_values`) and V. `chunk` bounds the
    support entries processed at once, so transients stay O(chunk * V.shape[1]).
    """
    return _SparsePlanMatmulFn.apply(x, y, a, b, V, rows, cols, log_S.detach(), phi.detach(), psi.detach(),
                                     eps, adjoint_tol, adjoint_max_iter, chunk)
