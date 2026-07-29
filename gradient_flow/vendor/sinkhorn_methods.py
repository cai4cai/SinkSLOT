"""Dense SOT/EOT/SROT baselines for the gradient-flow experiment.

Vendored (trimmed to just what gradient_flow/run.py needs) from
mva-internship-2026/SROT's lib/sinkhorn_methods.py -- this repo has no
differentiable EOT/SROT loss of its own (torch-ext/flash_sinkhorn/bench is
benchmark-timing code only), so the dense autograd-through-Sinkhorn baselines
live here instead of being re-derived.
"""
import numpy as np
import ot
import torch


def build_cost(X, Y):
    """Squared Euclidean cost matrix."""
    return ot.dist(X, Y, metric="sqeuclidean")


def exact_ot(a, b, C):
    """Exact LP optimal transport cost (ground truth)."""
    G = ot.emd(a, b, C)
    return float(np.sum(G * C)), G


def build_sot_plan(X, Y, a, b, L=50, delta=1e-8, rng=None):
    """
    Uniform-average SOT reference plan from L random 1D projections.

    Returns (1 - delta) pi_SOT + delta pi_ind with pi_ind = a outer b (product /
    independent coupling). Same marginals as pi_SOT; for delta > 0 every entry is
    strictly positive when a, b > 0 (helps Sinkhorn kernel support).
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    m = Y.shape[0]
    pi_sot = np.zeros((n, m), dtype=np.float64)

    thetas = rng.standard_normal((L, d))
    theta_norms = np.linalg.norm(thetas, axis=1, keepdims=True)
    theta_norms = np.maximum(theta_norms, 1e-300)
    thetas /= theta_norms

    px_all = X @ thetas.T
    py_all = Y @ thetas.T
    emd_1d = ot.emd_1d
    inv_L = 1.0 / L

    for ell in range(L):
        pi_sot += emd_1d(px_all[:, ell], py_all[:, ell], a, b)

    pi_sot *= inv_L
    if delta > 0.0:
        pi_ind = np.outer(a, b)
        pi_sot = (1.0 - delta) * pi_sot + delta * pi_ind
    return pi_sot


def _sinkhorn_std_torch_plan(a_t, b_t, C_t, eps, max_iter=200, tol=1e-9):
    """Differentiable Torch Sinkhorn plan for K=exp(-C/eps)."""
    K = torch.exp(-C_t / eps).clamp_min(1e-12)
    u = torch.ones_like(a_t)
    v = torch.ones_like(b_t)
    for _ in range(max_iter):
        u_prev = u
        u = a_t / (K @ v).clamp_min(1e-12)
        v = b_t / (K.t() @ u).clamp_min(1e-12)
        if tol > 0:
            err = torch.mean(torch.abs(u - u_prev))
            if float(err.detach().cpu().item()) < tol:
                break
    return (u[:, None] * K) * v[None, :]


def _sinkhorn_sot_torch_plan(a_t, b_t, C_t, pi_sot_t, eps, max_iter=200, tol=1e-9):
    """Differentiable Torch Sinkhorn plan for K=pi_sot*exp(-C/eps)."""
    K = (pi_sot_t * torch.exp(-C_t / eps)).clamp_min(1e-12)
    u = torch.ones_like(a_t)
    v = torch.ones_like(b_t)
    for _ in range(max_iter):
        u_prev = u
        u = a_t / (K @ v).clamp_min(1e-12)
        v = b_t / (K.t() @ u).clamp_min(1e-12)
        if tol > 0:
            err = torch.mean(torch.abs(u - u_prev))
            if float(err.detach().cpu().item()) < tol:
                break
    return (u[:, None] * K) * v[None, :]


def sinkhorn_divergence_torch_autograd(X_t, Y_t, eps, max_iter=100, tol=0.0):
    """
    Differentiable Sinkhorn divergence for autograd-based optimization.
    Returns a scalar torch.Tensor.
    """
    n = X_t.shape[0]
    m = Y_t.shape[0]
    a_t = torch.full((n,), 1.0 / n, dtype=X_t.dtype, device=X_t.device)
    b_t = torch.full((m,), 1.0 / m, dtype=Y_t.dtype, device=Y_t.device)

    C_xy = torch.cdist(X_t, Y_t, p=2) ** 2
    C_xx = torch.cdist(X_t, X_t, p=2) ** 2
    C_yy = torch.cdist(Y_t, Y_t, p=2) ** 2

    pi_xy = _sinkhorn_std_torch_plan(a_t, b_t, C_xy, eps=eps, max_iter=max_iter, tol=tol)
    pi_xx = _sinkhorn_std_torch_plan(a_t, a_t, C_xx, eps=eps, max_iter=max_iter, tol=tol)
    pi_yy = _sinkhorn_std_torch_plan(b_t, b_t, C_yy, eps=eps, max_iter=max_iter, tol=tol)

    ot_xy = torch.sum(pi_xy * C_xy)
    ot_xx = torch.sum(pi_xx * C_xx)
    ot_yy = torch.sum(pi_yy * C_yy)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


def sr_sinkhorn_divergence_torch_autograd(
    X_t,
    Y_t,
    eps,
    L=80,
    max_iter=100,
    tol=0.0,
    use_softmax=False,
    temperature=0.05,
    delta=1e-8,
):
    """
    Differentiable SR-Sinkhorn divergence for autograd-based optimization.
    The SOT references are rebuilt each call and treated as constants (stop-gradient).
    """
    n = X_t.shape[0]
    m = Y_t.shape[0]
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m
    a_t = torch.full((n,), 1.0 / n, dtype=X_t.dtype, device=X_t.device)
    b_t = torch.full((m,), 1.0 / m, dtype=Y_t.dtype, device=Y_t.device)

    X_np = X_t.detach().cpu().numpy()
    Y_np = Y_t.detach().cpu().numpy()

    if use_softmax:
        raise NotImplementedError("softmax SOT weighting not vendored here; see the full "
                                   "sinkhorn_methods.py in mva-internship-2026/SROT if needed.")
    pi_sot_xy = build_sot_plan(X_np, Y_np, a, b, L=L, delta=delta)

    # For self terms, the SOT reference is diagonal (identity coupling) for identical supports.
    # We use it directly and only apply the same delta smoothing used elsewhere.
    pi_sot_xx = np.eye(n, dtype=np.float64) / n
    pi_sot_yy = np.eye(m, dtype=np.float64) / m
    if delta > 0.0:
        pi_sot_xx = (1.0 - delta) * pi_sot_xx + delta * np.outer(a, a)
        pi_sot_yy = (1.0 - delta) * pi_sot_yy + delta * np.outer(b, b)

    C_xy = torch.cdist(X_t, Y_t, p=2) ** 2
    C_xx = torch.cdist(X_t, X_t, p=2) ** 2
    C_yy = torch.cdist(Y_t, Y_t, p=2) ** 2
    pi_sot_xy_t = torch.as_tensor(pi_sot_xy, dtype=X_t.dtype, device=X_t.device)
    pi_sot_xx_t = torch.as_tensor(pi_sot_xx, dtype=X_t.dtype, device=X_t.device)
    pi_sot_yy_t = torch.as_tensor(pi_sot_yy, dtype=X_t.dtype, device=X_t.device)

    pi_xy = _sinkhorn_sot_torch_plan(a_t, b_t, C_xy, pi_sot_xy_t, eps=eps, max_iter=max_iter, tol=tol)
    pi_xx = _sinkhorn_sot_torch_plan(a_t, a_t, C_xx, pi_sot_xx_t, eps=eps, max_iter=max_iter, tol=tol)
    pi_yy = _sinkhorn_sot_torch_plan(b_t, b_t, C_yy, pi_sot_yy_t, eps=eps, max_iter=max_iter, tol=tol)

    ot_xy = torch.sum(pi_xy * C_xy)
    ot_xx = torch.sum(pi_xx * C_xx)
    ot_yy = torch.sum(pi_yy * C_yy)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
