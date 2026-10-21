"""Implicit backward of sparse_transport_plan(differentiable=True).

The support (rows, cols, S) is built once and frozen, as the backward assumes:
its dependence on the points is piecewise constant, and finite differences
across a rank flip would measure the jump, not the gradient. Each evaluation
re-solves Sinkhorn to convergence on that support in float64 on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torchsparsegradutils")

from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost  # noqa: E402
from sinkslot.sinkhorn_solvers import sinkslot_alternating_torch, sparse_transport_plan  # noqa: E402
from sinkslot.plan_grad import plan_values  # noqa: E402

DT = torch.float64
EPS = 0.5
STOP = SimpleNamespace(mode="potential", max_iter=10**6, tol=1e-15, check_every=5)


def _problem(n=6, m=7, d=2, L=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=DT)
    y = torch.randn(m, d, generator=g, dtype=DT) + 0.5
    a = torch.rand(n, generator=g, dtype=DT) + 0.5
    b = torch.rand(m, generator=g, dtype=DT) + 0.5
    a, b = a / a.sum(), b / b.sum()
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=seed)
    return x, y, a, b, rows, cols, S.log()


def _frozen_plan(rows, cols, log_S):
    """Plan values on a fixed support, re-solved to convergence at each call."""
    def vals(x, y, a, b):
        n, m = x.shape[0], y.shape[0]
        with torch.no_grad():
            lam = log_S - sparse_sqeuclidean_cost(x, y, rows, cols) / EPS
            phi, psi, *_ = sinkslot_alternating_torch(rows, cols, lam, a.log(), b.log(), n, m, 0, STOP, EPS)
        return plan_values(x, y, a, b, rows, cols, log_S, phi, psi, EPS,
                           adjoint_tol=1e-13, adjoint_max_iter=500)
    return vals


def test_gradcheck_points():
    x, y, a, b, rows, cols, log_S = _problem()
    vals = _frozen_plan(rows, cols, log_S)
    xg, yg = x.clone().requires_grad_(), y.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda X, Y: vals(X, Y, a, b), (xg, yg), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mass_change", [0.0, 0.3])
def test_weight_gradients_along_balanced_directions(mass_change):
    """alpha, beta are defined up to the gauge (c, -c), so only directions with
    sum(da) = sum(db) are compared against finite differences."""
    x, y, a, b, rows, cols, log_S = _problem()
    vals = _frozen_plan(rows, cols, log_S)
    g = torch.Generator().manual_seed(1)
    R = torch.randn(len(rows), generator=g, dtype=DT)
    ag, bg = a.clone().requires_grad_(), b.clone().requires_grad_()
    (vals(x, y, ag, bg) * R).sum().backward()
    da = torch.randn(len(a), generator=g, dtype=DT)
    db = torch.randn(len(b), generator=g, dtype=DT)
    da = da - da.mean() + mass_change / len(a)
    db = db - db.mean() + mass_change / len(b)
    h = 1e-6
    fd = ((vals(x, y, a + h * da, b + h * db) * R).sum()
          - (vals(x, y, a - h * da, b - h * db) * R).sum()) / (2 * h)
    torch.testing.assert_close(ag.grad @ da + bg.grad @ db, fd, rtol=1e-6, atol=1e-8)


def test_constant_upstream_gradient_gives_zero():
    """sum(P) is fixed by the marginals, so G = 1 must give zero gradients."""
    x, y, a, b, rows, cols, log_S = _problem()
    xg, yg = x.clone().requires_grad_(), y.clone().requires_grad_()
    _frozen_plan(rows, cols, log_S)(xg, yg, a, b).sum().backward()
    assert xg.grad.abs().max() < 1e-12 and yg.grad.abs().max() < 1e-12


def test_translation_invariance():
    x, y, a, b, rows, cols, log_S = _problem()
    xg, yg = x.clone().requires_grad_(), y.clone().requires_grad_()
    R = torch.randn(len(rows), generator=torch.Generator().manual_seed(2), dtype=DT)
    (_frozen_plan(rows, cols, log_S)(xg, yg, a, b) * R).sum().backward()
    assert (xg.grad.sum(0) + yg.grad.sum(0)).abs().max() < 1e-12


def test_matches_unrolled_autograd():
    """Autograd through enough Sinkhorn iterations to converge reaches the same gradient."""
    x, y, a, b, rows, cols, log_S = _problem(n=40, m=50, d=3, L=8)
    n, m = len(x), len(y)
    R = torch.randn(len(rows), generator=torch.Generator().manual_seed(3), dtype=DT)
    xg, yg = x.clone().requires_grad_(), y.clone().requires_grad_()
    (_frozen_plan(rows, cols, log_S)(xg, yg, a, b) * R).sum().backward()

    xu, yu = x.clone().requires_grad_(), y.clone().requires_grad_()
    lam = log_S - sparse_sqeuclidean_cost(xu, yu, rows, cols) / EPS
    phi, psi, *_ = sinkslot_alternating_torch(rows, cols, lam, a.log(), b.log(), n, m, 3000)
    ((phi[rows] + psi[cols] + lam).exp() * R).sum().backward()

    torch.testing.assert_close(xg.grad, xu.grad, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(yg.grad, yu.grad, rtol=1e-6, atol=1e-9)


def test_sparse_transport_plan_differentiable():
    x, y, *_ = _problem(n=30, m=30, d=2)
    kw = dict(eps=EPS, L=8, stop_mode="potential", stop_tol=1e-12, stop_max_iter=10**5)
    P0 = sparse_transport_plan(x, y, **kw)
    xg, yg = x.clone().requires_grad_(), y.clone().requires_grad_()
    P = sparse_transport_plan(xg, yg, differentiable=True, **kw)
    torch.testing.assert_close(P.indices(), P0.indices())
    torch.testing.assert_close(P.values().detach(), P0.values())
    V = torch.randn(len(y), 4, dtype=DT, generator=torch.Generator().manual_seed(4))
    torch.sparse.mm(P, V).square().sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    assert yg.grad is not None and torch.isfinite(yg.grad).all()
    assert not sparse_transport_plan(xg, yg, **kw).values().requires_grad


def test_cost_gradient_has_zero_row_and_column_sums():
    """Gamma 1 = H's first block row minus the right-hand side, so it vanishes when the adjoint solve is exact."""
    from sinkslot.plan_grad import adjoint

    x, y, a, b, rows, cols, log_S = _problem(n=40, m=50, d=3, L=8)
    p = _frozen_plan(rows, cols, log_S)(x, y, a, b)
    G = torch.randn(len(rows), generator=torch.Generator().manual_seed(5), dtype=DT)
    _, _, gamma = adjoint(p, G, rows, cols, len(x), len(y), EPS, 1e-13, 500)
    scale = (p * G.abs()).sum() / EPS
    assert torch.zeros(len(x), dtype=DT).index_add_(0, rows, gamma).abs().max() < 1e-10 * scale
    assert torch.zeros(len(y), dtype=DT).index_add_(0, cols, gamma).abs().max() < 1e-10 * scale
