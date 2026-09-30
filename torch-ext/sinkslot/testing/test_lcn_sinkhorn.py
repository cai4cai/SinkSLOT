"""sinkslot.bench.lcn_sinkhorn against dense references (CPU, small problems)."""

import torch

from sinkslot.bench.lcn_sinkhorn import _sqdist, lcn_factors, lcn_sinkhorn


def _problem(n=120):
    torch.manual_seed(0)
    x, y = torch.randn(n, 2), torch.randn(n, 2) + 0.5
    a, b = torch.rand(n) + 0.1, torch.rand(n) + 0.1
    return x, y, a / a.sum(), b / b.sum()


def _dense_kernel(f, n):
    k = (f.sign_v[None] * torch.exp(f.log_u[:, :, None] + f.log_v[None])).sum(1)
    k[f.rows, f.cols] += f.sign_s * torch.exp(f.log_s)
    return k


def test_all_points_as_landmarks_recover_the_kernel():
    x, y, _, _ = _problem()
    eps = 1.0
    f = lcn_factors(x, y, eps, landmarks=2 * x.shape[0], neighbors=4)
    k = torch.exp(-_sqdist(x.double(), y.double()) / eps)
    assert float((k - _dense_kernel(f, x.shape[0])).norm() / k.norm()) < 1e-3


def test_iterates_match_dense_sinkhorn_on_the_lcn_kernel():
    x, y, a, b = _problem()
    eps, n_iter = 1.0, 50
    f = lcn_factors(x, y, eps, landmarks=2 * x.shape[0], neighbors=4)
    k = _dense_kernel(f, x.shape[0])
    u, v = torch.zeros(x.shape[0], dtype=torch.float64), torch.ones(x.shape[0], dtype=torch.float64)
    for _ in range(n_iter):
        u = a.double() / (k @ v)
        v = b.double() / (k.T @ u)
    res = lcn_sinkhorn(f, a, b, eps, max_iter=n_iter, tol=0.0, check_every=n_iter + 1)
    assert not res.failed
    assert torch.allclose(res.u, u.log(), atol=1e-8)
    assert torch.allclose(res.v, v.log(), atol=1e-8)
