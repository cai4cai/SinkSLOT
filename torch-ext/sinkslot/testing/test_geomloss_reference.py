"""reference_solvers.geomloss_online and geomloss_multiscale (CUDA + KeOps)."""

import pytest
import torch

pytest.importorskip("geomloss")
pytest.importorskip("pykeops")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KeOps block-sparse reductions need CUDA")

from sinkslot.bench.reference_solvers import geomloss_multiscale, geomloss_online  # noqa: E402


def _problem(n=4000, d=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g).cuda()
    y = (torch.randn(n, d, generator=g) * 0.7 + 0.5).cuda()
    a = torch.full((n,), 1 / n, device="cuda")
    return x, y, a, a.clone()


def _gap(f1, g1, f2, g2):
    """Largest potential difference after removing the (f + c, g - c) gauge."""
    c = float((f1 - f2).mean())
    return max(float((f1 - f2 - c).abs().max()), float((g1 - g2 + c).abs().max()))


def test_online_matches_the_legacy_kernels():
    """geomloss.ot's cost_matrix + softmin_sample give the legacy lse_genred + softmin_online updates."""
    from functools import partial

    from geomloss._legacy.sinkhorn_samples import lse_genred, softmin_online

    x, y, a, b = _problem(n=2000)
    eps, iters = 0.05, 50
    softmin = partial(softmin_online, log_conv=lse_genred("SqDist(X,Y)", 2))
    a_log, b_log = a.log(), b.log()
    g = softmin(eps, (y, x), a_log)
    f = softmin(eps, (x, y), b_log)
    for _ in range(iters):
        ft, gt = softmin(eps, (x, y), b_log + g / eps), softmin(eps, (y, x), a_log + f / eps)
        f, g = 0.5 * (f + ft), 0.5 * (g + gt)
    f_new, g_new, *_ = geomloss_online(x, y, a, b, eps, iters)
    assert torch.allclose(f_new, f.view(-1), rtol=0, atol=1e-5)
    assert torch.allclose(g_new, g.view(-1), rtol=0, atol=1e-5)


def test_without_truncation_it_reaches_the_online_fixed_point():
    x, y, a, b = _problem()
    eps, tol = 0.05, 1e-6
    fo, go, _, conv_o, cost_o, _ = geomloss_online(x, y, a, b, eps, 20000, threshold=tol)
    fm, gm, _, conv_m, cost_m, _, info = geomloss_multiscale(x, y, a, b, eps, 20000, threshold=tol, truncate=None)
    assert conv_o and conv_m
    assert info["kept_fraction"] == 1.0
    assert abs(cost_m - cost_o) <= 1e-4 * abs(cost_o)
    assert _gap(fo, go, fm, gm) < 1e-3


def test_truncation_keeps_a_fraction_of_the_pairs():
    # Truncation applies when the grid cells are wider than the blur: cluster_scale**2 > eps / 2.
    x, y, a, b = _problem()
    eps, tol = 0.01, 1e-5
    _, _, _, _, cost_o, _ = geomloss_online(x, y, a, b, eps, 50000, threshold=tol)
    f, g, iters, conv, cost, _, info = geomloss_multiscale(x, y, a, b, eps, 50000, threshold=tol)
    assert info["cluster_scale"] ** 2 > eps / 2
    assert conv and iters == info["coarse_iters"] + info["fine_iters"]
    assert 0 < info["kept_fraction"] < 1
    # Truncating from the coarse potentials is an approximation: the cost is close, not equal.
    assert abs(cost - cost_o) <= 2e-2 * abs(cost_o)
    assert f.shape == (len(x),) and g.shape == (len(y),)


def test_potentials_come_back_in_input_order():
    x, y, a, b = _problem(n=2000)
    perm = torch.randperm(len(x), device="cuda")
    f1, *_ = geomloss_multiscale(x, y, a, b, 0.05, 20000, threshold=1e-6, truncate=None)
    f2, *_ = geomloss_multiscale(x[perm], y, a, b, 0.05, 20000, threshold=1e-6, truncate=None)
    assert float((f1[perm] - f2 - (f1[perm] - f2).mean()).abs().max()) < 1e-3


def test_rejects_more_than_three_dimensions():
    x, y, a, b = _problem(n=100, d=4)
    with pytest.raises(ValueError, match="d <= 3"):
        geomloss_multiscale(x, y, a, b, 0.1, 10)
