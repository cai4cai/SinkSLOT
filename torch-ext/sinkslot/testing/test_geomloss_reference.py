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


def test_online_matches_geomloss_ot_sinkhorn_loop():
    """At a constant eps, geomloss_online is geomloss.ot's sinkhorn_loop without its last extrapolation."""
    from geomloss import _backends as bk
    from geomloss._typing import CostMatrices, DescentParameters
    from geomloss.ot._abstract_solvers import sinkhorn_loop
    from geomloss.ot._implementations.sample import cost_matrix, softmin_sample

    x, y, a, b = _problem(n=2000)
    eps, iters = 0.05, 50
    C = [CostMatrices(xy=cost_matrix(x, y), yx=cost_matrix(y, x), xx=None, yy=None)]
    descent = DescentParameters(scale_list=[0] * iters, eps_list=[eps] * iters, rho_list=[None] * iters)
    ref = sinkhorn_loop(softmin=softmin_sample, log_a_list=[bk.stable_log(a)], log_b_list=[bk.stable_log(b)],
                        C_list=C, descent=descent, debias=False, last_extrapolation=False)
    f, g, *_ = geomloss_online(x, y, a, b, eps, iters)
    assert torch.allclose(f, ref.f_ba.view(-1), rtol=0, atol=1e-5)
    assert torch.allclose(g, ref.g_ab.view(-1), rtol=0, atol=1e-5)


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


def test_truncated_plan_is_feasible_on_its_kept_pairs():
    """Truncation leaves the dropped pairs unconstrained: the plan meets the marginals on the
    kept pairs (support), not on the full kernel."""
    from sinkslot.bench.plan_diagnostics import plan_diagnostics_dense

    x, y, a, b = _problem()
    eps = 0.01
    f, g, _, conv, _, _, info = geomloss_multiscale(x, y, a, b, eps, 50000, threshold=1e-5, truncate=1.0)
    assert conv and info["support"] is not None
    label_x, label_y, keep = info["support"]
    assert label_x.shape == (len(x),) and label_y.shape == (len(y),) and keep.dtype == torch.bool
    viol_kept, *_ = plan_diagnostics_dense(x, y, a, b, eps, f, g, support=info["support"])
    viol_full, *_ = plan_diagnostics_dense(x, y, a, b, eps, f, g)
    assert viol_kept < 1e-5
    assert viol_full > viol_kept


def test_untruncated_run_has_no_support():
    x, y, a, b = _problem(n=2000)
    *_, info = geomloss_multiscale(x, y, a, b, 0.05, 20000, threshold=1e-6, truncate=None)
    assert info["support"] is None and info["kept_fraction"] == 1.0


def test_online_alpha_reaches_the_same_fixed_point():
    """Damping alpha changes the path, not the solution: alpha=0.9 and 0.5 agree at convergence."""
    x, y, a, b = _problem(n=2000)
    f5, g5, it5, c5, cost5, _ = geomloss_online(x, y, a, b, 0.05, 50000, threshold=1e-5, alpha=0.5)
    f9, g9, it9, c9, cost9, _ = geomloss_online(x, y, a, b, 0.05, 50000, threshold=1e-5, alpha=0.9)
    assert c5 and c9 and it9 < it5
    assert abs(cost9 - cost5) <= 1e-4 * abs(cost5)
    assert _gap(f5, g5, f9, g9) < 1e-3


def test_flash_symmetric_alpha_matches_the_library_at_one_half():
    flash = pytest.importorskip("flash_sinkhorn.sinkhorn_solvers")
    from sinkslot.bench.reference_solvers import flash_symmetric_alpha

    x, y, a, b = _problem(n=2000)
    kw = dict(eps=0.05, n_iters=500, allow_tf32=False, threshold=1e-6, check_every=5)
    f1, g1, n1 = flash.sinkhorn_flashstyle_symmetric(x, y, a, b, use_epsilon_scaling=False, last_extrapolation=False,
                                                     return_n_iters=True, **kw)
    f2, g2, n2 = flash_symmetric_alpha(x, y, a, b, 0.5, **kw)
    assert int(n1) == int(n2)
    assert torch.allclose(f1, f2, atol=1e-5) and torch.allclose(g1, g2, atol=1e-5)


def test_multiscale_alpha_reaches_the_same_fixed_point():
    x, y, a, b = _problem(n=3000, d=2)
    out = {al: geomloss_multiscale(x, y, a, b, 0.05, 50000, threshold=1e-5, truncate=None, alpha=al) for al in (0.5, 0.9)}
    (f5, g5, it5, c5, cost5, *_), (f9, g9, it9, c9, cost9, *_) = out[0.5], out[0.9]
    assert c5 and c9
    assert abs(cost5 - cost9) < 1e-4 * abs(cost5)


def test_lift_only_skips_the_fine_stage_when_nothing_is_truncated():
    x, y, a, b = _problem(n=3000, d=2)
    # truncate=None: no truncation, so lift_only returns the lifted coarse solution.
    *_, info_full = geomloss_multiscale(x, y, a, b, 0.05, 50000, threshold=1e-5, truncate=None, alpha=0.9)
    f, g, it, conv, cost, _, info = geomloss_multiscale(x, y, a, b, 0.05, 50000, threshold=1e-5, truncate=None,
                                                        alpha=0.9, lift_only=True)
    assert info["fine_iters"] == 0 and info_full["fine_iters"] > 0
    assert it == info["coarse_iters"] and conv
    assert torch.isfinite(f).all() and torch.isfinite(g).all()
