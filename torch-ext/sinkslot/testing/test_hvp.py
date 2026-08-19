"""Validate hvp_x_sqeuclid against a FROZEN-SUPPORT central finite difference, #18.

hvp_x_sqeuclid differentiates grad_X SLOT_eps treating the sliced-OT support (S,
rows, cols) as fixed -- it only differentiates the cost-dependent lam_ij =
log(S_ij) - cost_ij(X)/eps through cost_ij, not through S itself. That's the
same assumption slot_grad's own envelope-theorem gradient makes (S is
piecewise-constant in X almost everywhere; the paper's own term (II) argument,
reproduced and validated in gradient_flow/finite_diff.py).

A finite difference that calls slot_grad(X+h*v) and slot_grad(X-h*v)
separately does NOT test that assumption fairly: each call rebuilds the
support from scratch via sot_plan_coo, and finite_diff.py already
demonstrates (empirically, in this exact repo) that doing so at any h small
enough to resolve second-order structure picks up O(1/h) rank-flip jump
artifacts that swamp the real signal. So this test builds the support ONCE
at X and reuses it for both perturbed evaluations -- the "FD frozen" arm in
finite_diff.py's own terminology, which is what a correct comparison to
hvp_x_sqeuclid (support held fixed by construction) requires.
"""

import pytest
import torch

triton = pytest.importorskip("triton")
tsgu = pytest.importorskip("torchsparsegradutils")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from sinkslot.solver import (  # noqa: E402
    sot_plan_coo, sparse_sqeuclidean_cost, to_csr,
    _ot_1d_coo_batched_cuda,
)
from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton  # noqa: E402
from sinkslot.gradient import plan_barycentric_sparse  # noqa: E402
from sinkslot.hvp import hvp_x_sqeuclid  # noqa: E402


def _problem(n=300, d=3, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g).cuda()
    y = (torch.randn(n, d, generator=g) + 1.0).cuda()
    a = torch.full((n,), 1.0 / n, device="cuda")
    return x, y, a


def _frozen_grad(X, Y, a, rows, cols, S, eps, n_iters):
    """slot_grad's formula, on a FIXED (rows, cols, S) instead of rebuilding it.

    Mirrors gradient_flow/finite_diff.py's envelope_grad, but through the real
    Triton solver (sinkslot_alternating_triton) rather than the plain-torch loop, so this is
    checking hvp_x_sqeuclid against the exact same numerical path slot_grad itself
    uses -- not a second, independent implementation of the inner solve.
    """
    n, m = X.shape[0], Y.shape[0]
    cost = sparse_sqeuclidean_cost(X, Y, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps
    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n, narrow_key=True)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m, narrow_key=True)
    log_a = a.log()
    phi, psi, _, _, _ = sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                 log_a, log_a, n, m, n_iters)
    T_vals = (phi[rows] + psi[cols] + lam).exp()
    Tx, _ = plan_barycentric_sparse(T_vals, rows, cols, X, Y)
    return 2.0 * a[:, None] * (X - Tx)


def test_slot_hvp_matches_frozen_support_finite_difference():
    X, Y, a = _problem()
    eps, L, seed, n_iters = 0.05, 40, 0, 5000
    g = torch.Generator(device="cpu").manual_seed(1)
    v = torch.randn(X.shape, generator=g).cuda()
    v = v / v.norm()

    hvp = hvp_x_sqeuclid(X, Y, a, eps, L, seed, n_iters, v)

    rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=seed, ot1d=_ot_1d_coo_batched_cuda)
    h = 1e-3
    g_plus = _frozen_grad(X + h * v, Y, a, rows, cols, S, eps, n_iters)
    g_minus = _frozen_grad(X - h * v, Y, a, rows, cols, S, eps, n_iters)
    fd = (g_plus - g_minus) / (2 * h)

    rel_err = float((hvp - fd).norm() / fd.norm())
    print(f"\n|hvp|={float(hvp.norm()):.4e}  |fd|={float(fd.norm()):.4e}  "
          f"rel_err={rel_err:.3e}")
    assert rel_err < 0.08, f"hvp_x_sqeuclid disagrees with the frozen-support FD: {rel_err:.3e}"


def test_slot_hvp_matches_frozen_support_finite_difference_at_two_step_sizes():
    """A second h, to make sure the agreement isn't a fluke of one h."""
    X, Y, a = _problem(n=200, seed=2)
    eps, L, seed, n_iters = 0.08, 30, 0, 5000
    g = torch.Generator(device="cpu").manual_seed(3)
    v = torch.randn(X.shape, generator=g).cuda()
    v = v / v.norm()

    hvp = hvp_x_sqeuclid(X, Y, a, eps, L, seed, n_iters, v)
    rows, cols, S = sot_plan_coo(X, Y, a, a, L=L, seed=seed, ot1d=_ot_1d_coo_batched_cuda)

    for h in (1e-3, 3e-4):
        g_plus = _frozen_grad(X + h * v, Y, a, rows, cols, S, eps, n_iters)
        g_minus = _frozen_grad(X - h * v, Y, a, rows, cols, S, eps, n_iters)
        fd = (g_plus - g_minus) / (2 * h)
        rel_err = float((hvp - fd).norm() / fd.norm())
        print(f"\nh={h:g}  rel_err={rel_err:.3e}")
        assert rel_err < 0.08, f"h={h:g}: hvp_x_sqeuclid disagrees: {rel_err:.3e}"


def test_slot_hvp_is_linear_in_v():
    """H(X)(2v) should be 2*H(X)v -- a cheap internal-consistency check."""
    X, Y, a = _problem(n=150, seed=4)
    eps, L, seed, n_iters = 0.05, 30, 0, 5000
    g = torch.Generator(device="cpu").manual_seed(5)
    v = torch.randn(X.shape, generator=g).cuda()

    hvp1 = hvp_x_sqeuclid(X, Y, a, eps, L, seed, n_iters, v)
    hvp2 = hvp_x_sqeuclid(X, Y, a, eps, L, seed, n_iters, 2.0 * v)

    rel_err = float((hvp2 - 2.0 * hvp1).norm() / hvp1.norm())
    assert rel_err < 1e-2, f"not linear in v: rel_err={rel_err:.3e}"
