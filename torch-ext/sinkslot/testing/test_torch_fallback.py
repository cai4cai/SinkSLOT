"""Tests for the pure-torch fallback (`sinkslot_alternating_torch`, `sinkslot_solve`).

Every test here except one is deliberately unmarked -- no
`skipif(not torch.cuda.is_available())`, no `pytest.importorskip("triton")` --
because the whole point of this module is that it runs on a plain CPU machine
with no Triton installed, which every other test file in this directory
assumes it doesn't have to handle. The one exception,
`test_sinkslot_solve_all_three_settings_agree_on_cuda`, genuinely needs a GPU:
it's specifically checking the torch backend's numerics ON a CUDA tensor
(distinct from Triton, and distinct from the torch backend on a CPU tensor,
which the rest of this file already covers) -- there's no way to attempt that
without one.
"""

from dataclasses import dataclass

import pytest
import torch

from sinkslot.solver import (
    sot_plan_coo,
    sparse_sqeuclidean_cost,
)
from sinkslot.sinkhorn_solvers import (
    sinkslot_alternating_torch,
    sinkslot_symmetric_torch,
    _seg_lse_coo,
    sinkslot_solve,
)


@dataclass
class _Stop:
    mode: str = "fixed"
    max_iter: int = 20000
    tol: float = 1e-6
    check_every: int = 5


def _problem(n=300, m=250, d=3, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = (torch.randn(m, d, generator=g) + 1.0)
    a = (torch.rand(n, generator=g) + 0.1)
    b = (torch.rand(m, generator=g) + 0.1)
    return x, y, a / a.sum(), b / b.sum()


def test_no_triton_import_required():
    """Importing and using sinkslot must not require Triton to be installed."""
    from sinkslot import solver
    # Whatever the environment's Triton availability actually is, the module
    # must have imported successfully to get this far -- that's the assertion.
    assert hasattr(solver, "_HAS_TRITON")


def test_sparse_sqeuclidean_cost_matches_dense_on_cpu():
    x, y, a, b = _problem()
    rows, cols, S = sot_plan_coo(x, y, a, b, L=20, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    ref = (x[rows] - y[cols]).square().sum(1)
    assert torch.allclose(cost, ref)


def test_sinkslot_alternating_torch_fixed_mode_gives_correct_marginals():
    # "fixed" mode has no convergence criterion -- it just runs `iters` times --
    # so this needs enough iterations to actually be near the fixed point,
    # unlike the marginal/potential tests below which stop on their own
    # explicit tolerance and don't depend on picking "enough" iterations.
    n, m, eps, L, iters = 300, 250, 0.05, 40, 3000
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi, psi, it, converged, viol = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, iters)
    assert it == iters and converged is None and viol is None

    P = (phi[rows] + psi[cols] + lam).exp()
    row_marg = torch.zeros(n).index_add_(0, rows, P)
    col_marg = torch.zeros(m).index_add_(0, cols, P)
    # Column marginal is exact after the last (column) half-step by
    # construction; row marginal is the one still converging.
    assert torch.allclose(col_marg, b, atol=1e-6)
    assert torch.allclose(row_marg, a, atol=1e-4)


def test_sinkslot_alternating_torch_marginal_mode_converges():
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi_m, psi_m, it_m, conv_m, viol_m = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000, _Stop(mode="marginal"))
    assert conv_m and viol_m <= 1e-6


def test_sinkslot_alternating_torch_potential_mode_converges():
    """stop.mode == "potential" (renamed from "potential_linf", the only
    genuinely distinct third mode -- the old "potential" was byte-for-byte
    identical to "marginal" and was dropped, see issue #47).
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi, psi, it, converged, change = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000,
        _Stop(mode="potential", tol=1e-4), eps=eps)
    assert converged and change < 1e-4


def test_sinkslot_alternating_torch_rejects_unknown_mode():
    n, m, eps, L = 100, 80, 0.1, 20
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    with pytest.raises(ValueError, match="unknown stop.mode"):
        sinkslot_alternating_torch(rows, cols, lam, log_a, log_b, n, m, 100,
                      _Stop(mode="bogus", max_iter=100))


def test_sinkslot_solve_runs_end_to_end_on_cpu():
    """One call, no CUDA, no Triton, correct answer."""
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    assert not x.is_cuda

    phi, psi, rows, cols, S, it, converged, viol = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000,
        stop_mode="marginal", stop_max_iter=20000)
    assert converged and viol <= 1e-6

    P = (phi[rows] + psi[cols] +
         (S.clamp_min(torch.finfo(S.dtype).tiny).log() -
          sparse_sqeuclidean_cost(x, y, rows, cols) / eps)).exp()
    row_marg = torch.zeros(n).index_add_(0, rows, P)
    col_marg = torch.zeros(m).index_add_(0, cols, P)
    assert torch.allclose(row_marg, a, atol=1e-5)
    assert torch.allclose(col_marg, b, atol=1e-5)


def test_sparse_transport_plan_matches_manual_sinkslot_solve_construction():
    """sparse_transport_plan wraps sinkslot_solve + the standard plan_vals formula;
    must return a sparse (n, m) COO tensor whose marginals match a/b and
    whose dense form agrees exactly with building the same plan by hand.
    """
    from sinkslot import sparse_transport_plan

    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    P = sparse_transport_plan(x, y, a, b, eps=eps, L=L, seed=0, n_iters=20000,
                        stop_mode="marginal", stop_max_iter=20000, backend="torch")
    assert P.is_sparse and P.shape == (n, m)

    row_marg = torch.sparse.sum(P, dim=1).to_dense()
    col_marg = torch.sparse.sum(P, dim=0).to_dense()
    assert torch.allclose(row_marg, a, atol=1e-5)
    assert torch.allclose(col_marg, b, atol=1e-5)

    phi, psi, rows, cols, S, *_ = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000,
        stop_mode="marginal", stop_max_iter=20000, backend="torch")
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    plan_vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    manual = torch.zeros(n, m).index_put_((rows, cols), plan_vals, accumulate=True)
    assert torch.allclose(P.to_dense(), manual, atol=1e-6)

    # a/b default to uniform when omitted.
    P_default = sparse_transport_plan(x, y, eps=eps, L=L, seed=0, n_iters=200, backend="torch")
    assert P_default.shape == (n, m)


def test_sparse_barycentric_map_sparse_tensor_form_matches_explicit_coo():
    """sparse_barycentric_map(P, x, y) (P a sparse_coo_tensor) must agree
    exactly with the explicit (T_vals, rows, cols, x, y) form it wraps --
    the two are meant to be interchangeable, not just numerically close.
    """
    from sinkslot import sparse_transport_plan, sparse_barycentric_map

    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    P = sparse_transport_plan(x, y, a, b, eps=eps, L=L, seed=0, n_iters=2000,
                               backend="torch")
    rows, cols = P.indices()
    Tx_explicit, Ty_explicit = sparse_barycentric_map(P.values(), rows, cols, x, y)
    Tx_sparse, Ty_sparse = sparse_barycentric_map(P, x, y)
    assert torch.equal(Tx_explicit, Tx_sparse)
    assert torch.equal(Ty_explicit, Ty_sparse)

    with pytest.raises(TypeError, match="expects"):
        sparse_barycentric_map(P, x)


def test_sinkslot_solve_backend_override_on_cpu():
    """backend='torch' and backend='auto' must agree on CPU (both pick torch),
    backend='triton' must raise cleanly (no CUDA here), and a bogus backend
    string must raise too -- the API surface this issue actually asks for.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    auto = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000,
                           stop_mode="marginal", stop_max_iter=20000, backend="auto")
    torch_backend = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000,
                                    stop_mode="marginal", stop_max_iter=20000, backend="torch")
    assert torch.equal(auto[0], torch_backend[0]) and torch.equal(auto[1], torch_backend[1])

    with pytest.raises(ValueError, match="backend='triton'"):
        sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=200, backend="triton")
    with pytest.raises(ValueError, match="backend must be"):
        sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=200, backend="bogus")


def test_sinkslot_solve_variant_default_matches_explicit_alternating():
    """variant='alternating' is the default -- omitting it must be identical
    to passing it explicitly, not just "close" (every existing caller omits
    it, so this is the compatibility guarantee).
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    default = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=500, backend="torch")
    explicit = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=500, backend="torch",
                               variant="alternating")
    assert torch.equal(default[0], explicit[0]) and torch.equal(default[1], explicit[1])

    with pytest.raises(ValueError, match="variant must be"):
        sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=200, variant="bogus")


def test_sinkslot_solve_rejects_unequal_total_mass():
    """sum(a) != sum(b) is mathematically ill-posed and previously produced a
    silently wrong plan -- now checked upfront, matching every other
    argument-validation error sinkslot_solve raises.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    with pytest.raises(ValueError, match="equal total mass"):
        sinkslot_solve(x, y, a, b * 2.0, eps, L, seed=0, n_iters=10)

    # Equal mass but not each summing to 1 is fine (a well-posed, just
    # non-normalized problem) -- only the *mismatch* between a and b raises.
    sinkslot_solve(x, y, a * 3.0, b * 3.0, eps, L, seed=0, n_iters=10)


def test_sinkslot_solve_returns_namedtuple_matching_positional_unpack():
    """SinkslotSolveResult is a NamedTuple: named-field access must agree
    with plain positional unpacking/indexing, so it's a strict addition,
    not a behavior change for existing callers.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)

    result = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=10)
    phi, psi, rows, cols, S, it, converged, viol = result
    assert result.phi is phi and result[0] is phi
    assert result.psi is psi and result[1] is psi
    assert result.rows is rows and result.cols is cols and result.S is S
    assert result.iters_run == it == 10
    assert result.converged is converged is None
    assert result.final_viol is viol is None


def test_sinkslot_symmetric_torch_fixed_mode_gives_correct_marginals():
    """Same shape of test as alternating's own fixed-mode marginal check --
    enough iterations to be near the fixed point, since "fixed" mode has no
    convergence criterion of its own.
    """
    n, m, eps, L, iters = 300, 250, 0.05, 40, 4000
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi, psi, it, converged, viol = sinkslot_symmetric_torch(
        rows, cols, lam, log_a, log_b, n, m, iters)
    assert it == iters and converged is None and viol is None

    T_vals = (phi[rows] + psi[cols] + lam).exp()
    row_marg = torch.zeros(n).index_add_(0, rows, T_vals)
    col_marg = torch.zeros(m).index_add_(0, cols, T_vals)
    assert torch.allclose(row_marg, a, atol=1e-3)
    assert torch.allclose(col_marg, b, atol=1e-3)


def test_sinkslot_symmetric_torch_marginal_mode_converges():
    """Unlike alternating, neither marginal is exact after a symmetric update
    (see sinkslot_symmetric_triton's own docstring), so this checks BOTH row
    and col marginals directly against the achieved plan, not just the
    solver's own self-reported `viol`.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    stop = _Stop(mode="marginal", max_iter=20000)
    phi, psi, it, converged, viol = sinkslot_symmetric_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000, stop=stop)
    assert converged and viol <= stop.tol

    T_vals = (phi[rows] + psi[cols] + lam).exp()
    row_marg = torch.zeros(n).index_add_(0, rows, T_vals)
    col_marg = torch.zeros(m).index_add_(0, cols, T_vals)
    assert float((row_marg - a).abs().max()) <= stop.tol * 10
    assert float((col_marg - b).abs().max()) <= stop.tol * 10


def test_sinkslot_symmetric_torch_potential_mode_converges():
    """stop.mode == "potential" (renamed from "potential_linf", see issue #47)."""
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    stop = _Stop(mode="potential", max_iter=20000, tol=1e-6)
    phi, psi, it, converged, change = sinkslot_symmetric_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000, stop=stop, eps=eps)
    assert converged
    assert change < stop.tol


def test_sinkslot_symmetric_converges_to_the_same_plan_as_alternating():
    """The two schemes solve the same convex problem, so at tight tolerance
    they must agree on the achieved TRANSPORT PLAN -- not on phi/psi
    themselves, which are only defined up to the usual (+c,-c) potential
    -shift ambiguity and can (and do) converge to different additive shifts.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    stop_kwargs = dict(stop_mode="marginal", stop_max_iter=20000, stop_tol=1e-8)

    r_alt = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000, **stop_kwargs,
                            backend="torch", variant="alternating")
    r_sym = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000, **stop_kwargs,
                            backend="torch", variant="symmetric")
    phi_a, psi_a, rows_a, cols_a, S_a = r_alt[:5]
    phi_s, psi_s, rows_s, cols_s, S_s = r_sym[:5]
    assert torch.equal(rows_a, rows_s) and torch.equal(cols_a, cols_s)

    cost = (x[rows_a] - y[cols_a]).square().sum(1)
    lam = S_a.clamp_min(torch.finfo(S_a.dtype).tiny).log() - cost / eps
    T_a = (phi_a[rows_a] + psi_a[cols_a] + lam).exp()
    T_s = (phi_s[rows_s] + psi_s[cols_s] + lam).exp()
    relerr = float((T_a - T_s).norm() / T_a.norm())
    assert relerr < 1e-3, f"alternating and symmetric converged to different plans: {relerr:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sinkslot_solve_all_three_settings_agree_on_cuda():
    """Triton, pure-torch-on-CPU, and pure-torch-on-CUDA
    must all solve the same problem and agree. Triton and torch-on-CPU are
    covered elsewhere; this is the one that needs a GPU to even attempt --
    backend='torch' forced on a CUDA tensor, previously never exercised
    (sinkslot_solve's own auto-dispatch always preferred Triton whenever it
    was available on a CUDA tensor, so the torch path had only ever run on
    CPU tensors before this test).
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    x, y, a, b = x.cuda(), y.cuda(), a.cuda(), b.cuda()
    stop_kwargs = dict(stop_mode="marginal", stop_max_iter=20000)

    phi_triton, psi_triton, *_, conv_t, viol_t = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, **stop_kwargs, backend="triton")
    phi_torch, psi_torch, *_, conv_g, viol_g = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, **stop_kwargs, backend="torch")
    phi_auto, psi_auto, *_ = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, **stop_kwargs, backend="auto")

    assert conv_t and viol_t <= 1e-6
    assert conv_g and viol_g <= 1e-6
    # auto must pick triton when both are available on a CUDA tensor.
    assert torch.equal(phi_auto, phi_triton) and torch.equal(psi_auto, psi_triton)

    dphi = float((phi_triton - phi_torch).abs().max())
    dpsi = float((psi_triton - psi_torch).abs().max())
    assert dphi < 1e-3 and dpsi < 1e-3, f"triton vs torch-on-cuda disagree: {dphi:.2e}, {dpsi:.2e}"


def test_seg_lse_coo_matches_brute_force():
    """The primitive both loops share, against a direct per-row logsumexp."""
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 12
    idx = torch.randint(0, n, (200,), generator=g)
    vals = torch.randn(200, generator=g)

    got = _seg_lse_coo(vals, idx, n)
    want = torch.full((n,), float("-inf"))
    for i in range(n):
        group = vals[idx == i]
        if group.numel() > 0:
            want[i] = torch.logsumexp(group, dim=0)
    assert torch.allclose(got, want, atol=1e-5)


def test_samples_loss_symmetric_option_runs_and_agrees_with_alternating():
    """SamplesLoss(variant="symmetric"): forward value and backward gradient
    must both be finite, and -- given enough iterations for both schemes to
    be near their (shared, see test_sinkslot_symmetric_converges_to_the_same
    _plan_as_alternating above) fixed point -- close to the default
    (alternating) SamplesLoss on the same problem. Not bit-identical: they
    take different iterative paths to the same plan.
    """
    from sinkslot import SamplesLoss

    n, d = 100, 3
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(n, d, generator=g) + 1.0

    x_sym = x.clone().requires_grad_(True)
    loss_sym = SamplesLoss(eps=0.05, L=20, n_iters=3000, backend="torch", variant="symmetric")
    L_sym = loss_sym(x_sym, y)
    L_sym.backward()
    assert torch.isfinite(L_sym) and torch.isfinite(x_sym.grad).all()

    x_alt = x.clone().requires_grad_(True)
    loss_alt = SamplesLoss(eps=0.05, L=20, n_iters=3000, backend="torch", variant="alternating")
    L_alt = loss_alt(x_alt, y)
    L_alt.backward()

    loss_relerr = abs(float(L_sym) - float(L_alt)) / abs(float(L_alt))
    grad_relerr = float((x_sym.grad - x_alt.grad).norm() / x_alt.grad.norm())
    assert loss_relerr < 0.01, f"loss disagrees too much: {loss_relerr:.3e}"
    assert grad_relerr < 0.02, f"grad disagrees too much: {grad_relerr:.3e}"

    # alpha=1.0: unaveraged simultaneous update: must still run and be finite.
    loss_a1 = SamplesLoss(eps=0.05, L=20, n_iters=500, backend="torch",
                           variant="symmetric", alpha=1.0)
    assert torch.isfinite(loss_a1(x, y))

    # potentials=True escape hatch under symmetric too.
    phi, psi = loss_sym(x, y, potentials=True)
    assert phi.shape == (n,) and psi.shape == (n,)


def test_samples_loss_stop_early_stopping_agrees_with_fixed_iters():
    """SamplesLoss(stop_mode=...) forwards to sinkslot_solve unchanged: forward
    value and backward gradient must both be finite, close to a
    fixed-iteration SamplesLoss run long enough to be near the same fixed
    point, and potentials=True must still work with stop set.
    """
    from sinkslot import SamplesLoss

    n, d = 100, 3
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(n, d, generator=g) + 1.0

    x_stop = x.clone().requires_grad_(True)
    loss_stop = SamplesLoss(eps=0.05, L=20, n_iters=200, backend="torch",
                             stop_mode="marginal", stop_max_iter=20000,
                             stop_tol=1e-4, stop_check_every=5)
    L_stop = loss_stop(x_stop, y)
    L_stop.backward()
    assert torch.isfinite(L_stop) and torch.isfinite(x_stop.grad).all()

    x_fixed = x.clone().requires_grad_(True)
    loss_fixed = SamplesLoss(eps=0.05, L=20, n_iters=3000, backend="torch")
    L_fixed = loss_fixed(x_fixed, y)
    L_fixed.backward()

    loss_relerr = abs(float(L_stop) - float(L_fixed)) / abs(float(L_fixed))
    grad_relerr = float((x_stop.grad - x_fixed.grad).norm() / x_fixed.grad.norm())
    assert loss_relerr < 0.01, f"loss disagrees too much: {loss_relerr:.3e}"
    assert grad_relerr < 0.02, f"grad disagrees too much: {grad_relerr:.3e}"

    # potentials=True escape hatch under stop too.
    phi, psi = loss_stop(x, y, potentials=True)
    assert phi.shape == (n,) and psi.shape == (n,)


def test_samples_loss_forward_matches_primal_slot_eps_at_convergence():
    """SamplesLoss returns the dual objective eps*(<phi,a>+<psi,b>), not the
    transport-only <T,C> term -- at a tightly-converged fixed point this must
    equal the primal <T,C> + eps*KL(T, P^SOT) (strong duality), same
    convention flash_sinkhorn/geomloss use for their own Sinkhorn cost.
    """
    from sinkslot import SamplesLoss, sinkslot_solve
    from sinkslot.solver import sparse_sqeuclidean_cost

    n, m, eps, L = 300, 250, 0.03, 40
    x, y, a, b = _problem(n, m)
    x, y, a, b = x.double(), y.double(), a.double(), b.double()

    loss = SamplesLoss(eps=eps, L=L, n_iters=20000, backend="torch",
                        stop_mode="marginal", stop_max_iter=20000,
                        stop_tol=1e-10, stop_check_every=5)
    dual = loss(x, y, a=a, b=b)

    result = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000,
                             backend="torch", stop_mode="marginal",
                             stop_max_iter=20000, stop_tol=1e-10, stop_check_every=5)
    assert result.final_viol < 1e-9, f"not tightly converged: {result.final_viol:.3e}"
    cost = sparse_sqeuclidean_cost(x, y, result.rows, result.cols, use_triton=False)
    log_S = result.S.clamp_min(torch.finfo(result.S.dtype).tiny).log()
    T_vals = (result.phi[result.rows] + result.psi[result.cols] + log_S - cost / eps).exp()
    primal = (T_vals * cost).sum() + eps * (
        T_vals * (T_vals.clamp_min(1e-300).log() - log_S)).sum()

    relerr = abs(float(dual) - float(primal)) / abs(float(primal))
    assert relerr < 1e-6, f"dual vs primal SLOT_eps disagree: {relerr:.3e}"


def test_samples_loss_and_slot_grad_independent_ab_match_frozen_support_finite_diff():
    """SamplesLoss/slot_grad's envelope-theorem gradient (grad_x = 2*a*(x-Tx))
    is unchanged by using independent a/b (and n != m) instead of a shared
    weight vector -- only the plan it's built from differs. Verified against
    a FROZEN-SUPPORT central finite difference (same convention as
    hvp.py/test_hvp.py: freezing rows/cols/phi/psi and letting only the
    cost-dependent lam term respond to x is what isolates the envelope-
    theorem formula from the sliced-OT support's own piecewise-constant
    jumps, which a naive re-solve at x+-h*v would otherwise pick up as noise
    an order of magnitude larger than this test's tolerance).
    """
    from sinkslot import SamplesLoss, slot_grad, sinkslot_solve
    from sinkslot.solver import sparse_sqeuclidean_cost

    n, m, eps, L = 50, 35, 0.05, 60
    g = torch.Generator(device="cpu").manual_seed(1)
    x = (torch.randn(n, 3, generator=g) * 1.0).double()
    y = (torch.randn(m, 3, generator=g) + 1.0).double()
    a = torch.rand(n, generator=g).double(); a = a / a.sum()
    b = torch.rand(m, generator=g).double(); b = b / b.sum()

    phi, psi, rows, cols, S, *_ = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=3000, backend="torch")
    cost = sparse_sqeuclidean_cost(x, y, rows, cols, use_triton=False)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    T_vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()

    def cost_frozen_plan(xp):
        c = (xp[rows] - y[cols]).square().sum(1)
        return (T_vals * c).sum()

    h = 1e-4
    v = torch.randn(n, 3, generator=g).double(); v = v / v.norm()
    fd = (cost_frozen_plan(x + h * v) - cost_frozen_plan(x - h * v)) / (2 * h)

    g_slot = slot_grad(x, y, a, eps, L, seed=0, n_iters=3000, backend="torch", b=b)
    relerr_slot_grad = abs(float(fd) - float((g_slot * v).sum())) / abs(float(fd))
    assert relerr_slot_grad < 1e-6, f"slot_grad(b=...) vs frozen FD: {relerr_slot_grad:.3e}"

    x_req = x.clone().requires_grad_(True)
    loss = SamplesLoss(eps=eps, L=L, n_iters=3000, backend="torch")
    cost_sl = loss(x_req, y, a=a, b=b)
    g_sl, = torch.autograd.grad(cost_sl, [x_req])
    relerr_samples_loss = abs(float(fd) - float((g_sl * v).sum())) / abs(float(fd))
    assert relerr_samples_loss < 1e-6, f"SamplesLoss(a,b) vs frozen FD: {relerr_samples_loss:.3e}"
