"""Tests for the pure-torch fallback (`sinkslot_alternating_torch`, `sinkslot_solve`), #10.

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


def test_run_v5_torch_fixed_mode_gives_correct_marginals():
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


def test_run_v5_torch_marginal_and_potential_modes_converge_and_agree():
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi_m, psi_m, it_m, conv_m, viol_m = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000, _Stop(mode="marginal"))
    assert conv_m and viol_m <= 1e-6

    phi_p, psi_p, it_p, conv_p, viol_p = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000, _Stop(mode="potential"))
    assert conv_p
    # Documented to fall back to the same check as marginal mode.
    assert torch.equal(phi_m, phi_p) and torch.equal(psi_m, psi_p)


def test_run_v5_torch_potential_linf_mode_converges():
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    rows, cols, S = sot_plan_coo(x, y, a, b, L=L, seed=0)
    cost = sparse_sqeuclidean_cost(x, y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    log_a, log_b = a.log(), b.log()

    phi, psi, it, converged, change = sinkslot_alternating_torch(
        rows, cols, lam, log_a, log_b, n, m, 20000,
        _Stop(mode="potential_linf", tol=1e-4), eps=eps)
    assert converged and change < 1e-4


def test_run_v5_torch_rejects_unknown_mode():
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
    """The actual #10 ask: one call, no CUDA, no Triton, correct answer."""
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    assert not x.is_cuda

    phi, psi, rows, cols, S, it, converged, viol = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000,
        stop=_Stop(mode="marginal", max_iter=20000))
    assert converged and viol <= 1e-6

    P = (phi[rows] + psi[cols] +
         (S.clamp_min(torch.finfo(S.dtype).tiny).log() -
          sparse_sqeuclidean_cost(x, y, rows, cols) / eps)).exp()
    row_marg = torch.zeros(n).index_add_(0, rows, P)
    col_marg = torch.zeros(m).index_add_(0, cols, P)
    assert torch.allclose(row_marg, a, atol=1e-5)
    assert torch.allclose(col_marg, b, atol=1e-5)


def test_sinkslot_solve_backend_override_on_cpu():
    """backend='torch' and backend='auto' must agree on CPU (both pick torch),
    backend='triton' must raise cleanly (no CUDA here), and a bogus backend
    string must raise too -- the API surface this issue actually asks for.
    """
    n, m, eps, L = 300, 250, 0.05, 40
    x, y, a, b = _problem(n, m)
    stop = _Stop(mode="marginal", max_iter=20000)

    auto = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000, stop=stop, backend="auto")
    torch_backend = sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=20000, stop=stop, backend="torch")
    assert torch.equal(auto[0], torch_backend[0]) and torch.equal(auto[1], torch_backend[1])

    with pytest.raises(ValueError, match="backend='triton'"):
        sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=200, backend="triton")
    with pytest.raises(ValueError, match="backend must be"):
        sinkslot_solve(x, y, a, b, eps, L, seed=0, n_iters=200, backend="bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sinkslot_solve_all_three_settings_agree_on_cuda():
    """The actual #22 ask: Triton, pure-torch-on-CPU, and pure-torch-on-CUDA
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
    stop = _Stop(mode="marginal", max_iter=20000)

    phi_triton, psi_triton, *_, conv_t, viol_t = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, stop=stop, backend="triton")
    phi_torch, psi_torch, *_, conv_g, viol_g = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, stop=stop, backend="torch")
    phi_auto, psi_auto, *_ = sinkslot_solve(
        x, y, a, b, eps, L, seed=0, n_iters=20000, stop=stop, backend="auto")

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
