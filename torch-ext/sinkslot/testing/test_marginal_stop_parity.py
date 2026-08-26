"""`stop_mode="marginal"` means the same thing in every implementation.

SinkSLOT, SROT, Spar-Sink and FlashSinkhorn each carry their own Sinkhorn
loop, and the paper's headline comparisons run all of them under
`stop_mode="marginal"` at a shared tolerance. That is only a fair comparison
if "marginal" denotes one rule everywhere:

    viol = max(max|P1 - a|, max|P^T 1 - b|) <= tol

Two things can silently break it, and both have happened before:

* a solver measuring total variation (sum) instead of max. A TV sum over n
  terms against a fixed absolute tolerance is unreachable at n=10,000
  however converged the solve is, so a TV solver either never stops or, at
  small n, stops somewhere entirely different from the others.
* a solver checking only one marginal. Gauss-Seidel satisfies one marginal
  exactly by construction after each half-update, so a one-sided check
  reports convergence that has not happened.

Neither shows up as a crash; both show up as one method appearing faster
than the rest because it stopped on an easier rule. Hence this file.

The check is deliberately not "did the solver's own counter say converged".
Each solver reports a *proxy* violation computed from consecutive potentials
(phi_old - phi) rather than from the plan, so trusting it would only test
the proxy against itself. Instead every test here reconstructs the transport
plan from the returned potentials, sums its marginals in float64, and holds
that independent number against the tolerance the caller asked for.
"""

import pytest
import torch

from sinkslot.bench.bench_forward import (
    StopCfg,
    _sparsink_sinkhorn,
    _srot_sinkhorn,
    build_sot_plan,
    build_sparse_kernel,
)
from sinkslot.sinkhorn_solvers import sinkslot_solve

TOL = 1e-4
MAX_ITER = 2000
CHECK_EVERY = 10

# The reconstructed-plan violation is computed in float64 from float32
# potentials, so it will not land exactly on the solver's own float32 proxy.
# The slack absorbs that, and nothing more: at 2x, a solver that switched to
# a one-sided or TV rule would still fail, because those miss by orders of
# magnitude rather than by a rounding step.
SLACK = 2.0


def _problem(n=384, m=320, d=4, seed=0, device="cpu"):
    """Non-uniform a and b on purpose: under uniform marginals Spar-Sink's
    sqrt(a_i b_j) importance sampling degenerates to the uniform rand_sink
    scheme, and a row/column asymmetry bug would have nothing to bite on."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(m, d, generator=g) + 1.5
    a = torch.rand(n, generator=g) + 0.5
    b = torch.rand(m, generator=g) + 0.5
    a, b = a / a.sum(), b / b.sum()
    return (t.to(device) for t in (x, y, a, b))


def _viol(row_marg, col_marg, a, b):
    """The rule itself, in float64, from actual plan marginals."""
    return float(
        torch.maximum(
            (row_marg - a.double()).abs().max(),
            (col_marg - b.double()).abs().max(),
        )
    )


def _tv(row_marg, col_marg, a, b):
    return float((row_marg - a.double()).abs().sum() + (col_marg - b.double()).abs().sum())


def _sqeuclid(x, y):
    return torch.cdist(x, y) ** 2


def _stop():
    return StopCfg(mode="marginal", max_iter=MAX_ITER, tol=TOL, check_every=CHECK_EVERY)


# ---------------------------------------------------------------------------
# Per-implementation: converge, then verify against the reconstructed plan
# ---------------------------------------------------------------------------


def _run_srot(device="cpu", eps=0.05):
    x, y, a, b = _problem(device=device)
    cost = _sqeuclid(x, y)
    pi = build_sot_plan(x, y, a, b, slices=64, seed=0)
    log_pi = pi.clamp_min(torch.finfo(pi.dtype).tiny).log()
    f, g, iters, converged, proxy = _srot_sinkhorn(
        cost, log_pi, a.log(), b.log(), eps, MAX_ITER, _stop()
    )
    # P = pi_SOT * exp((f (+) g - C)/eps), the fixed point SROT solves for.
    P = (log_pi + (f[:, None] + g[None, :] - cost) / eps).double().exp()
    return P.sum(1), P.sum(0), a, b, iters, converged, proxy


def _run_sparsink(method="spar_sink", device="cpu", eps=0.05):
    x, y, a, b = _problem(device=device)
    cost = _sqeuclid(x, y)
    # Large enough that every row and column keeps at least one sampled entry:
    # an empty row cannot carry its mass anywhere, so its marginal violation is
    # bounded below by a_i no matter how converged the solve is, and the run is
    # reported N/A rather than compared.
    rows, cols, log_values = build_sparse_kernel(
        cost, a, b, eps, method=method, sample_size=200_000, seed=0
    )
    f, g, empty, iters, converged, proxy = _sparsink_sinkhorn(
        rows, cols, log_values, a.log(), b.log(), eps, MAX_ITER, _stop()
    )
    assert empty == 0, f"{method} sampled an empty row/col; raise sample_size"
    vals = (log_values + (f[rows] + g[cols]) / eps).double().exp()
    row_marg = torch.zeros(a.shape[0], dtype=torch.float64).index_add_(0, rows, vals)
    col_marg = torch.zeros(b.shape[0], dtype=torch.float64).index_add_(0, cols, vals)
    return row_marg, col_marg, a, b, iters, converged, proxy


def _run_sinkslot(device="cpu", eps=0.05, variant="alternating"):
    x, y, a, b = _problem(device=device)
    res = sinkslot_solve(
        x, y, a, b, eps=eps, L=64, seed=0, n_iters=MAX_ITER,
        stop_mode="marginal", stop_max_iter=MAX_ITER, stop_tol=TOL,
        stop_check_every=CHECK_EVERY, variant=variant,
    )
    cost = _sqeuclid(x, y)[res.rows, res.cols]
    log_S = res.S.clamp_min(torch.finfo(res.S.dtype).tiny).log()
    vals = (res.phi[res.rows] + res.psi[res.cols] + log_S - cost / eps).double().exp()
    row_marg = torch.zeros(a.shape[0], dtype=torch.float64).index_add_(0, res.rows, vals)
    col_marg = torch.zeros(b.shape[0], dtype=torch.float64).index_add_(0, res.cols, vals)
    return row_marg, col_marg, a, b, res.iters_run, res.converged, res.final_viol


_RUNNERS = {
    "sinkslot": _run_sinkslot,
    "srot": _run_srot,
    "spar_sink": lambda: _run_sparsink("spar_sink"),
    "rand_sink": lambda: _run_sparsink("rand_sink"),
}


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_marginal_mode_stops_at_the_tolerance_it_was_given(name):
    """Converges, and the plan it converged to really is within tol."""
    row_marg, col_marg, a, b, iters, converged, _proxy = _RUNNERS[name]()

    assert converged, f"{name} hit max_iter={MAX_ITER} without converging"
    assert iters < MAX_ITER

    viol = _viol(row_marg, col_marg, a, b)
    assert viol <= TOL * SLACK, f"{name}: true max marginal violation {viol:.3e} > tol {TOL:.0e}"


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_reported_violation_tracks_the_real_one(name):
    """The phi_old proxy is an approximation; this is what bounds the error.

    Each solver reports a violation derived from consecutive potentials rather
    than from the plan. That proxy is what decides when to stop, so if it ever
    drifts away from the quantity it stands for, every method stops on a
    different rule while all of them still claim "marginal".
    """
    row_marg, col_marg, a, b, _iters, _converged, proxy = _RUNNERS[name]()
    viol = _viol(row_marg, col_marg, a, b)
    assert proxy == pytest.approx(viol, rel=0.25, abs=TOL * 0.5), (
        f"{name}: reported {proxy:.3e} but plan says {viol:.3e}"
    )


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_marginal_rule_is_max_not_total_variation(name):
    """Guards the specific regression that made marginal mode look broken.

    At convergence the TV sum is far above the same tolerance -- roughly n
    times the max, since it adds one term per row. Asserting that keeps anyone
    from "fixing" a solver back onto a TV rule: under TV these runs would not
    have stopped at all.
    """
    row_marg, col_marg, a, b, _iters, _converged, _proxy = _RUNNERS[name]()
    assert _tv(row_marg, col_marg, a, b) > TOL, (
        f"{name}: TV violation is below tol, so this test can no longer "
        f"distinguish a max rule from a TV one -- lower TOL or raise n"
    )


def test_every_implementation_agrees_on_the_same_problem():
    """All of them, one tolerance, one rule -- the comparison the paper makes."""
    results = {name: _RUNNERS[name]() for name in sorted(_RUNNERS)}
    viols = {n: _viol(r[0], r[1], r[2], r[3]) for n, r in results.items()}

    assert all(r[5] for r in results.values()), (
        f"not all converged: { {n: r[5] for n, r in results.items()} }"
    )
    for name, viol in viols.items():
        assert viol <= TOL * SLACK, f"{name} at {viol:.3e}, tol {TOL:.0e}, all={viols}"


# ---------------------------------------------------------------------------
# FlashSinkhorn: same rule, but it needs the bench extra and a GPU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["alternating", "symmetric"])
def test_flash_sinkhorn_marginal_mode_matches(backend):
    """The fork's stop_mode="marginal" must mean what everyone else's does.

    Skipped without CUDA/Triton: FlashSinkhorn is Triton-only with no
    pure-torch fallback, so this leg cannot run in the CPU CI matrix. It is
    the one implementation living outside this repo, which is exactly why it
    is worth pinning down here rather than trusting the dependency.
    """
    pytest.importorskip("triton")
    flash_sinkhorn = pytest.importorskip("flash_sinkhorn")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    eps = 0.05
    x, y, a, b = _problem(device="cuda")
    # debias/normalize/last_extrapolation off, matching how bench_forward
    # drives it: with debiasing on, SamplesLoss solves the Sinkhorn-divergence
    # problem instead, and (f, g) are then not the potentials of the plan whose
    # marginals this test is about to check.
    loss_fn = flash_sinkhorn.SamplesLoss(
        "sinkhorn", backend=backend, use_epsilon_scaling=False, eps=eps,
        n_iters=MAX_ITER, debias=False, potentials=True, normalize=False,
        last_extrapolation=False, return_n_iters=True,
        threshold=TOL, inner_iterations=CHECK_EVERY, stop_mode="marginal",
        # allow_tf32=False is load-bearing, not tidiness. TF32 carries ~1e-3
        # relative precision, so with it on (the SamplesLoss default) the
        # potentials cannot represent a 1e-4 marginal violation at all: the run
        # still reports converged at the same iteration, but the plan it
        # returns is off by ~2.4e-3. See
        # test_flash_marginal_mode_cannot_beat_the_tf32_noise_floor, which pins
        # that behaviour down deliberately. bench_forward drives every
        # published comparison with --no-tf32 for the same reason.
        allow_tf32=False,
    )
    f, g, iters = loss_fn(a, x, b, y)

    assert iters < MAX_ITER, f"flash_{backend} never converged"

    cost = _sqeuclid(x, y)
    P = ((f[:, None] + g[None, :] - cost) / eps).double().exp()
    P = P * a.double()[:, None] * b.double()[None, :]
    viol = _viol(P.sum(1), P.sum(0), a, b)
    assert viol <= TOL * SLACK, (
        f"flash_{backend}: true max marginal violation {viol:.3e} > tol {TOL:.0e}"
    )


def test_flash_marginal_mode_cannot_beat_the_tf32_noise_floor():
    """TF32 silently caps how tight `stop_mode="marginal"` can actually get.

    Not a test of desired behaviour -- a record of a real trap. TF32 matmuls
    carry roughly 1e-3 relative precision, which is coarser than the 1e-4
    tolerance asked for here. The solver still reports convergence, at the very
    same iteration it would have without TF32, because the proxy it checks is
    computed in the same arithmetic that is losing the precision. The plan that
    comes back misses the requested tolerance by well over an order of
    magnitude.

    Anything comparing methods on a shared marginal tolerance must therefore
    disable TF32, which is what bench_forward's --no-tf32 does for every
    published run. If a future FlashSinkhorn learns to reject or warn about
    this combination, this test will fail -- and that would be an improvement,
    so update it rather than restoring TF32.
    """
    pytest.importorskip("triton")
    flash_sinkhorn = pytest.importorskip("flash_sinkhorn")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    eps = 0.05
    x, y, a, b = _problem(device="cuda")
    cost = _sqeuclid(x, y)

    def solve(allow_tf32):
        fn = flash_sinkhorn.SamplesLoss(
            "sinkhorn", backend="alternating", use_epsilon_scaling=False, eps=eps,
            n_iters=MAX_ITER, debias=False, potentials=True, normalize=False,
            last_extrapolation=False, return_n_iters=True, threshold=TOL,
            inner_iterations=CHECK_EVERY, stop_mode="marginal",
            allow_tf32=allow_tf32,
        )
        f, g, iters = fn(a, x, b, y)
        P = ((f[:, None] + g[None, :] - cost) / eps).double().exp()
        P = P * a.double()[:, None] * b.double()[None, :]
        return iters, _viol(P.sum(1), P.sum(0), a, b)

    iters_off, viol_off = solve(False)
    iters_on, viol_on = solve(True)

    assert viol_off <= TOL * SLACK, f"fp32 leg should hold tol, got {viol_off:.3e}"
    assert viol_on > TOL * SLACK, (
        f"TF32 run met the tolerance ({viol_on:.3e}); if FlashSinkhorn fixed "
        f"this, delete this test and drop allow_tf32=False above"
    )
    # The trap is that the run looks normal: it stops after essentially the
    # same amount of work (within a check interval or two), so nothing about
    # the iteration count hints that the result is an order of magnitude worse.
    assert abs(iters_on - iters_off) <= 3 * CHECK_EVERY, (
        f"TF32 changed the iteration count materially ({iters_on} vs "
        f"{iters_off}); the failure would then be visible rather than silent, "
        f"so this test's premise needs revisiting"
    )
    assert viol_on > 10 * viol_off, (
        f"TF32 leg only degraded from {viol_off:.3e} to {viol_on:.3e}"
    )
