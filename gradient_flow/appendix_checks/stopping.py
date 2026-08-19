"""How early can the inner Sinkhorn stop before the gradient goes wrong?

    python -m gradient_flow.appendix_checks.stopping

`sinkslot.gradient.slot_grad` differentiates SLOT_eps by the envelope theorem,

    grad_X SLOT_eps(X, Y) = 2 * diag(a) * (X - T_eps(X)),

which is exact *at* the optimum: it discards the terms that vanish because the
potentials are stationary there. Stop the inner solve early and those terms are
no longer zero, so the gradient carries an error the formula cannot see. The
gradient-flow figure sidesteps the question by running a fixed MAX_ITER=1000
iterations; this script asks what that buys.

Two curves, because they do not say the same thing:

  * ||grad||, which is what you can actually measure at run time, and
  * the relative error against a converged reference gradient, which is what you
    care about and cannot measure without already having the answer.

The point of plotting both is that the norm plateaus well before the error does.
A norm that has stopped moving looks like convergence and is not: the gradient's
magnitude settles while its direction is still turning. Anything that reads the
norm as a stopping signal therefore stops too early, and the marginal violation
-- the solver's actual stopping statistic, plotted alongside -- is the honest
proxy.

Setup is identical to run.py's (same densities, same rng seed, same N, L, eps),
and the measurement is taken at the first gradient step, where X is still the
source blob. fp32 throughout. Runs on CPU or CUDA (see _build_support/_grad_at's
own dispatch between the fused Triton loop and its pure-torch fallback).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sinkslot.solver import (  # noqa: E402
    _HAS_TRITON,
    _ot_1d_coo_batched,
    _ot_1d_coo_batched_cuda,
    expected_sliced_plan,
    sparse_sqeuclidean_cost,
    to_csr,
)
from sinkslot.sinkhorn_solvers import (  # noqa: E402
    sinkslot_alternating_triton,
    sinkslot_alternating_torch,
)
from sinkslot.gradient import plan_barycentric_sparse  # noqa: E402
from gradient_flow.config import DATA_SCALE, L, N  # noqa: E402
from gradient_flow.run import DATA_DIR, DEVICE, OUT_DIR, draw_samples  # noqa: E402

# Geometric-ish ladder: the interesting behaviour is all in the first ~200
# iterations, and a linear grid wastes most of its points on the flat tail.
ITER_GRID = [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
REFERENCE_ITERS = 5000  # >> the largest grid point; asserted converged below
EPS = 0.01
SEED = 0


def _build_support(X, Y, a, eps, slices, seed):
    """The setup half of slot_grad: sliced support, sparse cost, CSR + CSC.

    Hoisted out so the iteration sweep pays for it once. The plan depends only
    on (X, Y, a, L, seed), none of which vary across the sweep, so every
    iteration count below solves the *same* problem -- otherwise the curves
    would confound solver progress with a re-randomised support.

    CSR/CSC are only needed for the fused Triton loop (sinkslot_alternating_triton); on the
    pure-torch path (sinkslot_alternating_torch) they're skipped entirely -- it operates
    on the COO (rows, cols, lam) directly, so there's nothing to hoist there.
    """
    n, m = X.shape[0], Y.shape[0]
    ot1d = _ot_1d_coo_batched_cuda if X.is_cuda else _ot_1d_coo_batched
    rows, cols, S = expected_sliced_plan(X, Y, a, a, L=slices, seed=seed, ot1d=ot1d)
    cost = sparse_sqeuclidean_cost(X, Y, rows, cols)
    lam = S.clamp_min(torch.finfo(S.dtype).tiny).log() - cost / eps
    if _HAS_TRITON and X.is_cuda:
        csr = to_csr(rows, cols, lam, n, narrow_key=True)
        csc = to_csr(cols, rows, lam, m, narrow_key=True)
    else:
        csr = csc = None
    return rows, cols, lam, csr, csc


def _grad_at(n_iters, X, Y, a, rows, cols, lam, csr, csc):
    """(gradient, max marginal violation) after exactly `n_iters` inner iterations.

    `sinkslot_alternating_triton`/`sinkslot_alternating_torch` both initialise the potentials to zero, so
    running with n_iters=k reproduces the state the solver would be in after
    k iterations -- no need to checkpoint a single long run. Dispatches on
    whether `_build_support` built CSR/CSC (Triton available and X on CUDA)
    or not (pure-torch path).
    """
    n, m = X.shape[0], Y.shape[0]
    log_a = a.log()
    if csr is not None:
        r_ptr, r_idx, r_lam, _ = csr
        c_ptr, c_idx, c_lam, _ = csc
        phi, psi, _, _, _ = sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                    log_a, log_a, n, m, n_iters)
    else:
        phi, psi, _, _, _ = sinkslot_alternating_torch(rows, cols, lam, log_a, log_a, n, m, n_iters)

    T_vals = (phi[rows] + psi[cols] + lam).exp()
    # The solver's own stopping statistic: after the column half-step the column
    # marginal is exact by construction, so only the row marginal deviates.
    row_marg = torch.zeros(n, device=X.device, dtype=T_vals.dtype).index_add_(0, rows, T_vals)
    viol = float((row_marg - a).abs().max())

    Tx, _ = plan_barycentric_sparse(T_vals, rows, cols, X, Y)
    return 2.0 * a[:, None] * (X - Tx), viol


def main():
    if DEVICE != "cuda":
        print("gradient_flow/stopping.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    # Same rng draw order as run.py's main(), so X0/Y are the identical clouds.
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", N, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", N, rng, device=DEVICE).float()
    a = torch.full((N,), 1.0 / N, dtype=torch.float32, device=DEVICE)

    rows, cols, lam, csr, csc = _build_support(X, Y, a, EPS, L, SEED)
    print(f"support: nnz={rows.numel()}  N={N}  L={L}  eps={EPS}")

    g_ref, viol_ref = _grad_at(REFERENCE_ITERS, X, Y, a, rows, cols, lam, csr, csc)
    ref_norm = float(g_ref.norm())
    print(f"reference @ {REFERENCE_ITERS} iters: ||grad||={ref_norm:.6e}  viol={viol_ref:.3e}")
    assert viol_ref < 1e-6, (
        f"reference gradient is not converged (marginal violation {viol_ref:.3e}); "
        "raise REFERENCE_ITERS before trusting the error curve"
    )

    norms, rel_errs, cosines, viols = [], [], [], []
    for k in ITER_GRID:
        g, viol = _grad_at(k, X, Y, a, rows, cols, lam, csr, csc)
        rel = float((g - g_ref).norm()) / ref_norm
        cos = float(torch.nn.functional.cosine_similarity(
            g.flatten(), g_ref.flatten(), dim=0))
        norms.append(float(g.norm()))
        rel_errs.append(rel)
        cosines.append(cos)
        viols.append(viol)
        print(f"  iters={k:>5}  ||grad||={norms[-1]:.6e}  rel.err={rel:.3e}  "
              f"cos={cos:.6f}  viol={viol:.3e}")

    _plot(norms, rel_errs, cosines, viols, ref_norm)
    _report(norms, rel_errs, viols, ref_norm)


def _plot(norms, rel_errs, cosines, viols, ref_norm):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_l.axhline(ref_norm, ls="--", lw=1, color="0.6", label="converged $\\|\\nabla\\|$")
    ax_l.plot(ITER_GRID, norms, "o-", ms=4, color="#1f77b4")
    ax_l.set_xscale("log")
    ax_l.set_xlabel("inner Sinkhorn iterations")
    ax_l.set_ylabel("$\\|\\nabla_X \\mathrm{SLOT}_\\epsilon\\|$")
    ax_l.set_title("Gradient norm")
    ax_l.legend(frameon=False, fontsize=9)
    ax_l.grid(alpha=0.3)

    ax_r.plot(ITER_GRID, rel_errs, "o-", ms=4, color="#d62728",
              label="relative error vs converged")
    ax_r.plot(ITER_GRID, viols, "s--", ms=4, color="#2ca02c",
              label="max marginal violation")
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("inner Sinkhorn iterations")
    ax_r.set_ylabel("relative error  /  violation")
    ax_r.set_title("Gradient error and the solver's stopping statistic")
    ax_r.legend(frameon=False, fontsize=9)
    ax_r.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"SinkSLOT envelope-theorem gradient under early stopping "
        f"($N={N}$, $L={L}$, $\\epsilon={EPS}$)", y=1.02)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"gradient_stopping_eps_{EPS:g}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def _report(norms, rel_errs, viols, ref_norm):
    """The quantitative claims the figure is meant to support."""
    norm_errs = [abs(v - ref_norm) / ref_norm for v in norms]

    def first_below(series, thresh):
        for k, v in zip(ITER_GRID, series):
            if v < thresh:
                return k
        return None

    print("\n--- summary ---")
    for thresh in (1e-2, 1e-3):
        kn, ke = first_below(norm_errs, thresh), first_below(rel_errs, thresh)
        print(f"  within {thresh * 100:g}%: norm settles at {kn} iters, gradient at {ke}")

    # The sharper statement. At a fixed iteration count, how far does the norm's
    # own deviation understate the error in the gradient it belongs to? This is
    # the quantity that makes "watch the norm" an unsafe stopping rule -- not the
    # iteration gap, which is small, but the confidence gap at any given stop.
    # Restricted to the regime where both quantities are well above the fp32
    # noise floor; past that the ratio is a division of two rounding errors.
    worst, worst_k = 0.0, None
    for k, ne, re in zip(ITER_GRID, norm_errs, rel_errs):
        if ne > 1e-6 and re > 1e-5 and re / ne > worst:
            worst, worst_k = re / ne, k
    print(f"  norm understates the gradient error by up to {worst:.0f}x (at {worst_k} iters)")

    # The solver stops on marginal violation, so the useful calibration is the
    # constant relating the two -- it converts a tolerance into a gradient error.
    ratios = [re / v for re, v in zip(rel_errs, viols) if v > 1e-7]
    if ratios:
        # quantile(0.5), not .median(), to match numpy's median() (averages the
        # middle two for an even-length input; .median() would just pick one).
        med = float(torch.quantile(torch.tensor(ratios), 0.5))
        print(f"  relative gradient error ~= {med:.0f} x marginal violation")
        print(f"     -> for 1% gradient error, stop at viol ~= {1e-2 / med:.1e}")


if __name__ == "__main__":
    main()
