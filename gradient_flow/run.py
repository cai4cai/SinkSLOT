"""Gradient flow, blob -> crescent: SOT / EOT / SROT / SinkSLOT, one compiled figure.

    python -m gradient_flow.run

Runs on CPU or CUDA, in float64 throughout.

Four methods, one figure:
  * SOT (Bonneel et al., 2015): the plain, unregularized sliced W2 distance --
    no Sinkhorn solve at all, each projection's 1-D transport is exact (a
    sort), so autograd through `ot.sliced_wasserstein_distance` already gives
    the analytical gradient.
  * EOT (Feydy et al., 2019): `eot_divergence`, POT's own solve_sample at a
    fixed iteration count (POT's default stopping rule converges well before
    MAX_ITER for typical eps, not comparable to the other three arms' fixed
    budget), with envelope-theorem gradients (see that function's own
    docstring).
  * SROT (Nguyen 2026): `srot_divergence`, the debiased SROT divergence with the dense
    reference (1 - gamma) * P^SOT + gamma * a b^T (gamma = 1e-8) on every pair, solved in
    the log domain with no floor on the kernel, with envelope-theorem gradients.
  * SinkSLOT (ours): the native solver (torch-ext/sinkslot/sinkhorn_solvers.py),
    the exact pipeline this repo's own speed benchmarks exercise,
    plus the closed-form envelope-theorem gradient
    grad_X SLOT_eps(X,Y) = 2*diag(a)*(X - T_eps(X)) (see sinkslot/gradient.py).

    All three regularized arms (EOT, SROT, SinkSLOT) now use envelope-theorem
    gradients -- the converged plan is held fixed and only the explicit cost
    term is differentiated, never the Sinkhorn iteration itself -- verified
    against full backprop-through-every-iteration at X0 in each case.

    All four arms run in float64 (DTYPE below): POT for SOT/EOT, dense torch for
    SROT, and slot_grad's torch backend for SinkSLOT (its Triton kernels are
    float32-only).

Compiled figure formatting: compact vertically with no dead space between
adjacent method rows; for every intermediate checkpoint (step > 0), the
lowest -- i.e. best -- W2^2 across the 4 methods is bolded.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import ot
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gradient_flow.config import (
    N, STEPS, N_STEPS, LR, EPS_VALUES, MAX_ITER, L, DELTA_SROT, DATA_SCALE,
    METHOD_NAMES, ROW_LABELS,
)
from sinkslot.gradient import slot_grad
from sinkslot.solver import sot_plan_coo


def eot_divergence(X_t, Y_t, a_t, b_t, eps, n_iters):
    """Debiased Sinkhorn divergence via POT's own solve_sample, at a fixed
    iteration count, with envelope-theorem gradients.

    grad="envelope" holds the converged plan and dual potentials fixed and
    differentiates only the explicit cost term each is affine in, rather than
    backpropagating through every Sinkhorn iteration -- the same
    envelope-theorem trick sinkslot.gradient.slot_grad uses for SinkSLOT
    itself, here via POT's own public API. Verified to match a hand-rolled
    envelope implementation's gradient at X0 to float64 precision
    (grad_norm 0.06424097343284979 vs 0.06424097343284978).

    tol=-1 makes solve_sample's internal `err < stopThr` stopping check never
    trip (err is never negative), so this always runs exactly n_iters
    iterations like the other three arms, instead of POT's own default
    early-stopping behaviour -- POT's default schedule would otherwise
    converge in far fewer than MAX_ITER steps for typical eps, the same
    fixed-vs-adaptive-iteration mismatch GeomLoss's own SamplesLoss has.
    debias=True reproduces this repo's own ot_xy - 0.5*ot_xx - 0.5*ot_yy
    convention directly (computing all three terms internally, including
    ot_yy even though d(ot_yy)/dX = 0 -- a small, unavoidable overhead of
    using this generic public entry point rather than hand-rolling the
    three terms).

    POT's own sqeuclidean cost is ||x-y||^2, this repo's and sinkslot's own
    convention, so unlike GeomLoss's 0.5*||x-y||^2 no eps/output correction
    is needed here.

    tol=-1 also means POT's internal sinkhorn_log always exhausts its "did
    not convergence" check (by construction, since it never breaks out early)
    and warns accordingly on every call; that warning is expected here (this
    experiment deliberately always runs the full fixed budget) and is
    suppressed rather than left to fire 3 times (xy/xx/yy) per gradient step.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sinkhorn did not converge")
        res = ot.solve_sample(
            X_t, Y_t, a_t, b_t, reg=eps, max_iter=n_iters, tol=-1.0,
            grad="envelope", debias=True,
        )
    return res.value


def srot_divergence(X, Y, a, b, eps, L, seed, n_iters, gamma):
    """SROT divergence (Nguyen 2026, Def. 2) up to its X-independent Y-Y term: OT(X, Y) - OT(X, X) / 2.

    Each OT term uses the dense reference (1 - gamma) * P^SOT + gamma * a b^T on every pair, with P^SOT
    from the same slices as SinkSLOT (sot_plan_coo, same seed), and n_iters log-domain Sinkhorn
    iterations with no floor on the kernel. Envelope-theorem gradient: each plan is solved without
    gradient tracking and only <C, P> is differentiated.
    """
    def term(Xp, Yp, ap, bp):
        with torch.no_grad():
            rows, cols, S = sot_plan_coo(Xp, Yp, ap, bp, L=L, seed=seed)
            ref = torch.zeros(len(ap), len(bp), dtype=Xp.dtype, device=Xp.device)
            ref.index_put_((rows, cols), S, accumulate=True)
            ref = (1.0 - gamma) * ref + gamma * torch.outer(ap, bp)
            log_k = ref.log() - torch.cdist(Xp, Yp) ** 2 / eps
            log_a, log_b = ap.log(), bp.log()
            f, g = torch.zeros_like(ap), torch.zeros_like(bp)
            for _ in range(n_iters):
                f = log_a - torch.logsumexp(log_k + g[None, :], dim=1)
                g = log_b - torch.logsumexp(log_k + f[:, None], dim=0)
            plan = (log_k + f[:, None] + g[None, :]).exp()
        cost = ((Xp[:, None, :] - Yp[None, :, :]) ** 2).sum(-1)  # differentiable; finite gradient at x_i = y_j
        return (plan * cost).sum()

    return term(X, Y, a, b) - 0.5 * term(X, X, a, a)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_density(fname):
    """Grayscale density in [0,1]; row-flipped and inverted, matching Feydy et al. (2019).

    Reads PIL's raw bytes straight into a torch tensor (torch.frombuffer), no numpy.
    """
    img = Image.open(fname).convert("L")
    w, h = img.size
    buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(h, w)
    arr = buf.double() / 255.0
    arr = arr.flip(0)
    return 1.0 - arr


def draw_samples(fname, n, rng, dtype=DTYPE, device="cpu"):
    """`rng` is a torch.Generator (CPU); sampling happens on CPU, moved to `device` last."""
    A = load_density(fname)
    h, w = A.shape
    xg, yg = torch.meshgrid(torch.linspace(0, 1, h, dtype=torch.float64),
                             torch.linspace(0, 1, w, dtype=torch.float64), indexing="xy")
    grid = torch.stack([xg.reshape(-1), yg.reshape(-1)], dim=1)
    dens = A.reshape(-1)
    dens = dens / dens.sum()
    idx = torch.multinomial(dens, n, replacement=True, generator=rng)
    dots = grid[idx].clone()
    dots += (0.5 / h) * torch.randn(dots.shape, generator=rng, dtype=dots.dtype)
    dots *= DATA_SCALE
    return dots.to(dtype=dtype, device=device)


def exact_ot_cost(X, Y):
    """Raw squared-W2 exact OT cost (not sqrt'd). X, Y: torch tensors, any device.

    Calls POT directly (ot.dist, ot.emd2) -- POT's own backend accepts torch
    tensors natively (verified against 0.9.7), so no numpy round-trip is
    needed here at all.
    """
    n, m = X.shape[0], Y.shape[0]
    Xc, Yc = X.detach().cpu().double(), Y.detach().cpu().double()
    a = torch.full((n,), 1.0 / n, dtype=torch.float64)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64)
    return float(ot.emd2(a, b, ot.dist(Xc, Yc, metric="sqeuclidean")))


def run_flow(method, X0, Y, a_t, eps):
    x_i = X0.clone()
    rows = {}
    for step in range(N_STEPS + 1):
        if method == "SinkSLOT":
            x_i_d = x_i.detach()
            g = slot_grad(x_i_d, Y, a_t, a_t, eps, L, seed=0, n_iters=MAX_ITER, backend="torch")
        else:
            x_i.requires_grad_(True)
            if method == "SOT":
                loss = ot.sliced_wasserstein_distance(x_i, Y, n_projections=L, p=2, seed=0)
            elif method == "EOT":
                loss = eot_divergence(x_i, Y, a_t, a_t, eps, n_iters=MAX_ITER)
            else:  # SROT
                loss = srot_divergence(
                    x_i, Y, a_t, a_t, eps, L, seed=0, n_iters=MAX_ITER, gamma=DELTA_SROT)
            (g,) = torch.autograd.grad(loss, [x_i])
            x_i_d = x_i.detach()

        if step in STEPS:
            Xn = x_i_d.cpu().clone()
            w2 = exact_ot_cost(Xn, Y.cpu())
            rows[step] = (Xn, w2)
        if step == N_STEPS:
            break
        x_i = (x_i_d - LR * N * g).clone()
    return rows


def main():
    rng = torch.Generator(device="cpu").manual_seed(1)
    X0 = draw_samples(DATA_DIR / "density_a.png", N, rng, device=DEVICE)
    Y = draw_samples(DATA_DIR / "density_b.png", N, rng, device=DEVICE)
    a_t = torch.full((N,), 1.0 / N, dtype=DTYPE, device=DEVICE)
    colors = (10 * X0[:, 0]).cos() * (10 * X0[:, 1]).cos()
    colors = colors.detach().cpu()

    X0n, Yn = X0.cpu(), Y.cpu()
    pts = torch.cat([X0n, Yn], dim=0)
    lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
    pad = (hi - lo) * 0.06
    xlim = (float(lo[0] - pad[0]), float(hi[0] + pad[0]))
    ylim = (float(lo[1] - pad[1]), float(hi[1] + pad[1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for eps in EPS_VALUES:
        results = {}
        for method in METHOD_NAMES:
            t0 = time.perf_counter()
            rows = run_flow(method, X0, Y, a_t, eps)
            for step, (Xn, w2) in rows.items():
                results[(method, step)] = (Xn, w2)
                print(f"[{method} eps={eps:g}] step {step:>3}/{N_STEPS} "
                      f"W2={w2:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        # For every intermediate checkpoint, find the best (lowest) W2 across
        # methods so its label can be bolded. Step 0 (shared start, identical
        # for all methods) and step 50 (final step) are excluded.
        best_method_at = {}
        for step in STEPS:
            if step in (0, N_STEPS):
                continue
            vals = {m: results[(m, step)][1] for m in METHOD_NAMES if (m, step) in results}
            if vals:
                best_method_at[step] = min(vals, key=vals.get)

        nr, nc = len(METHOD_NAMES), len(STEPS)
        data_w, data_h = xlim[1] - xlim[0], ylim[1] - ylim[0]
        panel_w = 1.9
        panel_h = panel_w * data_h / data_w
        label_h, title_h = 0.12, 0.24
        fig, axes = plt.subplots(
            nr, nc, squeeze=False,
            figsize=(panel_w * nc, (panel_h + label_h) * nr + title_h),
            gridspec_kw=dict(wspace=0.015, hspace=0.0,
                              top=1 - title_h / ((panel_h + label_h) * nr + title_h),
                              bottom=0.005, left=0.032, right=0.998),
        )
        for i, method in enumerate(METHOD_NAMES):
            for j, step in enumerate(STEPS):
                ax = axes[i][j]
                ax.set_xlim(*xlim); ax.set_ylim(*ylim)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                key = (method, step)
                if key not in results:
                    ax.text(0.5, 0.5, "--", ha="center", va="center",
                            transform=ax.transAxes, color="0.6")
                else:
                    Xn, w2 = results[key]
                    ax.scatter(Yn[:, 0], Yn[:, 1], s=3, c="0.82", zorder=1, linewidths=0)
                    ax.scatter(Xn[:, 0], Xn[:, 1], s=3, c=colors, cmap="hsv",
                               zorder=2, linewidths=0)
                    is_best = best_method_at.get(step) == method
                    ax.set_xlabel(rf"$W_2^2$ = {w2:.4f}", fontsize=9, labelpad=2,
                                  fontweight="bold" if is_best else "normal")
                if i == 0:
                    ax.set_title(f"step {step}", fontsize=12)
                if j == 0:
                    ax.set_ylabel(ROW_LABELS[method], fontsize=10)
        out = OUT_DIR / f"gradient_flow_eps_{eps:g}.pdf"
        out_png = OUT_DIR / f"gradient_flow_eps_{eps:g}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
