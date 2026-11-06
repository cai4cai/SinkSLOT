"""Gradient flow, blob -> crescent, for SOT, EOT, SROT and SinkSLOT; one figure per eps.

    python -m gradient_flow.run

Each step moves the source points by X <- X - LR * N * grad_X D(X, Y), in float64 on CPU or CUDA:
  * SOT (Bonneel et al., 2015): sliced W2 (`ot.sliced_wasserstein_distance`), gradient by autograd.
  * EOT (Feydy et al., 2019): debiased Sinkhorn divergence (`eot_divergence`), envelope-theorem gradient.
  * SROT (Nguyen 2026): debiased SROT divergence with the dense smoothed reference (`srot_divergence`),
    envelope-theorem gradient.
  * SinkSLOT (ours): `sinkslot.gradient.slot_grad`, the closed-form envelope-theorem gradient
    grad_X SLOT_eps(X, Y) = 2 diag(a) (X - T_eps(X)), on its torch backend.
The regularised arms run MAX_ITER Sinkhorn iterations per step; envelope-theorem gradients hold the plan
fixed and differentiate only the cost.

The figure shows each method (rows) at the checkpoints in STEPS (columns), labelled with the exact W2^2;
at each intermediate checkpoint, the lowest W2^2 is in bold.
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
    """Debiased Sinkhorn divergence OT(X, Y) - OT(X, X) / 2 - OT(Y, Y) / 2, each term
    <C, P> + eps KL(P | a b^T) with C = ||x - y||^2 (POT's `solve_sample`, `debias=True`).

    Exactly n_iters log-domain Sinkhorn iterations per term (tol=-1 disables the early stop; the
    resulting non-convergence warning is suppressed). grad="envelope" sets the gradient from the
    converged plan with the plan held fixed.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sinkhorn did not converge")
        res = ot.solve_sample(
            X_t, Y_t, a_t, b_t, reg=eps, max_iter=n_iters, tol=-1.0,
            grad="envelope", debias=True,
        )
    return res.value


def srot_divergence(X, Y, a, b, eps, L, seed, n_iters, gamma):
    """Surrogate <C, P_XY> - <C, P_XX> / 2 whose gradient in X is the envelope-theorem gradient of
    the debiased SROT divergence (Nguyen 2026, Def. 2); the Y-Y term does not depend on X.

    Each plan solves SROT with the dense reference (1 - gamma) P^SOT + gamma a b^T on every pair
    (P^SOT from sot_plan_coo, the same slices as SinkSLOT) by n_iters log-domain Sinkhorn
    iterations, without gradient tracking; only the cost is differentiated.
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
        cost = ((Xp[:, None, :] - Yp[None, :, :]) ** 2).sum(-1)  # finite gradient at x_i = y_j, unlike cdist
        return (plan * cost).sum()

    return term(X, Y, a, b) - 0.5 * term(X, X, a, a)

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
DTYPE = torch.float64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_density(fname):
    """Grayscale density in [0, 1], row-flipped and inverted as in Feydy et al. (2019)."""
    img = Image.open(fname).convert("L")
    w, h = img.size
    buf = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).reshape(h, w)
    arr = buf.double() / 255.0
    arr = arr.flip(0)
    return 1.0 - arr


def draw_samples(fname, n, rng, dtype=DTYPE, device="cpu"):
    """n density-weighted pixel samples with sub-pixel jitter, scaled by DATA_SCALE; `rng` is a CPU torch.Generator."""
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
    """Exact W2^2 between uniform point clouds X and Y (POT's network simplex, float64, CPU)."""
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

        # Lowest W2^2 per intermediate checkpoint (first and last steps excluded), shown in bold.
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
