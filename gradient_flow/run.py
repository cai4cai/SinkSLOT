"""Gradient flow, blob -> crescent: SOT / EOT / SROT / SinkSLOT, one compiled figure.

    python -m gradient_flow.run

Runs on CPU or CUDA. A CUDA GPU with Triton reproduces the reported numbers
(the fused kernels, see sinkslot/sinkhorn_solvers.py); without one, slot_grad
falls back to a pure-torch path (same algorithm, much slower) automatically.

Four methods, one figure:
  * SOT (Bonneel et al., 2015): the plain, unregularized sliced W2 distance --
    no Sinkhorn solve at all, each projection's 1-D transport is exact (a
    sort), so autograd through `ot.sliced_wasserstein_distance` already gives
    the analytical gradient.
  * EOT (Feydy et al., 2019): `sinkhorn_divergence_torch_autograd`.
  * SROT (Nguyen 2026): `sr_sinkhorn_divergence_torch_autograd`, delta=1e-8.
  * SinkSLOT (ours): the native solver (torch-ext/sinkslot/sinkhorn_solvers.py),
    the exact pipeline this repo's own speed benchmarks exercise,
    plus the closed-form envelope-theorem gradient
    grad_X SLOT_eps(X,Y) = 2*diag(a)*(X - T_eps(X)) (see sinkslot/gradient.py).

    SOT/EOT/SROT run in float64 (vendor/sinkhorn_methods.py, ported from a
    sibling research repository's own copy, which has no CUDA/Triton
    dependency). SinkSLOT runs in float32 on the Triton path: its fused cost
    kernel accumulates in fp32 regardless of input dtype (see
    sparse_sqeuclidean_cost's own docstring), so there is no float64 path
    there. The pure-torch fallback (used automatically without a GPU) has no
    such restriction and follows X/Y/a's own dtype, but this script still
    passes fp32 either way for a like-for-like comparison across both
    backends. In practice the float32-vs-float64 difference is invisible at
    4-decimal reporting precision: this script's Triton-path output (W2^2 per
    step, per method) matches a float64 reference run exactly, digit for
    digit.

Compiled figure formatting: compact vertically with no dead space between
adjacent method rows; for every intermediate checkpoint (step > 0), the
lowest -- i.e. best -- W2^2 across the 4 methods is bolded.
"""
from __future__ import annotations

import time
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
from gradient_flow.vendor.sinkhorn_methods import (
    sinkhorn_divergence_torch_autograd, sr_sinkhorn_divergence_torch_autograd,
)
from sinkslot.gradient import slot_grad

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

    Calls POT directly (ot.dist, ot.emd2) rather than through vendor's
    exact_ot/build_cost -- POT's own backend accepts torch tensors natively
    (verified against 0.9.7), so no numpy round-trip is needed here at all.
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
            g = slot_grad(x_i_d.float(), Y.float(), a_t.float(), eps, L, seed=0,
                          n_iters=MAX_ITER).double()
        else:
            x_i.requires_grad_(True)
            if method == "SOT":
                loss = ot.sliced_wasserstein_distance(x_i, Y, n_projections=L, p=2, seed=0)
            elif method == "EOT":
                loss = sinkhorn_divergence_torch_autograd(x_i, Y, eps, max_iter=MAX_ITER, tol=0.0)
            else:  # SROT
                loss = sr_sinkhorn_divergence_torch_autograd(
                    x_i, Y, eps, L=L, max_iter=MAX_ITER, tol=0.0, delta=DELTA_SROT)
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
    if DEVICE != "cuda":
        print("gradient_flow/run.py: no CUDA GPU found, running the SinkSLOT arm on the "
              "pure-torch fallback (slot_grad's backend='auto') -- much slower than the "
              "fused Triton kernels the reported numbers use, but the same algorithm.")

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
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
