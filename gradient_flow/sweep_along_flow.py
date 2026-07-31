"""Does the envelope-vs-complete gradient gap survive a sweep over L and eps?

    python -m gradient_flow.sweep_along_flow

along_flow.py measured one cell (L=100, eps=0.01) and found the two gradients
agreeing at the start of the flow and disagreeing by step 50 -- with the gap
attributable to the entropic term, since unrolling <P,C> + eps*KL(P||S) instead
recovers the envelope gradient. Both claims came from a single point in (L, eps).

This runs the full trajectory over a grid and reports, per cell, the cosine at
step 0, its minimum over the flow, its value at the last step, and the norm ratio
there. The complete gradient's O(iters) stored activations are the cost driver,
but the whole grid is a few minutes at N=1000 -- L only scales the support.

At N=1000, 50 steps, 600 inner iterations, cosine at the final step:

           L=10     L=30    L=100    L=300
  eps
  0.003   0.916*   0.835    0.770*   0.777*
  0.01    0.828    0.705    0.611    0.578
  0.03    0.820    0.659    0.431    0.345
  0.1     0.877    0.814    0.658    0.549

Three things hold across every cell. The two gradients start out agreeing
(cos@0 is 0.9968 to 0.9991 everywhere, with no trend), they disagree by the end
(0.35 to 0.92, never recovering), and the entropic explanation survives: the
--regularized control arm's *minimum* cosine over the whole flow is 1.000000 at
eps >= 0.03 and >= 0.99998 at eps = 0.01. So the single-cell result was not an
artifact of that cell, and the gap is the entropic term throughout.

L has a clean monotone effect -- more projections, worse agreement, at every
eps, roughly halving the cosine from L=10 to L=300 at eps=0.03. The support gets
denser and the plan more spread out, which gives the entropic term more to act
on.

The eps dependence is not what the "it is the eps-weighted term" reading
predicts. That reading says the gap should grow with eps; measured, it is
non-monotone, worst at eps=0.03 and *better* again at eps=0.1 (0.345 -> 0.549 at
L=300). eps sets the size of the entropic term but also changes the flow itself:
a larger eps blurs the fixed point the flow descends to, so |g_env| does not
decay as far, and the ratio of the two terms is not simply proportional to eps.
The magnitude of the gap therefore should not be predicted from eps alone; only
its identity as the entropic term is stable, and that is what the control arm
pins down.

The flow is driven by the envelope gradient in every cell, so cells differ in
the trajectory they walk, not only in where the gradients are evaluated. That is
the intended comparison: each cell is the flow you would actually run at those
settings. It also means the eps rows are not strictly comparable point by point.

Cells whose final marginal violation is above VIOL_WARN are flagged (*): even at
600 iterations the eps=0.003 row does not reliably converge, and the cosine
there mixes the effect being measured with plain truncation error. Its numbers
are the least trustworthy in the table -- note its control arm is also the only
one to fall below 0.999.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from gradient_flow.along_flow import trajectory
from gradient_flow.config import N, N_STEPS
from gradient_flow.run import DATA_DIR, OUT_DIR, draw_samples

EPS_GRID = [0.003, 0.01, 0.03, 0.1]
L_GRID = [10, 30, 100, 300]
VIOL_WARN = 1e-3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--no-regularized", action="store_true",
                   help="skip the <P,C>+eps*KL control arm (halves the cost)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU: the support builder is Triton-only")
    reg = not args.no_regularized

    n = args.n
    rng = np.random.default_rng(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device="cuda").float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device="cuda").float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device="cuda")

    print(f"N={n}  steps={args.steps}  inner iters={args.iters}  "
          f"grid: eps={EPS_GRID} x L={L_GRID}")
    hdr = f"\n{'eps':>7} {'L':>5} {'cos@0':>9} {'min cos':>9} {'cos@end':>9} {'ratio@end':>10}"
    print(hdr + (f" {'min cos_reg':>12}" if reg else "") + f" {'viol@end':>10} {'nnz':>8}")

    cells = {}
    for eps in EPS_GRID:
        for n_proj in L_GRID:
            t0 = time.perf_counter()
            s = trajectory(X, Y, a, n, args.steps, args.iters, eps, n_proj,
                           regularized=reg)
            cells[f"{eps}/{n_proj}"] = s
            ratio = s["norm_full"][-1] / s["norm_env"][-1]
            flag = " *" if s["viol"][-1] > VIOL_WARN else ""
            extra = f" {min(s['cos_reg']):>12.6f}" if reg else ""
            print(f"{eps:>7g} {n_proj:>5} {s['cos'][0]:>9.6f} {min(s['cos']):>9.6f} "
                  f"{s['cos'][-1]:>9.6f} {ratio:>10.4f}{extra} "
                  f"{s['viol'][-1]:>10.2e} {s['nnz'][-1]:>8}{flag}", flush=True)
            print(f"          ({time.perf_counter() - t0:.0f}s)", flush=True)

    if any(s["viol"][-1] > VIOL_WARN for s in cells.values()):
        print(f"\n* solve not converged at {args.iters} iterations "
              f"(max |a/r-1| > {VIOL_WARN:g}); cosine there is confounded with "
              f"truncation error.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "sweep_along_flow.json"
    raw.write_text(json.dumps({"eps": EPS_GRID, "L": L_GRID, "steps": args.steps,
                               "iters": args.iters, "n": n, "cells": cells}))
    print(f"wrote {raw}")
    _plot(cells, args.steps)


def _plot(cells, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(steps + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.28))

    # cos vs step, varying eps at fixed L, then varying L at fixed eps
    L_FIX, EPS_FIX = 100, 0.01
    cmap = plt.get_cmap("viridis")
    for ax, (vals, fixed, sym, is_eps) in zip(axes[:2], [
            (EPS_GRID, L_FIX, r"\epsilon", True), (L_GRID, EPS_FIX, "L", False)]):
        for i, v in enumerate(vals):
            key = f"{v}/{fixed}" if is_eps else f"{fixed}/{v}"
            ax.plot(xs, cells[key]["cos"], "-", lw=1.6,
                    color=cmap(i / max(len(vals) - 1, 1)), label=f"${sym}={v:g}$")
        ax.set_xlabel("gradient step")
        ax.set_ylabel(r"cos$(g_{\rm env},\,g_{\rm full})$")
        ax.set_ylim(top=1.02)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_title(f"varying $\\epsilon$ at $L={L_FIX}$")
    axes[1].set_title(f"varying $L$ at $\\epsilon={EPS_FIX}$")

    # final-step cosine over the grid
    M = np.array([[cells[f"{e}/{l}"]["cos"][-1] for l in L_GRID] for e in EPS_GRID])
    ax = axes[2]
    im = ax.imshow(M, cmap="magma", vmin=min(0.0, M.min()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(L_GRID)), [str(v) for v in L_GRID])
    ax.set_yticks(range(len(EPS_GRID)), [f"{v:g}" for v in EPS_GRID])
    ax.set_xlabel("$L$ (projections)")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title(f"cosine at step {steps}")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    out = OUT_DIR / "sweep_along_flow.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
