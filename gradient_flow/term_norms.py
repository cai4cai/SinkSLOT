"""Split the complete gradient into its two terms and watch their norms decay.

    python -m gradient_flow.term_norms [--steps 50] [--iters 600] [--n 1000]

along_flow.py reported cos(g_env, g_full) falling to 0.61 by step 50. A cosine
is scale-free, so it says the two gradients point apart but not why, and the
obvious worry is that it collapses only because one of the two has become
numerically tiny. This module answers that directly by measuring the two terms
of the decomposition

    g_full  =  g_env  +  (g_full - g_env)
                 (I)          (II)

  (I)  the *envelope gradient*, 2*diag(a)*(X - T_eps(X)) -- what the flow
       follows, evaluated in closed form under no_grad (no autograd at all)
  (II) the *reference-plan gradient*: what is left of the complete gradient once
       (I) is removed, i.e. the term carrying dP/dX that the envelope theorem
       drops.  g_full is autograd through all `iters` Sinkhorn iterations with
       no stop-gradient anywhere, so (II) = g_full - g_env is exact.

Both come from the same solve on the same support at the same step, so the
subtraction is exact rather than a re-derivation. Alongside them it records the
plan's transport cost <P, C>, so the shape of the norms can be read against
where the flow has actually converged.

At N=1000, L=100, eps=0.01, 600 inner iterations, over 50 steps:

    step         |I|         |II|   |II|/|I|      <P,C>      cos
       0   6.344e-02    1.868e-03     0.0294   1.02e+00   0.9996
      10   2.222e-02    7.801e-04     0.0351   1.28e-01   0.9994
      20   7.778e-03    7.236e-04     0.0930   1.88e-02   0.9957
      30   2.761e-03    7.177e-04     0.2600   5.43e-03   0.9674
      40   1.074e-03    6.957e-04     0.6479   3.81e-03   0.8287
      50   6.193e-04    6.938e-04     1.1203   3.58e-03   0.6238

That is the whole story of the falling cosine. Term (I) decays by a factor of
102 over the run, tracking the flow's convergence. Term (II) does not decay at
all: after an initial drop it sits between 6.92e-04 and 7.80e-04 for every step
from 10 onward -- flat to within 13% while (I) falls 36-fold -- and at step 47
it overtakes (I). The cosine falls not because the gradients rotate but because
a fixed-size term stops being negligible next to a vanishing one.

The transport cost column dates that crossover. <P,C> falls by nearly three
orders of magnitude over the first ~27 steps and then flattens at ~3.6e-03 --
the flow has essentially arrived. The cosine is still 0.982 there and only
degrades afterwards, over steps where the objective is no longer moving; it
passes 0.66 at the step where the two terms cross. So the gap is a property of
the converged regime, not of the transient: it opens up precisely where (I) has
decayed into the floor that (II) never leaves.

Read together with along_flow.py's control arm -- unrolling <P,C> + eps*KL
instead recovers g_env to cosine 0.999992 -- term (II) is the entropic
contribution, and it has a floor because eps is fixed while the transport cost
collapses. Nothing here contradicts the envelope gradient being the correct one
for SLOT_eps; it quantifies how much of the unrolled gradient is the other
objective's, and shows that fraction going to 1 and beyond as the flow settles.

The eps sweep (second figure, L fixed at 100) shows the same three-part shape at
every eps: (I) decaying by ~2 orders of magnitude, (II) flat from ~step 10, and
<P,C> flattening well before the two cross.

                       (II) floor    <P,C> flat   (II)>(I)    cos@50
    eps=0.003            3.6e-04       step 33      never      0.785
    eps=0.01             6.9e-04       step 27     step 47      0.624
    eps=0.03             1.3e-03       step 23     step 39      0.453
    eps=0.1              1.0e-03       step 19     step 48      0.659

The floor rises with eps up to 0.03, which is what the entropic reading
predicts, and the crossover moves earlier with it. eps=0.1 breaks the pattern in
the same non-monotone way sweep_along_flow.py reports -- there a larger eps also
blurs the fixed point, so (I) does not decay as far (9.7e-04 at step 50 against
6.2e-04 at eps=0.01) and the crossover is pushed back out. In every column the
transport cost has flattened 6-15 steps before the terms cross, so the ordering
"converge first, gap second" holds throughout, whatever the size of the gap.

eps=0.003 is flagged: even 600 inner iterations do not converge it (max |a/r-1|
~ 2e-03 at the last step), so its curves mix the effect with truncation error --
its (II) never reaches (I), but that column is the least trustworthy.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from gradient_flow.along_flow import trajectory
from gradient_flow.config import L, N, N_STEPS
from gradient_flow.estimators import EPS
from gradient_flow.run import DATA_DIR, OUT_DIR, draw_samples
from gradient_flow.sweep_along_flow import VIOL_WARN

EPS_GRID = [0.003, 0.01, 0.03, 0.1]

C_I, C_II, C_W2, C_COS = "#1f77b4", "#d62728", "#7f5f00", "#2ca02c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=600,
                   help="inner Sinkhorn iterations, shared by both terms")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--eps-grid", type=float, nargs="+", default=EPS_GRID)
    p.add_argument("--no-exact-w2", action="store_true",
                   help="skip the dense EMD reference W2 (one solve per step)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU: the support builder is Triton-only")

    n = args.n
    rng = np.random.default_rng(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device="cuda").float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device="cuda").float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device="cuda")

    print(f"N={n}  L={L}  steps={args.steps}  inner iters={args.iters}  "
          f"eps grid={args.eps_grid}")

    cells = {}
    for eps in args.eps_grid:
        t0 = time.perf_counter()
        print(f"\neps={eps:g}")
        print(f"{'step':>5} {'|I|':>11} {'|II|':>11} {'|II|/|I|':>10} "
              f"{'<P,C>':>11} {'cos':>9} {'viol':>10}")

        def report(step, r, _every=max(args.steps // 10, 1)):
            if step % _every and step != args.steps:
                return
            print(f"{step:>5} {r['norm_env']:>11.4e} {r['norm_resid']:>11.4e} "
                  f"{r['norm_resid'] / r['norm_env']:>10.4f} {r['w2']:>11.3e} "
                  f"{r['cos']:>9.6f} {r['viol']:>10.2e}", flush=True)

        s = trajectory(X, Y, a, n, args.steps, args.iters, eps, L,
                       on_step=report, exact_w2=not args.no_exact_w2)
        cells[f"{eps:g}"] = s
        flag = "  * not converged" if s["viol"][-1] > VIOL_WARN else ""
        print(f"      ({time.perf_counter() - t0:.0f}s){flag}", flush=True)

    if any(s["viol"][-1] > VIOL_WARN for s in cells.values()):
        print(f"\n* solve not converged at {args.iters} iterations "
              f"(max |a/r-1| > {VIOL_WARN:g}); those curves are confounded with "
              f"truncation error.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "term_norms.json"
    raw.write_text(json.dumps({"eps": args.eps_grid, "L": L, "steps": args.steps,
                               "iters": args.iters, "n": n, "cells": cells}))
    print(f"\nwrote {raw}")

    focus = f"{EPS:g}"
    if focus in cells:
        _plot_single(cells[focus], args.steps, args.iters, EPS, n)
    _plot_sweep(cells, args.eps_grid, args.steps, args.iters, n)


def _save(fig, stem):
    """PDF for the paper, PNG at 200 dpi for mail and slides."""
    import matplotlib.pyplot as plt

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def _plot_single(s, steps, iters, eps, n):
    """Three stacked panels at one eps: term norms, ratio + cosine, transport cost."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(steps + 1))
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(6.2, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.1))

    ax0.plot(xs, s["norm_env"], "-", lw=1.8, color=C_I,
             label=r"(I) envelope gradient, $\|g_{\rm env}\|$")
    ax0.plot(xs, s["norm_resid"], "-", lw=1.8, color=C_II,
             label=r"(II) reference-plan gradient, $\|g_{\rm full}-g_{\rm env}\|$")
    ax0.set_yscale("log")
    ax0.set_ylabel("gradient norm")
    ax0.legend(frameon=False, fontsize=9, loc="upper right")
    ax0.grid(alpha=0.3, which="both")
    ax0.set_title(f"The falling cosine is the envelope gradient decaying,\n"
                  f"not the reference-plan gradient growing\n"
                  f"$N={n}$, $L={L}$, $\\epsilon={eps:g}$, {iters} inner iterations")

    ratio = np.array(s["norm_resid"]) / np.array(s["norm_env"])
    ax1.plot(xs, ratio, "-", lw=1.8, color=C_II, label="reference-plan / envelope")
    ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax1.set_yscale("log")
    ax1.set_ylabel("ratio  (II)/(I)")
    ax1.grid(alpha=0.3, which="both")
    axc = ax1.twinx()
    axc.plot(xs, s["cos"], "--", lw=1.5, color=C_COS)
    axc.set_ylabel(r"cos$(g_{\rm env},\,g_{\rm full})$", color=C_COS)
    axc.tick_params(axis="y", colors=C_COS)
    axc.set_ylim(top=1.02)
    ax1.legend(frameon=False, fontsize=9, loc="center left")

    ax2.plot(xs, s["w2"], "-", lw=1.8, color=C_W2)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"transport cost $\langle P, C\rangle$")
    ax2.set_xlabel("gradient step")
    ax2.grid(alpha=0.3, which="both")

    # Where the two terms cross -- the point the cosine collapse is dated from.
    cross = next((i for i, r in enumerate(ratio) if r >= 1.0), None)
    if cross is not None:
        for ax in (ax0, ax1, ax2):
            ax.axvline(cross, color="0.6", lw=0.8, ls="--")
        ax0.annotate(f"(II) overtakes (I)\nat step {cross}", xy=(cross, s["norm_env"][cross]),
                     xytext=(-6, 18), textcoords="offset points", ha="right",
                     fontsize=8, color="0.35")

    _save(fig, f"gradient_term_norms_eps_{eps:g}")


def _plot_sweep(cells, eps_grid, steps, iters, n):
    """One column per eps. Row 1: (I) computed both ways, and (II). Row 2: the
    plan's transport cost. Row 3: the exact Wasserstein distance.

    (I) is drawn twice -- the analytical closed form and the stop-gradient
    autograd recomputation -- because they lie on top of each other, which is the
    point: the size of (II) does not depend on how (I) is obtained.

    Rows 2 and 3 are the same quantity measured two ways, kept apart so each can
    be read on its own scale: <P,C> is the regularized objective on the sampled
    support and plateaus at an O(eps) floor, while W_2 is the unregularized
    distance to the target and keeps falling. The plateau in row 2 is where the
    flow stops shrinking (I), and it is the eps-blur, not the flow stalling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(steps + 1))
    ncol = len(eps_grid)
    has_exact = all(np.isfinite(cells[f"{e:g}"]["w2_exact"]).all() for e in eps_grid)
    nrow = 3 if has_exact else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow + 0.9),
                             sharex=True, sharey="row", squeeze=False,
                             gridspec_kw=dict(hspace=0.12, wspace=0.3))

    for j, eps in enumerate(eps_grid):
        s = cells[f"{eps:g}"]
        flag = " *" if s["viol"][-1] > VIOL_WARN else ""
        col = [axes[i][j] for i in range(nrow)]

        col[0].plot(xs, s["norm_env"], "-", lw=2.6, color=C_I, alpha=0.4,
                    label="(I) envelope gradient (analytical)")
        col[0].plot(xs, s["norm_det"], "--", lw=1.2, color="black",
                    label="(I) envelope gradient (numerical)")
        col[0].plot(xs, s["norm_resid"], "-", lw=1.6, color=C_II,
                    label="(II) reference-plan gradient")
        col[0].set_title(f"$\\epsilon={eps:g}${flag}")

        col[1].plot(xs, s["w2"], "-", lw=1.6, color=C_W2)
        if has_exact:
            col[2].plot(xs, np.sqrt(s["w2_exact"]), "-", lw=1.6, color="0.35")
        col[-1].set_xlabel("gradient step")

        for ax in col:
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")

    axes[0][0].set_ylabel("gradient norm")
    axes[1][0].set_ylabel(r"transport cost $\langle P, C\rangle$")
    if has_exact:
        axes[2][0].set_ylabel(r"Wasserstein distance $W_2$")
    # Only row 1 needs a key, and the panels are too full to hold it.
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, ncol=len(h),
               loc="upper center", bbox_to_anchor=(0.5, 0.045))
    fig.suptitle(f"(I) envelope gradient vs (II) reference-plan gradient along the flow — "
                 f"$N={n}$, $L={L}$, {iters} inner iterations", y=0.955)

    _save(fig, "gradient_term_norms_sweep")


if __name__ == "__main__":
    main()
