"""Why the envelope-vs-complete curve is erratic at small L.

    python -m gradient_flow.projection_noise

In sweep_along_flow.py's middle panel the cosine curve wiggles visibly at L=10
and L=30 -- non-monotone step to step, with bumps of a few times 1e-2 -- while
L=100 and L=300 are smooth. The suspicion is that this is not a property of the
gradients at all but sampling noise in the support: sot_plan_coo builds the
sparse support from L random 1-D projections, so L directly sets how many
independent draws the support averages over (~9.3k nnz at L=10 against ~72k at
L=100). As X moves, one projection's sort order flips and a correspondingly
larger fraction of a small support changes.

If that is the cause, re-running an identical cell with different projection
seeds should scatter widely at small L and collapse onto one curve at large L,
since everything else -- data, X0, eps, iteration count, step rule -- is held
fixed and the seed only redraws the projections.

Two numbers per (L, seed) group:
  spread   std across seeds of the final-step cosine, i.e. how much the answer
           depends on which projections you happened to draw
  rough    mean |second difference| of the cosine along the flow, a scale-free
           measure of step-to-step wiggle within a single run

Measured at N=1000, eps=0.01, 50 steps, 300 inner iterations, 5 seeds:

      L     mean cos@end      spread       rough        nnz
     10           0.8403    1.77e-02    5.69e-03       9690
     30           0.6996    1.85e-02    3.80e-03      26546
    100           0.6171    8.65e-03    2.17e-03      70653
    300           0.5881    5.58e-03    1.33e-03     168576

The plot is the clearer evidence: at L=10 the five seeds visibly scatter and
each curve wiggles; by L=300 they lie on top of one another as a single smooth
line. The within-run roughness falls 4.3x from L=10 to L=300 and the seed-to-
seed spread 3.2x, over a 30x range in L -- both close to the 5.5x that 1/sqrt(L)
predicts for a Monte Carlo average over L directions, and nowhere near the 30x
that would follow if the support's *size* (nnz, up 17x) set the noise directly.
The 1/sqrt(L) rate is the signature of the projection sampling.

Caveat on the spread column: 5 seeds pin a standard deviation only to about
+-35%, and L=10 and L=30 come out equal within that. Read the column as an
order of magnitude, not a curve; the roughness column, which averages 49 second
differences per run, is the better-resolved of the two.

So the wiggle at small L is support-resampling noise, not structure -- it says
nothing about the envelope-vs-complete gap itself. The gap survives it: the mean
final cosine still falls monotonically with L (0.84 -> 0.59), and that variation
is an order of magnitude larger than the spread within any single L. The
practical reading is that the small-L cells in sweep_along_flow.py carry an error
bar of a couple of 1e-2, and the monotone L trend reported there is well clear
of it.
"""
from __future__ import annotations

import argparse
import json

import torch

from gradient_flow.along_flow import trajectory
from gradient_flow.config import N, N_STEPS
from gradient_flow.run import DATA_DIR, DEVICE, OUT_DIR, draw_samples

L_GRID = [10, 30, 100, 300]
SEEDS = [0, 1, 2, 3, 4]
EPS = 0.01


def roughness(c):
    """Mean |second difference| -- wiggle that is not a smooth trend."""
    return float(torch.tensor(c).diff(n=2).abs().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()

    if DEVICE != "cuda":
        print("gradient_flow/projection_noise.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    n = args.n
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device=DEVICE).float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=DEVICE)

    print(f"N={n}  eps={EPS}  steps={args.steps}  iters={args.iters}  seeds={SEEDS}")
    print(f"\n{'L':>7} {'mean cos@end':>14} {'spread':>11} {'rough':>11} {'nnz':>10}")

    runs = {}
    for n_proj in L_GRID:
        finals, roughs = [], []
        for seed in SEEDS:
            s = trajectory(X, Y, a, n, args.steps, args.iters, EPS, n_proj, seed=seed)
            runs[f"{n_proj}/{seed}"] = s["cos"]
            finals.append(s["cos"][-1])
            roughs.append(roughness(s["cos"]))
            nnz = s["nnz"][-1]
        finals_t, roughs_t = torch.tensor(finals), torch.tensor(roughs)
        # unbiased=False to match numpy's std() default (ddof=0) -- matters at n=5 seeds.
        print(f"{n_proj:>7} {float(finals_t.mean()):>14.4f} "
              f"{float(finals_t.std(unbiased=False)):>11.2e} "
              f"{float(roughs_t.mean()):>11.2e} {nnz:>10}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "projection_noise.json"
    raw.write_text(json.dumps({"L": L_GRID, "seeds": SEEDS, "eps": EPS,
                               "steps": args.steps, "runs": runs}))
    print(f"wrote {raw}")
    _plot(runs, args.steps)


def _plot(runs, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(steps + 1))
    fig, axes = plt.subplots(1, len(L_GRID), figsize=(3.1 * len(L_GRID), 3.3),
                             sharey=True, gridspec_kw=dict(wspace=0.08))
    cmap = plt.get_cmap("plasma")
    for ax, n_proj in zip(axes, L_GRID):
        for i, seed in enumerate(SEEDS):
            ax.plot(xs, runs[f"{n_proj}/{seed}"], "-", lw=1.2,
                    color=cmap(0.12 + 0.7 * i / max(len(SEEDS) - 1, 1)), alpha=0.9)
        ax.set_title(f"$L={n_proj}$")
        ax.set_xlabel("gradient step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"cos$(g_{\rm env},\,g_{\rm full})$")
    fig.suptitle(f"Same cell, {len(SEEDS)} projection seeds "
                 f"($N={N}$, $\\epsilon={EPS}$) -- spread shrinks as $L$ grows",
                 y=1.02)

    out = OUT_DIR / "projection_noise.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
