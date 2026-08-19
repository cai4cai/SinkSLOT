"""Is the envelope gradient an artefact of the closed form? Recompute it by autograd.

    python -m gradient_flow.closed_form_check [--steps 50] [--iters 600]

term_norms.py computes term (I), the envelope gradient, from the barycentric
closed form 2*diag(a)*(X - T_eps(X)) under no_grad -- no autograd anywhere. That
leaves a fair worry: if the closed form were subtly wrong, term (II) would be
whatever the mistake is, and the whole result would be an artefact of how (I) is
written rather than a property of the objective.

This checks it against a route that never touches the formula. Stop-gradient the
converged plan and let autograd differentiate <P, C> through the cost alone:

  closed form   2*diag(a)*(X - T_eps(X)), evaluated directly       (g_env)
  stop-gradient autograd of <detach(P), C> wrt X                   (g_det)

Both use the same solve, the same support, the same X. They are the same
gradient up to one known factor: the closed form divides by the plan's target
row mass a_i, autograd by its achieved row mass r_i = sum_j P_ij, so
g_env = (a/r) * g_det exactly, and the two coincide as the marginal violation
|a/r - 1| goes to zero.

At N=1000, L=100, eps=0.01, 600 inner iterations, over 50 steps:

    step   cos(closed, stopgrad)   rel. diff   |II| via closed   |II| via stopgrad
       0             1.00000012     3.3e-06        1.8676e-03          1.8676e-03
      10             1.00000000     7.4e-05        7.7286e-04          7.7287e-04
      20             1.00000000     4.8e-05        7.3268e-04          7.3267e-04
      30             1.00000000     1.1e-04        7.0611e-04          7.0611e-04
      40             1.00000000     1.0e-04        7.0229e-04          7.0229e-04
      50             1.00000000     1.0e-04        6.9099e-04          6.9099e-04

The two routes agree to 1.0 in cosine at every step (to fp32 resolution -- the
1.00000012 at step 0 is rounding) and to ~1e-04 in relative norm, which is four
orders of magnitude below the ratio term (II)/(I) reaches, and which tracks the
marginal violation as predicted rather than growing along the flow. Term (II)
measured against the autograd gradient is the same curve as term (II) measured
against the closed form, to five digits, at every step.

So the closed form is not the source of the gap. Whichever way (I) is computed,
the complete gradient carries a component that (I) does not, it does not shrink
as the flow converges, and it overtakes (I) at the same step.
"""
from __future__ import annotations

import argparse
import json

import torch

from gradient_flow.along_flow import trajectory
from gradient_flow.config import L, N, N_STEPS
from gradient_flow.estimators import EPS
from gradient_flow.run import DATA_DIR, DEVICE, OUT_DIR, draw_samples
from gradient_flow.term_norms import C_I, C_II, _save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--eps", type=float, default=EPS)
    args = p.parse_args()

    if DEVICE != "cuda":
        print("gradient_flow/closed_form_check.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    n = args.n
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device=DEVICE).float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=DEVICE)

    print(f"N={n}  L={L}  eps={args.eps:g}  steps={args.steps}  "
          f"inner iters={args.iters}")
    print(f"\n{'step':>5} {'cos(closed,stopgrad)':>21} {'rel diff':>10} "
          f"{'|II| closed':>12} {'|II| stopgrad':>14} {'|a/r-1|':>10}")

    def report(step, r, _every=max(args.steps // 10, 1)):
        if step % _every and step != args.steps:
            return
        print(f"{step:>5} {r['cos_det']:>21.8f} {r['rel_det']:>10.1e} "
              f"{r['norm_resid']:>12.4e} {r['norm_resid_det']:>14.4e} "
              f"{r['viol']:>10.1e}", flush=True)

    s = trajectory(X, Y, a, n, args.steps, args.iters, args.eps, L, on_step=report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "closed_form_check.json"
    raw.write_text(json.dumps({"eps": args.eps, "L": L, "steps": args.steps,
                               "iters": args.iters, "n": n, "s": s}))
    print(f"\nwrote {raw}")
    _plot(s, args.steps, args.iters, args.eps, n)


def _plot(s, steps, iters, eps, n):
    """Top: (I) both ways, with their relative difference. Bottom: (II) both ways."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(steps + 1))
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.4, 6.2), sharex=True,
        gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.1))

    ax0.plot(xs, s["norm_env"], "-", lw=2.4, color=C_I, alpha=0.45,
             label="(I) closed form")
    ax0.plot(xs, s["norm_det"], "--", lw=1.3, color="black",
             label="(I) stop-gradient autograd")
    ax0.set_yscale("log")
    ax0.set_ylabel("envelope gradient norm")
    ax0.legend(frameon=False, fontsize=9, loc="lower center")
    ax0.grid(alpha=0.3, which="both")
    ax0.set_title(f"The envelope gradient does not depend on how it is computed\n"
                  f"$N={n}$, $L={L}$, $\\epsilon={eps:g}$, {iters} inner iterations")
    # The two routes differ only by the row-mass normalisation, so their relative
    # difference should *be* the marginal violation. Overlaying it says so: the
    # jitter in one is the jitter in the other, not a drift along the flow.
    axr = ax0.twinx()
    axr.plot(xs, s["rel_det"], ":", lw=1.6, color="0.35",
             label="relative difference, closed form vs stop-gradient")
    axr.plot(xs, s["viol"], "-", lw=1.0, color="0.62",
             label=r"marginal violation, max $|a/r-1|$")
    axr.set_yscale("log")
    axr.set_ylabel("relative difference / marginal violation", color="0.45",
                   fontsize=9)
    axr.tick_params(axis="y", colors="0.45")
    # Headroom so the two grey curves clear the (I) curves they sit on top of.
    axr.set_ylim(top=12 * max(max(s["rel_det"]), max(s["viol"])))
    axr.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor="0.35")

    ax1.plot(xs, s["norm_resid"], "-", lw=2.4, color=C_II, alpha=0.45,
             label="(II) = complete $-$ closed form")
    ax1.plot(xs, s["norm_resid_det"], "--", lw=1.3, color="black",
             label="(II) = complete $-$ stop-gradient")
    ax1.set_yscale("log")
    ax1.set_ylabel("reference-plan gradient norm")
    ax1.set_xlabel("gradient step")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3, which="both")

    _save(fig, "closed_form_vs_stopgrad")


if __name__ == "__main__":
    main()
