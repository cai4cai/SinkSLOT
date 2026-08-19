"""Is the falling cosine an artefact of finite Euler steps?

    python -m gradient_flow.step_size

The natural reading of along_flow.py's curve is that the two gradients differ a
little at every step and the difference piles up over 50 Euler steps, so that
shrinking the step size would keep the cosine flat.

That reading does not match how the quantity is measured. trajectory() advances X
with g_env alone; g_full is evaluated at the same X and never applied. There are
not two trajectories drifting apart -- there is one trajectory, with both
gradients measured at each point on it. The cosine at step t is therefore a
property of the configuration X_t, not an accumulated integration error.

That distinction is testable. Run the same flow at lr, lr/2 and lr/5, with the
step count scaled so all three cover the same flow time T = steps * lr, and plot
the cosine against t = step * lr rather than against the step index. If the
cosine is a property of position, the three curves lie on top of each other: the
smaller step sizes just sample the same underlying path more finely. If it were
accumulated discretisation error, the smaller steps would decay more slowly.

Measured at N=1000, eps=0.01, L=100, 600 inner iterations, T = 2.5:

        t      lr=0.05    lr=0.025    lr=0.01
     0.50       0.9994      0.9994     0.9994
     1.00       0.9955      0.9962     0.9964
     1.50       0.9685      0.9721     0.9744
     2.00       0.8298      0.8533     0.8638
     2.50       0.6239      0.6342     0.6380

The curves collapse. Against the step index the three look completely different
-- the lr/5 run takes 250 steps to do what the base run does in 50 -- and against
flow time they lie on one another. So the decay is not integration error being
accumulated; it is where the flow has got to.

There is a small, real step-size effect on top: the endpoint creeps up 0.6239 ->
0.6342 -> 0.6380 as lr shrinks. The increments (0.0103 then 0.0038) shrink about
as fast as lr does, which is the first-order Euler behaviour, and extrapolating
linearly in lr puts the continuous-time value at roughly 0.64.

That is the answer to "should the cosine stay the same under infinitely small
steps": no. It converges to ~0.64, not to 1. Finite steps account for about 2
points of the drop and the remaining 36 belong to the continuous flow itself.
The gradients genuinely disagree at the fixed point, which is what the entropic
term implies -- shrinking the step size cannot remove a difference in the
objective being differentiated.

(The accumulation reading *is* right about flow_qualitative.py, where each row is
driven by its own gradient. Those are genuinely two trajectories, and their W2^2
gap does build up over the run. The two figures measure different things.)
"""
from __future__ import annotations

import argparse
import json

import torch

from gradient_flow.along_flow import trajectory
from gradient_flow.config import L, LR, N, N_STEPS
from gradient_flow.run import DATA_DIR, DEVICE, OUT_DIR, draw_samples

EPS = 0.01
FACTORS = [1, 2, 5]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS, help="steps at the base lr")
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()

    if DEVICE != "cuda":
        print("gradient_flow/step_size.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    n = args.n
    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", n, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device=DEVICE).float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=DEVICE)

    T = args.steps * LR
    print(f"N={n}  eps={EPS}  L={L}  iters={args.iters}  flow time T={T:g}")
    runs = {}
    for f in FACTORS:
        lr, steps = LR / f, args.steps * f
        s = trajectory(X, Y, a, n, steps, args.iters, EPS, L, lr=lr)
        runs[str(f)] = {"lr": lr, "steps": steps, "cos": s["cos"],
                        "norm_env": s["norm_env"], "t": [i * lr for i in range(steps + 1)]}
        print(f"  lr={lr:<8.4g} steps={steps:<5} cos@T={s['cos'][-1]:.4f}  "
              f"|g_env|@T={s['norm_env'][-1]:.4e}", flush=True)

    # Compare at matched flow time: every factor lands on t = T exactly.
    print(f"\n{'t':>7}" + "".join(f"{'lr/' + str(f):>12}" for f in FACTORS))
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        t = T * frac
        row = f"{t:>7.2f}"
        for f in FACTORS:
            r = runs[str(f)]
            i = min(int(round(t / r["lr"])), r["steps"])
            row += f"{r['cos'][i]:>12.4f}"
        print(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = OUT_DIR / "step_size.json"
    raw.write_text(json.dumps({"T": T, "eps": EPS, "L": L, "runs": runs}))
    print(f"wrote {raw}")
    _plot(runs, T)


def _plot(runs, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.6, 4.1),
                                   gridspec_kw=dict(wspace=0.26))
    style = {"1": ("-", "#1f77b4"), "2": ("--", "#e07b39"), "5": (":", "#2b3a67")}
    for f in FACTORS:
        r = runs[str(f)]
        ls, col = style[str(f)]
        ax0.plot(range(r["steps"] + 1), r["cos"], ls, lw=1.8, color=col,
                 label=f"lr/{f} ({r['steps']} steps)")
        ax1.plot(r["t"], r["cos"], ls, lw=1.8, color=col, label=f"lr/{f}")
    ax0.set_xlabel("gradient step (index)")
    ax1.set_xlabel(r"flow time $t = \mathrm{step}\times\mathrm{lr}$")
    for ax in (ax0, ax1):
        ax.set_ylabel(r"cos$(g_{\rm env},\,g_{\rm full})$")
        ax.set_ylim(top=1.02)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax0.set_title("vs step index: smaller steps look slower", fontsize=10.5)
    ax1.set_title("vs flow time: the curves collapse", fontsize=10.5)
    fig.suptitle("The cosine is a property of where the flow is, not of the "
                 "Euler step size", fontsize=12, y=1.02)

    out = OUT_DIR / "step_size.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {out}\nwrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
