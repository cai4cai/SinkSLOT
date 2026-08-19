"""Does the envelope-vs-complete gradient gap change the flow you actually see?

    python -m gradient_flow.appendix_checks.flow_qualitative

along_flow.py and sweep_along_flow.py establish that the two gradients diverge
to a cosine of ~0.61 by step 50. That is a statement about directions in R^(Nx2);
it does not say whether a practitioner looking at the transported point cloud
would notice. This runs run.py's blob -> crescent flow three times, changing only
which gradient drives the descent, and lays the point clouds out in run.py's
panel format with the exact W2^2 underneath:

  envelope    grad_X SLOT_eps = 2*diag(a)*(X - T_eps(X))  -- the method
  unrolled    autograd through all inner Sinkhorn iterations of <P,C>
  regularized autograd through <P,C> + eps*KL(P||S), the control that should
              reproduce the envelope row

Everything else is held identical: same X0, same Y, same eps, L, learning rate,
inner iteration count, and projection seed. The three rows differ only in the
descent direction, so any visible difference is attributable to the gradient.

Uses the plain-torch fp32 solve from estimators.py for all three rows rather than
run.py's Triton slot_grad for the envelope row, so the arms differ only in the
gradient and not in the solver implementation. run.py remains the reference for
comparing SinkSLOT against SOT/EOT/SROT.

Result: the point clouds are visually indistinguishable at every checkpoint. A
cosine of 0.61 between the envelope and unrolled directions at step 50 does not
produce a flow anyone would call different -- the crescent forms at the same
rate, with the same interior structure and the same thin tail.

Exact W2^2 at N=1000, eps=0.01, L=100, 600 inner iterations:

    step        0        5       10       20       30       50
    env    1.0122   0.3540   0.1238   0.0154   0.0022   0.0005
    unr    1.0122   0.3533   0.1234   0.0152   0.0020   0.0003
    reg    1.0122   0.3540   0.1238   0.0154   0.0022   0.0005

The support builder is not bitwise deterministic across runs, so the last quoted
digit moves (step-30 unrolled comes out 0.0020 or 0.0021). The row ordering below
is stable; do not read the fourth decimal as exact.

Two things worth stating plainly. The control works exactly: the regularized row
reproduces the envelope row to four decimals at every checkpoint, which is the
entropic-term explanation confirmed on the flow itself rather than on gradient
vectors.

And the unrolled row is *marginally better* on W2^2 -- lower at every checkpoint,
and 0.0003 against 0.0005 at step 50. That is not a contradiction of the earlier
result and it is not evidence the envelope gradient is wrong: the envelope
gradient is the correct gradient of SLOT_eps, which is an entropically blurred
objective, while the unrolled arm descends <P,C>, which is closer to the
unregularized W2^2 this table measures. Optimising a slightly different objective
scores slightly better on that objective. The effect is small next to the ~2000x
reduction both achieve, and it comes at O(k) stored activations, but it is a real
and reproducible ordering and should not be reported as a tie.
"""
from __future__ import annotations

import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sinkslot.solver import _ot_1d_coo_batched, _ot_1d_coo_batched_cuda, sparse_sot_coo
from gradient_flow.along_flow import regularized_unrolled
from gradient_flow.config import L, LR, N, N_STEPS, STEPS
from gradient_flow.estimators import SEED, three_gradients
from gradient_flow.run import DATA_DIR, DEVICE, OUT_DIR, draw_samples, exact_ot_cost

EPS = 0.01
MODES = ["envelope", "unrolled", "regularized"]
ROW_LABELS = {
    "envelope": "envelope\n(ours)",
    "unrolled": "unrolled\n" + r"$\langle P,C\rangle$",
    "regularized": "unrolled\n" + r"$\langle P,C\rangle+\epsilon\mathrm{KL}$",
}


def run_flow(mode, X0, Y, a, n, steps, iters, eps, checkpoints):
    X = X0.clone()
    ot1d = _ot_1d_coo_batched_cuda if X0.is_cuda else _ot_1d_coo_batched
    out = {}
    for step in range(steps + 1):
        rows, cols, S = sparse_sot_coo(X, Y, a, a, L=L, seed=SEED, ot1d=ot1d)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()

        if mode == "regularized":
            g = regularized_unrolled(iters, X, Y, a, rows, cols, log_S, n, eps=eps)
        else:
            g_env, _, g_full, _, _ = three_gradients(
                iters, X, Y, a, rows, cols, log_S, n, n,
                unroll=(mode == "unrolled"), eps=eps)
            g = g_full if mode == "unrolled" else g_env

        if step in checkpoints:
            Xn = X.double().cpu()
            out[step] = (Xn, exact_ot_cost(Xn, Y.double().cpu()))
            print(f"  [{mode}] step {step:>3}  W2^2={out[step][1]:.4f}", flush=True)

        if step == steps:
            break
        X = (X - LR * n * g).detach().clone()
        del g, rows, cols, S, log_S
        if X.is_cuda:
            torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()

    if DEVICE != "cuda":
        print("gradient_flow/flow_qualitative.py: no CUDA GPU found, running on CPU "
              "(pure torch throughout -- much slower, same algorithm).")

    checkpoints = [s for s in STEPS if s <= args.steps]
    n = args.n
    rng = torch.Generator(device="cpu").manual_seed(1)
    X0 = draw_samples(DATA_DIR / "density_a.png", n, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", n, rng, device=DEVICE).float()
    a = torch.full((n,), 1.0 / n, dtype=torch.float32, device=DEVICE)

    X0n, Yn = X0.double().cpu(), Y.double().cpu()
    colors = ((10 * X0[:, 0]).cos() * (10 * X0[:, 1]).cos()).cpu()

    print(f"N={n}  L={L}  eps={EPS}  iters={args.iters}  lr={LR}  steps={args.steps}")
    results = {}
    for mode in MODES:
        for step, val in run_flow(mode, X0, Y, a, n, args.steps, args.iters,
                                  EPS, checkpoints).items():
            results[(mode, step)] = val

    _plot(results, checkpoints, X0n, Yn, colors, args.steps)


def _plot(results, checkpoints, X0n, Yn, colors, steps):
    pts = torch.cat([X0n, Yn], dim=0)
    lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
    pad = (hi - lo) * 0.06
    xlim = (float(lo[0] - pad[0]), float(hi[0] + pad[0]))
    ylim = (float(lo[1] - pad[1]), float(hi[1] + pad[1]))

    # Best (lowest) W2^2 per intermediate checkpoint, bolded -- run.py's convention.
    best = {}
    for s in checkpoints:
        if s in (0, steps):
            continue
        vals = {m: results[(m, s)][1] for m in MODES if (m, s) in results}
        if vals:
            best[s] = min(vals, key=vals.get)

    nr, nc = len(MODES), len(checkpoints)
    data_w, data_h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    panel_w = 1.9
    panel_h = panel_w * data_h / data_w
    # title_h reserves figure-inches for the two-line suptitle *and* the "step N"
    # column headers; too small and the second title line lands on the headers.
    label_h, title_h = 0.12, 1.05
    fig, axes = plt.subplots(
        nr, nc, squeeze=False,
        figsize=(panel_w * nc, (panel_h + label_h) * nr + title_h),
        gridspec_kw=dict(wspace=0.015, hspace=0.0,
                         top=1 - title_h / ((panel_h + label_h) * nr + title_h),
                         bottom=0.005, left=0.045, right=0.998),
    )
    for i, mode in enumerate(MODES):
        for j, step in enumerate(checkpoints):
            ax = axes[i][j]
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            Xn, w2 = results[(mode, step)]
            ax.scatter(Yn[:, 0], Yn[:, 1], s=3, c="0.82", zorder=1, linewidths=0)
            ax.scatter(Xn[:, 0], Xn[:, 1], s=3, c=colors, cmap="hsv", zorder=2,
                       linewidths=0)
            ax.set_xlabel(rf"$W_2^2$ = {w2:.4f}", fontsize=9, labelpad=2,
                          fontweight="bold" if best.get(step) == mode else "normal")
            if i == 0:
                ax.set_title(f"step {step}", fontsize=12)
            if j == 0:
                ax.set_ylabel(ROW_LABELS[mode], fontsize=9)

    fig.suptitle(
        "Same flow, three gradients: a 0.61 cosine gap between rows 1 and 2 "
        "leaves the transport visually indistinguishable\n"
        r"unrolled ends marginally lower in $W_2^2$; the $\epsilon\mathrm{KL}$ "
        "control row reproduces the envelope row to 4 decimals",
        fontsize=11.5, y=0.995, va="top")
    out = OUT_DIR / f"flow_qualitative_eps_{EPS:g}.pdf"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {out}\nwrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
