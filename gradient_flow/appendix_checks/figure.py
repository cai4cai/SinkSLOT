"""One figure summarising the envelope-vs-complete gradient result.

    python -m gradient_flow.appendix_checks.figure

Assembles six panels from the runs already on disk, so it is cheap to re-render
and always consistent with the numbers in the commit messages:

  (a) gradient norms along the flow -- the two estimators separating
  (b) cosine along the flow, with the <P,C>+eps*KL control arm flat at 1
  (c) the control against truncation: cosine at step 50 vs inner iterations
  (d) cosine along the flow at four eps
  (e) final cosine over the (eps, L) grid
  (f) same cell under five projection seeds, at the noisiest and cleanest L

Inputs, all written by the scripts that produced them:
  outputs/sweep_along_flow.json    <- gradient_flow.sweep_along_flow
  outputs/projection_noise.json    <- gradient_flow.projection_noise
  outputs/truncation_check.json    <- computed here on first run and cached,
                                      since it was never persisted by the
                                      scripts above (it re-solves one fixed
                                      configuration at several iteration counts,
                                      which nothing else needed to store)

Writes both PDF (for papers) and PNG (for sending).
"""
from __future__ import annotations

import argparse
import json

import torch

from gradient_flow.run import OUT_DIR

SWEEP = OUT_DIR / "sweep_along_flow.json"
NOISE = OUT_DIR / "projection_noise.json"
TRUNC = OUT_DIR / "truncation_check.json"

CELL_EPS, CELL_L = 0.01, 100
K_GRID = [25, 50, 100, 200, 400, 800]

C_ENV, C_FULL, C_REG = "#1f77b4", "#d62728", "#7d5ba6"


def truncation_check(steps, iters, n_):
    """Walk the flow to its last step, then re-solve there at several k.

    Answers whether the env-vs-full gap is just an under-converged solve: if it
    were, the cosine would climb toward 1 as k grows and the marginal violation
    falls. Cached to TRUNC because it is the one control not saved by any of the
    sweep scripts.
    """
    from sinkslot.solver import _ot_1d_coo_batched, _ot_1d_coo_batched_cuda, sot_coo
    from gradient_flow.along_flow import _cosine
    from gradient_flow.estimators import SEED, three_gradients
    from gradient_flow.run import DATA_DIR, DEVICE, draw_samples

    rng = torch.Generator(device="cpu").manual_seed(1)
    X = draw_samples(DATA_DIR / "density_a.png", n_, rng, device=DEVICE).float()
    Y = draw_samples(DATA_DIR / "density_b.png", n_, rng, device=DEVICE).float()
    a = torch.full((n_,), 1.0 / n_, dtype=torch.float32, device=DEVICE)
    ot1d = _ot_1d_coo_batched_cuda if DEVICE == "cuda" else _ot_1d_coo_batched

    from gradient_flow.config import LR
    for _ in range(steps):
        rows, cols, S = sot_coo(X, Y, a, a, L=CELL_L, seed=SEED, ot1d=ot1d)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        g, _, _, _, _ = three_gradients(iters, X, Y, a, rows, cols, log_S, n_, n_,
                                     unroll=False, eps=CELL_EPS)
        X = (X - LR * n_ * g).detach().clone()

    rows, cols, S = sot_coo(X, Y, a, a, L=CELL_L, seed=SEED, ot1d=ot1d)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    out = {"k": K_GRID, "cos": [], "viol": []}
    for k in K_GRID:
        ge, _, gf, viol, _ = three_gradients(k, X, Y, a, rows, cols, log_S, n_, n_,
                                          eps=CELL_EPS)
        out["cos"].append(_cosine(ge, gf))
        out["viol"].append(viol)
        print(f"  k={k:>4}  cos={out['cos'][-1]:.6f}  viol={viol:.2e}", flush=True)
        del ge, gf
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recompute-truncation", action="store_true")
    args = p.parse_args()

    for f in (SWEEP, NOISE):
        if not f.exists():
            raise SystemExit(f"missing {f}; run the script that writes it first "
                             f"(see this module's docstring)")
    sweep = json.loads(SWEEP.read_text())
    noise = json.loads(NOISE.read_text())

    if args.recompute_truncation or not TRUNC.exists():
        if not torch.cuda.is_available():
            print("gradient_flow/figure.py: no CUDA GPU found, running the truncation "
                  "check on CPU (pure torch throughout -- much slower, same algorithm).")
        print(f"computing truncation check (not cached at {TRUNC})")
        TRUNC.write_text(json.dumps(
            truncation_check(sweep["steps"], sweep["iters"], sweep["n"])))
    trunc = json.loads(TRUNC.read_text())

    _plot(sweep, noise, trunc)


def _plot(sweep, noise, trunc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sweep["steps"]
    xs = list(range(steps + 1))
    cell = sweep["cells"][f"{CELL_EPS}/{CELL_L}"]

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7.6))
    fig.subplots_adjust(wspace=0.30, hspace=0.34, top=0.86)
    (a, b, c), (d, e, f) = axes

    # (a) norms -------------------------------------------------------------
    a.plot(xs, cell["norm_env"], "-", lw=1.9, color=C_ENV,
           label=r"analytical (envelope) $\|g_{\rm env}\|$")
    a.plot(xs, cell["norm_full"], "-.", lw=1.9, color=C_FULL,
           label=r"complete (unrolled) $\|g_{\rm full}\|$")
    a.set_yscale("log")
    a.set_xlabel("gradient step"); a.set_ylabel("gradient norm")
    a.legend(frameon=False, fontsize=8.5)
    a.set_title("(a) norms separate as the flow converges", fontsize=10.5, loc="left")

    # (b) cosine + control ---------------------------------------------------
    b.plot(xs, cell["cos"], "-", lw=2.0, color=C_FULL,
           label=r"vs complete $\langle P,C\rangle$")
    if not bool(torch.tensor(cell["cos_reg"]).isnan().all()):
        b.plot(xs, cell["cos_reg"], "--", lw=1.8, color=C_REG,
               label=r"vs complete $\langle P,C\rangle+\epsilon\mathrm{KL}$")
    b.axhline(1.0, color="0.65", lw=0.8, ls=":")
    b.set_ylim(0.5, 1.03)
    b.set_xlabel("gradient step"); b.set_ylabel(r"cosine vs $g_{\rm env}$")
    b.legend(frameon=False, fontsize=8.5, loc="lower left")
    b.set_title("(b) the whole gap is the entropic term", fontsize=10.5, loc="left")

    # (c) truncation control -------------------------------------------------
    c.plot(trunc["k"], trunc["cos"], "o-", lw=1.8, ms=5, color="#2a9d5c")
    c.set_xscale("log")
    # Default log ticks collide here (the k grid is dense in decades), so label
    # exactly the k values that were run and drop the minor labels.
    c.set_xticks(trunc["k"], [str(k) for k in trunc["k"]], fontsize=8.5)
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_ylim(0, 1.03)
    c.axhline(1.0, color="0.65", lw=0.8, ls=":")
    c.set_xlabel("inner Sinkhorn iterations at step %d" % sweep["steps"])
    c.set_ylabel(r"cos$(g_{\rm env},g_{\rm full})$")
    c2 = c.twinx()
    c2.plot(trunc["k"], trunc["viol"], "s--", lw=1.2, ms=4, color="0.55")
    c2.set_yscale("log")
    c2.set_ylabel(r"max $|a/r-1|$", color="0.45", fontsize=9)
    c2.tick_params(axis="y", colors="0.45", labelsize=8)
    c.set_title("(c) not truncation: flat while the solve converges",
                fontsize=10.5, loc="left")

    # (d) varying eps --------------------------------------------------------
    cmap = plt.get_cmap("viridis")
    for i, ev in enumerate(sweep["eps"]):
        d.plot(xs, sweep["cells"][f"{ev}/{CELL_L}"]["cos"], "-", lw=1.6,
               color=cmap(i / max(len(sweep["eps"]) - 1, 1)), label=rf"$\epsilon={ev:g}$")
    d.set_ylim(top=1.03)
    d.set_xlabel("gradient step"); d.set_ylabel(r"cos$(g_{\rm env},g_{\rm full})$")
    d.legend(frameon=False, fontsize=8.5, loc="lower left")
    d.set_title(rf"(d) every $\epsilon$ at $L={CELL_L}$", fontsize=10.5, loc="left")

    # (e) heatmap ------------------------------------------------------------
    M = torch.tensor([[sweep["cells"][f"{ev}/{lv}"]["cos"][-1] for lv in sweep["L"]]
                      for ev in sweep["eps"]])
    im = e.imshow(M, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    e.set_xticks(range(len(sweep["L"])), [str(v) for v in sweep["L"]])
    e.set_yticks(range(len(sweep["eps"])), [f"{v:g}" for v in sweep["eps"]])
    e.set_xlabel("$L$ (projections)"); e.set_ylabel(r"$\epsilon$")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = float(M[i, j])
            e.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                   color="white" if val < 0.6 else "black")
    fig.colorbar(im, ax=e, fraction=0.046)
    e.set_title(f"(e) cosine at step {steps} over the grid", fontsize=10.5, loc="left")

    # (f) projection noise ---------------------------------------------------
    nsteps = noise["steps"]
    nxs = list(range(nsteps + 1))
    for lv, col in ((noise["L"][0], "#e07b39"), (noise["L"][-1], "#2b3a67")):
        for si, sd in enumerate(noise["seeds"]):
            f.plot(nxs, noise["runs"][f"{lv}/{sd}"], "-", lw=1.0, color=col,
                   alpha=0.85, label=f"$L={lv}$" if si == 0 else None)
    f.set_ylim(top=1.03)
    f.set_xlabel("gradient step"); f.set_ylabel(r"cos$(g_{\rm env},g_{\rm full})$")
    f.legend(frameon=False, fontsize=8.5, loc="lower left")
    f.set_title(f"(f) {len(noise['seeds'])} projection seeds: small-$L$ wiggle is noise",
                fontsize=10.5, loc="left")

    fig.suptitle(
        "The envelope gradient and the complete gradient diverge along the flow "
        "— and the difference is the entropic term\n"
        rf"blob$\to$crescent, $N={sweep['n']}$, {sweep['iters']} inner Sinkhorn "
        rf"iterations; panels (a)–(c) at $\epsilon={CELL_EPS}$, $L={CELL_L}$",
        fontsize=12.5, y=0.985)

    for ax in (a, b, c, d, f):
        ax.grid(alpha=0.28)

    pdf, png = OUT_DIR / "gradient_summary.pdf", OUT_DIR / "gradient_summary.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
