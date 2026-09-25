"""Cost gap vs runtime (Pareto) for FlashSinkhorn and SinkSLOT, from the merged speedup CSV.

One point per (method, L, eps): mean total_ms (setup + solve) and mean cost gap
over seeds. Colour = method (and L), marker = eps index within the dataset's grid
(1 = smallest). Writes pareto.pdf (the four low-dimensional datasets, 2x2) and
pareto_gauss64.pdf.

    python scripts/plot_pareto.py output/speedup_potential_final/forward_all.csv OUT_DIR
"""

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SERIES = [  # (method, L, label, colour): Okabe-Ito
    ("flash_alternating", None, "FlashSinkhorn", "#0072B2"),
    ("sinkslotcuda", "100", "SinkSLOT (ours), L=100", "#E69F00"),
    ("sinkslotcuda", "1000", "SinkSLOT (ours), L=1000", "#009E73"),
    ("sinkslotcuda", "5000", "SinkSLOT (ours), L=5000", "#CC79A7"),
]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "h", "X"]
LOWDIM = [("half_moon", "2", "Half-moons, d=2"), ("8gaussians", "2", "8-Gaussians, d=2"),
          ("two_rings", "2", "Two-rings, d=2"), ("gaussian", "3", "Gaussian, d=3")]
FONT = 9


def load(path):
    groups = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["tf32"] != "False" or r["mean_ms"] == "OOM" or r["cost_gap_pct"] in ("", "N/A"):
                continue
            L = r["srot_slices"] if r["srot_slices"] not in ("", "N/A") else None
            groups[(r["dataset"], r["d"], r["method"], L, float(r["eps"]))].append(r)
    return {k: (st.mean(float(r["total_ms"]) for r in v), st.mean(float(r["cost_gap_pct"]) for r in v))
            for k, v in groups.items()}


def panel(ax, data, dataset, d, title):
    eps_grid = sorted({k[4] for k in data if k[:2] == (dataset, d)})
    for method, L, label, colour in SERIES:
        pts = sorted((k[4], v) for k, v in data.items() if k[:4] == (dataset, d, method, L))
        pts = [(e, t, g) for e, (t, g) in pts if g > 0]  # log axis: gaps <= 0 are not drawn
        if not pts:
            continue
        ax.plot([t for _, t, _ in pts], [g for _, _, g in pts], "-", color=colour, lw=1.2, zorder=2)
        for e, t, g in pts:
            ax.plot(t, g, MARKERS[eps_grid.index(e)], color=colour, ms=4.5, mec="white", mew=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=FONT + 1, loc="left", pad=3)
    ax.grid(True, which="major", color="0.85", lw=0.6)
    ax.grid(True, which="minor", color="0.93", lw=0.4)
    ax.tick_params(labelsize=FONT - 1, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _method_handles():
    return [Line2D([], [], color=c, lw=1.5, label=lab) for _, _, lab, c in SERIES]


def _eps_handles():
    label = Line2D([], [], ls="", marker="", label=r"$\varepsilon$ index (1 = smallest):")
    return [label] + [Line2D([], [], ls="", marker=m, color="0.3", ms=4.5, label=f"{i + 1}")
                      for i, m in enumerate(MARKERS)]


def main():
    data = load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
    for ax, (dataset, d, title) in zip(axes.flat, LOWDIM):
        panel(ax, data, dataset, d, title)
    for ax in axes[1]:
        ax.set_xlabel("time (ms)", fontsize=FONT)
    for ax in axes[:, 0]:
        ax.set_ylabel("cost gap (%)", fontsize=FONT)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.6)
    fig.legend(handles=_method_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=4,
               fontsize=FONT - 1, frameon=False, handlelength=1.6, columnspacing=1.2)
    fig.legend(handles=_eps_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=11,
               fontsize=FONT - 1, frameon=False, title=None, handletextpad=0.2, columnspacing=0.9)
    fig.savefig(out / "pareto.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "pareto.png", dpi=150, bbox_inches="tight", pad_inches=0.02)

    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    panel(ax, data, "gaussian", "64", "Gaussian, d=64")
    ax.set_xlabel("time (ms)", fontsize=FONT)
    ax.set_ylabel("cost gap (%)", fontsize=FONT)
    ax.legend(handles=_method_handles(), loc="lower left", fontsize=FONT - 2, framealpha=0.9,
              handlelength=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=_eps_handles()[1:], loc="upper center", bbox_to_anchor=(0.5, 0.085), ncol=10,
               fontsize=FONT - 2, frameon=False, handletextpad=0.05, columnspacing=0.45,
               title=r"$\varepsilon$ index (1 = smallest)", title_fontsize=FONT - 2)
    fig.savefig(out / "pareto_gauss64.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "pareto_gauss64.png", dpi=150, bbox_inches="tight", pad_inches=0.02)

if __name__ == "__main__":
    main()
