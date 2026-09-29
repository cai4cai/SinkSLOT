"""Runtime or peak memory against N and d, from the scalability CSVs (configs/scalability.py).

Three panels: Gaussian d=3 against N, Gaussian d=64 against N, and N=10,000
against d. TF32 runs and the symmetric variants are left out unless --tf32 / --symmetric are given. One line per (method, L or s): mean over seeds of total_ms (setup +
solve) or peak_alloc_mb, with standard-error bars. Colour = method, line style =
L (SinkSLOT, SROT) or the multiple k of s0(N) (Spar-Sink). Hollow markers: at
least one seed hit max_iter. Out-of-memory points are not drawn.

    python scripts/plot_scalability.py OUT.pdf --metric time CSV [CSV ...]
    python scripts/plot_scalability.py OUT.pdf --metric memory CSV [CSV ...]
"""

import argparse
import csv
import math
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METHODS = [  # (method, tf32, label, colour, marker)
    ("sinkslotcuda", False, "SinkSLOT (ours, alternating)", "#E69F00", "o"),
    ("sinkslotcuda_symmetric", False, "SinkSLOT (ours, symmetric)", "#8C510A", "o"),
    ("flash_alternating", False, "FlashSinkhorn (alternating)", "#0072B2", "D"),
    ("flash_symmetric", False, "FlashSinkhorn (symmetric)", "#56B4E9", "D"),
    ("flash_alternating", True, "FlashSinkhorn (alternating, TF32)", "#009E73", "d"),
    ("flash_symmetric", True, "FlashSinkhorn (symmetric, TF32)", "#7FC97F", "d"),
    ("geomloss_online", False, "GeomLoss", "#000000", "s"),
    ("srot", False, "SROT", "#D55E00", "^"),
    ("spar_sink", False, "Spar-Sink", "#CC79A7", "v"),
]
STYLES = [":", "--", "-"]  # smallest to largest L, or k
PARAM_LABELS = ["L=100 / k=4", "L=1000 / k=16", "L=5000 / k=64"]
FONT = 9


def load(paths):
    rows = {}
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["method"] not in {m for m, *_ in METHODS}:
                    continue
                key = (r["method"], r["tf32"], r["n"], r["d"], r["srot_slices"], r["sample_size"], r["seed"])
                rows[key] = r
    return list(rows.values())


def _param(r):
    """Index of L or k within its grid (0, 1, 2), or None for methods without one."""
    if r["method"] == "spar_sink":
        return int(r["sample_size"])
    if r["srot_slices"] not in ("", "N/A"):
        return int(r["srot_slices"])
    return None


def series(rows, metric):
    """(method, tf32, param rank) -> {(n, d): (mean, se, any max_iter hit)}."""
    col = "total_ms" if metric == "time" else "peak_alloc_mb"
    groups = defaultdict(list)
    for r in rows:
        tf32 = r["tf32"] in ("True", "true")
        groups[(r["method"], tf32, int(r["n"]), int(r["d"]), _param(r))].append(r)
    # rank L or s within each (method, n, d); s depends on N, so rank per point
    params = defaultdict(set)
    for (m, tf32, n, d, p) in groups:
        if p is not None:
            params[(m, n, d)].add(p)
    out = defaultdict(dict)
    for (m, tf32, n, d, p), rs in groups.items():
        ok = [r for r in rs if r["oom"] not in ("True", "true") and r[col] not in ("", "N/A", "OOM")]
        if not ok:
            continue
        vals = [float(r[col]) for r in ok]
        se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        rank = sorted(params[(m, n, d)]).index(p) if p is not None else None
        out[(m, tf32, rank)][(n, d)] = (st.mean(vals), se, any(r["hit_max_iters"] == "True" for r in ok))
    return out


def panel(ax, data, xs_key, fixed, title, methods=METHODS):
    for m, tf32, _, colour, marker in methods:
        for (mm, tt, rank), pts in data.items():
            if (mm, tt) != (m, tf32):
                continue
            sel = sorted((xs_key(n, d), v) for (n, d), v in pts.items() if fixed(n, d))
            if not sel:
                continue
            ls = "-" if rank is None else STYLES[rank]
            x = [p[0] for p in sel]
            y = [p[1][0] for p in sel]
            ax.errorbar(x, y, yerr=[p[1][1] for p in sel], color=colour, ls=ls, lw=1.1, capsize=1.5,
                        elinewidth=0.7, zorder=2)
            for xi, (yi, _, capped) in sel:
                ax.plot(xi, yi, marker, ms=3.5, color=colour, mfc="white" if capped else colour, mew=0.8, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=FONT + 1, loc="left", pad=3)
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.tick_params(labelsize=FONT - 1, length=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--metric", choices=("time", "memory"), default="time")
    ap.add_argument("--tf32", action="store_true", help="Also plot the TF32 FlashSinkhorn runs.")
    ap.add_argument("--symmetric", action="store_true",
                    help="Also plot SinkSLOT (symmetric) and FlashSinkhorn (symmetric).")
    args = ap.parse_args()
    methods = [m for m in METHODS
               if (args.tf32 or not m[1]) and (args.symmetric or "symmetric" not in m[0])]
    data = series(load(args.csvs), args.metric)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    panel(axes[0], data, lambda n, d: n, lambda n, d: d == 3, "Gaussian, $d=3$", methods)
    panel(axes[1], data, lambda n, d: n, lambda n, d: d == 64, "Gaussian, $d=64$", methods)
    panel(axes[2], data, lambda n, d: d, lambda n, d: n == 10000, "Gaussian, $N=10^4$", methods)
    axes[0].set_xlabel("$N=M$", fontsize=FONT)
    axes[1].set_xlabel("$N=M$", fontsize=FONT)
    axes[2].set_xlabel("$d$", fontsize=FONT)
    axes[0].set_ylabel("time (ms)" if args.metric == "time" else "peak memory (MB)", fontsize=FONT)
    for ax in axes[:2]:
        ax.set_xticks([5000, 10000, 20000, 50000])
        ax.set_xticklabels(["5k", "10k", "20k", "50k"])
        ax.minorticks_off()
    axes[2].set_xticks([4, 16, 64, 256, 1024])
    axes[2].set_xticklabels(["4", "16", "64", "256", "1024"])
    axes[2].minorticks_off()
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.8)
    methods = [Line2D([], [], color=c, marker=mk, ms=3.5, lw=1.2, label=lab) for _, _, lab, c, mk in methods]
    params = [Line2D([], [], color="0.3", ls=s, lw=1.1, label=lab) for s, lab in zip(STYLES, PARAM_LABELS)]
    params.append(Line2D([], [], ls="", marker="o", color="0.3", mfc="white", ms=3.5, label="hit max_iter"))
    fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(0.5, 0.225), ncol=3, fontsize=FONT - 1.5,
               frameon=False, handlelength=1.8, columnspacing=1.0, labelspacing=0.3)
    fig.legend(handles=params, loc="upper center", bbox_to_anchor=(0.5, 0.225 - 0.052 * -(-len(methods) // 3)), ncol=4, fontsize=FONT - 1.5,
               frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=170, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
