"""Plot convergence trajectories recorded by color_transfer.trajectory:
marginal violation and primal-dual gap, each in its own panel, all four
(method, backend) traces overlaid per panel -- rendered twice, once
against iteration count and once against wall-clock runtime.

    python -m color_transfer.plot_trajectory --input_dir DIR --output trajectories.png

--input_dir should contain trajectory_{method}[_{backend}].json files
written by color_transfer.trajectory (one per method/backend combination
you want plotted; missing files are skipped, not an error).

Writes two files derived from --output: trajectories_iterations.png and
trajectories_runtime.png. Both cover the same data, just against a
different x-axis -- iterations is the more meaningful axis for comparing
convergence RATE (trajectory.py now runs each checkpoint to genuine
convergence under a real --tol, so each method's curve ends at its own
convergence point), while runtime is the more meaningful axis for
comparing wall-clock cost, since iteration count alone hides large
per-iteration cost differences between methods (see the PR's TF32 finding:
FlashSinkhorn needs far fewer iterations than GeomLoss to reach the same
tolerance, but each of ITS iterations costs more wall-clock time).

Two panels, one per metric, with all four traces (SinkSLOT, FlashSinkhorn,
GeomLoss-online, GeomLoss-multiscale) overlaid per panel -- keeps each
metric on its own natural y-scale. trajectory.py does not record a
potential_change field (SinkSLOT's own internal "potential" stop mode has
a confirmed correctness bug in the forced-full-budget usage that field
would need -- see trajectory.py's own docstring), so this plot covers only
the two metrics that are actually recorded.

The marginal violation panel draws --tol as a horizontal reference line on
the iterations figure only (a fixed value on the y-axis, so it renders the
same regardless of x-axis choice -- drawn once to avoid a redundant legend
entry on both figures).

Log-scale y-axis in every panel. primal_dual_gap is abs(primal - dual)
(see main_pixel.py's _kl_gap_sparse/_kl_gap_dense_chunked and trajectory.py's
own docstring), not clamped to >= 0, so it is virtually always plottable --
the exactly-0.0/None skip below is a defensive no-op for the near-impossible
float coincidence, kept in case a metric is ever missing rather than tiny,
not because zeros are expected in normal operation. Each trace's final
(converged) point is marked with a star, since that is the point the
method actually stopped at (by marginal violation, in every case -- see
trajectory.py), not just the largest value plotted. The empty-panel
fallback below is a defensive path for the case where a metric genuinely
has nothing to plot (e.g. missing trajectory files), not something normal
runs are expected to hit anymore.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TRACES = [
    # (json filename stem, display label, color)
    ("sinkslot", "SinkSLOT", "#1f77b4"),
    ("flashsinkhorn", "FlashSinkhorn (alternating)", "#ff7f0e"),
    ("flashsinkhorn_symmetric", "FlashSinkhorn (symmetric)", "#d62728"),
    ("geomloss_online", "GeomLoss (online)", "#2ca02c"),
    ("geomloss_multiscale", "GeomLoss (multiscale)", "#9467bd"),
]

_METRICS = [
    ("marginal_viol", "Marginal violation"),
    ("primal_dual_gap", "Primal-dual gap"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True,
                    help="Base path; actual files written are "
                         "<stem>_iterations<ext> and <stem>_runtime<ext>.")
    return p.parse_args()


def load_trajectory(input_dir, stem):
    path = os.path.join(input_dir, f"trajectory_{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _render(traces, x_key, x_label, output_path):
    fig, axes = plt.subplots(len(_METRICS), 1, figsize=(9, 4 * len(_METRICS)), sharex=False)

    for ax, (metric_key, metric_title) in zip(axes, _METRICS):
        any_plotted = False
        for label, color, data in traces:
            checkpoints = data["checkpoints"]
            xs, ys = [], []
            for c in checkpoints:
                v = c.get(metric_key)
                if v is None or v == 0.0:  # see module docstring: not real near-zero values
                    continue
                xs.append(c[x_key])
                ys.append(v)
            if xs:
                any_plotted = True
                ax.plot(xs, ys, "-o", color=color, label=label, markersize=4, linewidth=1.5)
                if checkpoints[-1].get("converged"):
                    ax.plot(xs[-1], ys[-1], "*", color=color, markersize=14,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        if metric_key == "marginal_viol" and x_key == "iters":
            tol = traces[0][2].get("tol")
            if tol is not None:
                ax.axhline(tol, color="grey", linestyle="--", linewidth=1,
                           label=f"tol={tol:g}")
                any_plotted = True

        ax.set_yscale("log")
        ax.set_ylabel(metric_title)
        ax.set_xlabel(x_label)
        ax.grid(True, which="both", alpha=0.25)

        if any_plotted:
            ax.legend(fontsize=8, loc="upper right")
        else:
            # Defensive fallback: every trace was empty for this metric (e.g.
            # missing trajectory files, or every checkpoint's value was None).
            # Not expected to fire for primal_dual_gap anymore now that it's
            # abs(primal - dual) rather than a >= 0 clamp -- see module
            # docstring -- but kept so a genuinely empty panel is explained
            # rather than left as an unlabeled blank log axis.
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"No {metric_title.lower()} data to plot.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="dimgrey", wrap=True)

    pair = traces[0][2].get("pair")
    eps = traces[0][2].get("eps")
    tol = traces[0][2].get("tol")
    title = f"Convergence trajectories, eps={eps:g}" if eps is not None else "Convergence trajectories"
    if tol is not None:
        title += f", tol={tol:g}"
    if pair:
        title += f"  ({pair[0]} -> {pair[1]})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    traces = []
    for stem, label, color in _TRACES:
        data = load_trajectory(args.input_dir, stem)
        if data is not None:
            traces.append((label, color, data))
        else:
            print(f"skipping {stem}: no trajectory_{stem}.json in {args.input_dir}")

    if not traces:
        raise SystemExit(f"no trajectory_*.json files found in {args.input_dir}")

    stem, ext = os.path.splitext(args.output)
    _render(traces, "iters", "Iterations", f"{stem}_iterations{ext}")
    _render(traces, "time", "Runtime (s)", f"{stem}_runtime{ext}")


if __name__ == "__main__":
    main()
