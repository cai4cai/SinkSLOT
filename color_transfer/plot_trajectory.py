"""Plot convergence trajectories recorded by color_transfer.trajectory:
marginal violation and primal-dual gap, each in its own panel, all four
(method, backend) traces overlaid per panel, against cumulative wall-clock
runtime.

    python -m color_transfer.plot_trajectory --input_dir DIR --output PATH

--input_dir should contain trajectory_{method}[_{backend}].json files
written by color_transfer.trajectory (one per method/backend combination
you want plotted; missing files are skipped, not an error).

Two panels, one per metric, with all four traces (SinkSLOT, FlashSinkhorn,
GeomLoss-online, GeomLoss-multiscale) overlaid per panel -- keeps each
metric on its own natural y-scale. trajectory.py does not record a
potential_change field (SinkSLOT's own internal "potential" stop mode has
a confirmed correctness bug in the forced-full-budget usage that field
would need -- see trajectory.py's own docstring), so this plot covers only
the two metrics that are actually recorded.

Log-scale y-axis in every panel: checkpoints where a metric reads exactly
0.0 (trajectory.py's own documented max(gap, 0.0) clamp, or an undefined
first-checkpoint potential_change) are dropped from that specific line
before plotting, per trajectory.py's own module docstring -- not treated
as real near-zero values.

SinkSLOT's setup_time (sparse-support build, paid once, not per
checkpoint) is shaded as its own region from t=0 to t=setup_time, on every
panel, since every checkpoint's own `time` already has it folded in (see
trajectory.py's own docstring) -- without the shading, that constant
offset would be invisible in the plot even though it is really there.
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
    ("flashsinkhorn", "FlashSinkhorn", "#ff7f0e"),
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
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def load_trajectory(input_dir, stem):
    path = os.path.join(input_dir, f"trajectory_{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


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

    fig, axes = plt.subplots(len(_METRICS), 1, figsize=(9, 4 * len(_METRICS)), sharex=False)

    for ax, (metric_key, metric_title) in zip(axes, _METRICS):
        for label, color, data in traces:
            checkpoints = data["checkpoints"]
            xs, ys = [], []
            for c in checkpoints:
                v = c.get(metric_key)
                if v is None or v == 0.0:  # see module docstring: not real near-zero values
                    continue
                xs.append(c["time"])
                ys.append(v)
            if xs:
                ax.plot(xs, ys, "-o", color=color, label=label, markersize=4, linewidth=1.5)

            setup_time = data.get("setup_time", 0.0)
            if setup_time and setup_time > 0:
                ax.axvspan(0, setup_time, color=color, alpha=0.12)

        ax.set_yscale("log")
        ax.set_ylabel(metric_title)
        ax.set_xlabel("Runtime (s)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    pair = traces[0][2].get("pair")
    eps = traces[0][2].get("eps")
    title = f"Convergence trajectories, eps={eps:g}" if eps is not None else "Convergence trajectories"
    if pair:
        title += f"  ({pair[0]} -> {pair[1]})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
