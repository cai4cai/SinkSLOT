"""Plot potential-change convergence trajectories recorded by
color_transfer.trajectory_potential, against iterations and runtime.

    python -m color_transfer.plot_trajectory_potential --input_dir DIR --output trajectories_potential.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TRACES = [
    ("sinkslot", "SinkSLOT", "#1f77b4"),
    ("flashsinkhorn", "FlashSinkhorn (alternating)", "#ff7f0e"),
    ("flashsinkhorn_symmetric", "FlashSinkhorn (symmetric)", "#d62728"),
    ("geomloss", "GeomLoss (online)", "#2ca02c"),
    ("geomloss_multiscale", "GeomLoss (multiscale)", "#9467bd"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def load(input_dir, stem):
    path = os.path.join(input_dir, f"trajectory_potential_{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def render(traces, x_key, x_label, y_key, y_label, output_path, y_log=True, show_tol=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, data in traces:
        checkpoints = data["checkpoints"]
        xs, ys = [], []
        for c in checkpoints:
            v = c.get(y_key)
            if v is None or (y_log and v == 0.0):
                continue
            xs.append(c[x_key])
            ys.append(v)
        if xs:
            ax.plot(xs, ys, "-o", color=color, label=label, markersize=4, linewidth=1.5)
            if checkpoints[-1].get("converged"):
                ax.plot(xs[-1], ys[-1], "*", color=color, markersize=14,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    tol = traces[0][2].get("tol")
    if show_tol and tol is not None:
        ax.axhline(tol, color="grey", linestyle="--", linewidth=1, label=f"tol={tol:g}")

    if y_log:
        ax.set_yscale("log")
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")

    pair = traces[0][2].get("pair")
    eps = traces[0][2].get("eps")
    title = f"eps={eps:g}, tol={tol:g}"
    if pair:
        title += f"  ({pair[0]} -> {pair[1]})"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    traces = []
    for stem, label, color in _TRACES:
        data = load(args.input_dir, stem)
        if data is not None:
            traces.append((label, color, data))
        else:
            print(f"skipping {stem}: no trajectory_potential_{stem}.json in {args.input_dir}")
    if not traces:
        raise SystemExit(f"no trajectory_potential_*.json files found in {args.input_dir}")

    stem, ext = os.path.splitext(args.output)
    render(traces, "iters", "Iterations", "potential_change", "Potential change: max(|Δf|, |Δg|)",
           f"{stem}_iterations{ext}")
    render(traces, "time", "Runtime (s)", "potential_change", "Potential change: max(|Δf|, |Δg|)",
           f"{stem}_runtime{ext}")
    render(traces, "iters", "Iterations", "cost", "Entropic transport cost: <a,f> + <b,g>",
           f"{stem}_cost{ext}", y_log=False, show_tol=False)


if __name__ == "__main__":
    main()
