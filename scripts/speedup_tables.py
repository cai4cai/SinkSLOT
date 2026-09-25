"""Time/iterations/memory to reach a cost-gap threshold, from the merged speedup CSV.

A configuration is (dataset, d, method, tf32, eps, L or s); its metrics are the
means over seeds. It reaches a threshold T when its mean cost gap is <= T,
whether or not its runs converged before max_iter. For each method and
threshold:

  runtime     min mean total_ms over the configurations that reach T,
  iterations  mean iters_run of that same configuration,
  memory      min mean peak_alloc_mb (PyTorch allocator peak over the timed
              solves) over the configurations that reach T,
  memfast     mean peak_alloc_mb of the fastest configuration.

Flags on a selected configuration: "M" if any seed hit max_iter, "V" if any
seed's marginal violation (L-infinity) exceeds 1e-6.

    python scripts/speedup_tables.py output/speedup_potential_final/forward_all.csv
"""

import argparse
import csv
import statistics as st
from collections import defaultdict

SLICES = [("half_moon", "2"), ("8gaussians", "2"), ("two_rings", "2"), ("gaussian", "3"), ("gaussian", "64")]
THRESHOLDS = [1.0, 5.0, 10.0]
METHODS = [
    ("flash_alternating", False, "FlashSinkhorn (alternating)"),
    ("flash_symmetric", False, "FlashSinkhorn (symmetric)"),
    ("flash_alternating", True, "FlashSinkhorn (alternating, TF32)"),
    ("flash_symmetric", True, "FlashSinkhorn (symmetric, TF32)"),
    ("geomloss_online", False, "GeomLoss"),
    ("srot", False, "SROT"),
    ("spar_sink", False, "Spar-Sink"),
    ("sinkslotcuda", False, "SinkSLOT (ours, alternating)"),
    ("sinkslotcuda_symmetric", False, "SinkSLOT (ours, symmetric)"),
]
REFERENCE = ("sinkslotcuda", False)
MARG_VIOL_FLAG = 1e-6


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_configs(path):
    groups = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("oom") in ("True", "true") or r.get("mean_ms") == "OOM":
                continue
            param = r["srot_slices"] if r["srot_slices"] not in ("", "N/A") else r["sample_size"]
            key = (r["dataset"], r["d"], r["method"], r["tf32"] in ("True", "true"), float(r["eps"]), param)
            groups[key].append(r)
    configs = {}
    for key, rows in groups.items():
        gaps = [_f(r["cost_gap_pct"]) for r in rows]
        if any(g is None for g in gaps):
            continue
        viol = [_f(r["marg_viol"]) for r in rows]
        flags = ("M" if any(r["hit_max_iters"] == "True" for r in rows) else "") + \
                ("V" if any(v is None or v > MARG_VIOL_FLAG for v in viol) else "")
        configs[key] = dict(
            gap=st.mean(gaps), flags=flags, n_seeds=len(rows),
            max_marg_viol=max((v for v in viol if v is not None), default=None),
            total_ms=st.mean(_f(r["total_ms"]) for r in rows),
            iters=st.mean(_f(r["iters_run"]) for r in rows),
            mem=st.mean(_f(r["peak_alloc_mb"]) for r in rows),
            converged=sum(r["converged"] == "True" for r in rows),
        )
    return configs


def best(configs, dataset, d, method, tf32, threshold):
    ok = [(k, c) for k, c in configs.items()
          if k[:4] == (dataset, d, method, tf32) and c["gap"] <= threshold]
    if not ok:
        return None, None
    fastest = min(ok, key=lambda kc: kc[1]["total_ms"])
    leanest = min(ok, key=lambda kc: kc[1]["mem"])
    return fastest, leanest


def _fmt_ms(t: float) -> str:
    if t >= 1000:
        return f"{t:,.0f}".replace(",", "{,}")
    return f"{t:.1f}" if t < 100 else f"{t:.0f}"


def _flags_tex(flags: str) -> str:
    marks = ("\\dagger" if "M" in flags else "") + ("\\ddagger" if "V" in flags else "")
    return f"$^{{{marks}}}$" if marks else ""


def latex_speedup_rows(configs, thresholds=(1.0, 10.0), wrap=lambda cell: cell) -> str:
    """Rows of Tables/speedup_potential.tex: runtime (ms) with SinkSLOT (alternating)'s
    speedup over each method; dagger/ddagger mark max_iter hits / marginal violation > 1e-6."""
    lines = []
    for i, (method, tf32, label) in enumerate(METHODS):
        ours = method.startswith("sinkslotcuda")
        name = f"\\textbf{{{label}}}" if ours else label
        cells = []
        for dataset, d in SLICES:
            for T in thresholds:
                ref, _ = best(configs, dataset, d, *REFERENCE, T)
                fastest, _ = best(configs, dataset, d, method, tf32, T)
                if fastest is None:
                    cells.append("---")
                    continue
                c = fastest[1]
                cell = _fmt_ms(c["total_ms"]) + _flags_tex(c["flags"])
                if (method, tf32) == REFERENCE:
                    cell = f"\\textbf{{{cell}}}"
                elif ref is not None:
                    cell += f"\\spd{{{c['total_ms'] / ref[1]['total_ms']:.1f}}}"
                cells.append(wrap(cell))
        row = ("\\rowcolor{LightGray}\n" if i % 2 else "") + name
        for j in range(0, len(cells), len(thresholds)):
            row += "\n & " + " & ".join(cells[j:j + len(thresholds)])
        lines.append(row + " \\\\")
    return "\n".join(lines)


def latex_threshold_rows(configs, kind: str, thresholds=(1.0, 10.0), wrap=lambda cell: cell) -> str:
    """Rows of Tables/convergence.tex (kind="iters": iterations of the fastest
    configuration), Tables/memory.tex (kind="mem": lowest mean peak memory, MB) or
    Tables/memory_fastest.tex (kind="memfast": peak memory of the fastest configuration, MB).
    The lowest value of each column is bold."""
    table = []  # [method][column] -> (value, flags) or None
    for method, tf32, _ in METHODS:
        row = []
        for dataset, d in SLICES:
            for T in thresholds:
                fastest, leanest = best(configs, dataset, d, method, tf32, T)
                if fastest is None:
                    row.append(None)
                    continue
                c = leanest[1] if kind == "mem" else fastest[1]
                row.append((c["iters"] if kind == "iters" else c["mem"], c["flags"]))
        table.append(row)

    def fmt(value):
        if kind != "iters" and value < 10:
            return f"{value:.1f}"
        return f"{value:,.0f}".replace(",", "{,}")

    lowest = []
    for j in range(len(table[0])):
        shown = [fmt(r[j][0]) for r in table if r[j] is not None]
        lowest.append(min(shown, key=lambda t: float(t.replace("{,}", ""))) if shown else None)

    lines = []
    for i, ((method, _, label), row) in enumerate(zip(METHODS, table)):
        name = f"\\textbf{{{label}}}" if method.startswith("sinkslotcuda") else label
        cells = []
        for j, entry in enumerate(row):
            if entry is None:
                cells.append("---")
                continue
            text = fmt(entry[0])
            if text == lowest[j]:
                text = f"\\textbf{{{text}}}"
            cells.append(wrap(text + _flags_tex(entry[1])))
        lines.append(("\\rowcolor{LightGray}\n" if i % 2 else "") + name + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--latex", choices=("speedup", "iters", "mem", "memfast"), nargs="?", const="speedup",
                    help="Print table rows: speedup (Tables/speedup_potential.tex), iters "
                         "(Tables/convergence.tex), mem (Tables/memory.tex) or memfast "
                         "(Tables/memory_fastest.tex).")
    ap.add_argument("--red", action="store_true", help="With --latex: wrap every cell in \\textcolor{red}.")
    args = ap.parse_args()
    configs = load_configs(args.csv)
    if args.latex:
        wrap = (lambda cell: f"\\textcolor{{red}}{{{cell}}}") if args.red else (lambda cell: cell)
        if args.latex == "speedup":
            print(latex_speedup_rows(configs, wrap=wrap))
        else:
            print(latex_threshold_rows(configs, args.latex, wrap=wrap))
        return
    for T in THRESHOLDS:
        print(f"\n=== cost gap <= {T:g}% ===")
        for dataset, d in SLICES:
            ref, _ = best(configs, dataset, d, *REFERENCE, T)
            ref_ms = ref[1]["total_ms"] if ref else None
            print(f"-- {dataset} d={d}")
            for method, tf32, label in METHODS:
                fastest, leanest = best(configs, dataset, d, method, tf32, T)
                if fastest is None:
                    print(f"   {label:34s} ---")
                    continue
                (k, c), (_, cm) = fastest, leanest
                sp = f"x{c['total_ms'] / ref_ms:.1f}" if ref_ms else "n/a"
                print(f"   {label:34s} {c['total_ms']:10.1f} ms {sp:>7s}  eps={k[4]:<9g} param={k[5]:<8s} "
                      f"gap={c['gap']:.3g}% iters={c['iters']:.0f} conv={c['converged']}/{c['n_seeds']} "
                      f"viol={c['max_marg_viol']:.1e} min_mem={cm['mem']:.0f} MB"
                      + (f"  FLAG {c['flags']}" if c["flags"] else "")
                      + (f"  mem-FLAG {cm['flags']}" if cm["flags"] else ""))


if __name__ == "__main__":
    main()
