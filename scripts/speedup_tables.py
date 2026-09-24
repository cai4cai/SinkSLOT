"""Time/iterations/memory to reach a cost-gap threshold, from the merged speedup CSV.

A configuration is (dataset, d, method, tf32, eps, L or s); its metrics are the
means over seeds. It reaches a threshold T when its mean cost gap is <= T and
every seed's plan is feasible (|mass - 1| < 1e-3 and marginal violation < 1e-4).
Rows count whether or not they converged before max_iter. For each method and
threshold:

  runtime     min mean total_ms over the configurations that reach T,
  iterations  mean iters_run of that same configuration,
  memory      min mean gpu_memory_mb over the configurations that reach T.

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
        feasible = all(
            _f(r["mass"]) is not None and abs(_f(r["mass"]) - 1) < 1e-3
            and _f(r["marg_viol"]) is not None and _f(r["marg_viol"]) < 1e-4 for r in rows)
        configs[key] = dict(
            gap=st.mean(gaps), feasible=feasible, n_seeds=len(rows),
            total_ms=st.mean(_f(r["total_ms"]) for r in rows),
            iters=st.mean(_f(r["iters_run"]) for r in rows),
            mem=st.mean(_f(r["gpu_memory_mb"]) for r in rows),
            converged=sum(r["converged"] == "True" for r in rows),
        )
    return configs


def best(configs, dataset, d, method, tf32, threshold):
    ok = [(k, c) for k, c in configs.items()
          if k[:4] == (dataset, d, method, tf32) and c["feasible"] and c["gap"] <= threshold]
    if not ok:
        return None, None
    fastest = min(ok, key=lambda kc: kc[1]["total_ms"])
    leanest = min(ok, key=lambda kc: kc[1]["mem"])
    return fastest, leanest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    args = ap.parse_args()
    configs = load_configs(args.csv)
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
                      f"min_mem={cm['mem']:.0f} MB")


if __name__ == "__main__":
    main()
