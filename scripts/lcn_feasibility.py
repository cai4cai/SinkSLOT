"""Does LCN-Sinkhorn reach the benchmark's 1% / 10% cost gaps?

For one slice of configs/speedup.py (same data, weights, eps grid and stop rule;
seed 0), runs sinkslot.bench.lcn_sinkhorn over the eps grid and a few landmark /
neighbour counts, and writes one CSV row per run: cost gap vs the exact-OT
reference, marginal violation, negative plan mass, iterations, solve time, and
whether the approximate kernel broke down (a non-positive row or column sum).

    python scripts/lcn_feasibility.py --dataset 8gaussians --d 2 --out lcn_8g.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import speedup  # noqa: E402
from sinkslot.bench.bench_forward import (  # noqa: E402
    _cached_exact_ot_reference, _sample_problem, cost_gap, plan_feasibility,
)
from sinkslot.bench.lcn_sinkhorn import lcn_factors, lcn_plan_metrics, lcn_sinkhorn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--landmarks", default="50,200")
    ap.add_argument("--neighbors", default="16,64")
    ap.add_argument("--max-iter", type=int, default=50000)
    ap.add_argument("--eps-over-median", default="",
                    help="comma-separated eps / median(C) values to run instead of the slice's eps grid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    n = speedup.N
    key = (args.dataset, args.d)
    tol = speedup.REL_TOL * speedup.MEDIAN_C[key]
    eps_values = speedup.eps_grid(*speedup.EPS_CROSSINGS[key])
    if args.eps_over_median:
        eps_values = [float(r) * speedup.MEDIAN_C[key] for r in args.eps_over_median.split(",")]
    x, y, a, b = _sample_problem(n, n, args.d, device, args.dataset, args.seed)
    ref = _cached_exact_ot_reference(n, n, args.d, args.seed, x, y, a, b, dataset=args.dataset)

    fields = ["dataset", "d", "seed", "eps", "landmarks", "neighbors", "pairs", "cost_gap_pct",
              "marg_viol", "marg_viol_l1", "negative_mass", "iters", "converged", "failed",
              "setup_s", "solve_s"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for landmarks in map(int, args.landmarks.split(",")):
            for neighbors in map(int, args.neighbors.split(",")):
                for eps in eps_values:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    f = lcn_factors(x, y, eps, landmarks, neighbors, seed=args.seed)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    res = lcn_sinkhorn(f, a, b, eps, args.max_iter, tol)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    row = dict(dataset=args.dataset, d=args.d, seed=args.seed, eps=eps,
                               landmarks=landmarks, neighbors=neighbors, pairs=int(f.rows.numel()),
                               iters=res.iters, converged=res.converged, failed=res.failed,
                               setup_s=t1 - t0, solve_s=t2 - t1)
                    if not res.failed:
                        pm = lcn_plan_metrics(f, res, x, y)
                        feas = plan_feasibility(pm["row_sums"], pm["col_sums"], a.double(), b.double())
                        row.update(cost_gap_pct=cost_gap(pm["plan_cost"], ref),
                                   marg_viol=feas["marg_viol"], marg_viol_l1=feas["marg_viol_l1"],
                                   negative_mass=pm["negative_mass"])
                    w.writerow(row)
                    fh.flush()
                    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
