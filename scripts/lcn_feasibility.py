"""Do LCN-Sinkhorn and the sparse Sinkhorn of the same paper reach the benchmark's
1% / 10% cost gaps?

For one slice of configs/speedup.py (same data, weights, eps grid and stop rule;
seed 0), runs sinkslot.bench.lcn_sinkhorn (--method lcn, a few landmark / neighbour counts)
or its sparse Sinkhorn (--method sparse, a few neighbour counts) over the eps grid, and writes one CSV row per run: cost gap vs the exact-OT
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
from sinkslot.bench.lcn_sinkhorn import (  # noqa: E402
    lcn_factors, lcn_plan_metrics, lcn_sinkhorn, sparse_kernel, sparse_plan_metrics, sparse_sinkhorn,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("lcn", "sparse", "lcn_ref", "sparse_ref"), default="lcn",
                    help="lcn / sparse: ports in sinkslot.bench.lcn_sinkhorn; lcn_ref / sparse_ref: the "
                         "authors' package (--lcn-path) with the settings of their experiments")
    ap.add_argument("--lcn-path", default="external/lcn")
    ap.add_argument("--sparse-config", choices=("kmeans_hier", "angular_lsh"), default="kmeans_hier",
                    help="sparse_ref neighbour method: their LCN config's kmeans_hier [10, 100] or their "
                         "sparse config's angular_lsh (130 clusters, 16 bands x 2 hashes)")
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

    if args.method in ("sparse", "sparse_ref"):
        args.landmarks = "0"
    if args.method.endswith("_ref"):
        from sinkslot.bench import lcn_reference as R
        lcn = R.import_lcn(str(Path(args.lcn_path).resolve()))
        sp_cfg = R.SPARSE_CONFIGS[args.sparse_config if args.method == "sparse_ref" else "kmeans_hier"]
        ny_cfg = R.NYSTROM_CONFIG if args.method == "lcn_ref" else None
        args.landmarks, args.neighbors = "0", "0"
    fields = ["method", "dataset", "d", "seed", "eps", "landmarks", "neighbors", "pairs", "cost_gap_pct",
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
                    if args.method.endswith("_ref"):
                        try:
                            f = R.cost_matrix(lcn, x, y, eps, ny_cfg, sp_cfg)
                        except (IndexError, RuntimeError) as e:   # e.g. no cluster holds both x and y points
                            print(f"eps={eps}: cost matrix failed: {type(e).__name__}: {e}", flush=True)
                            w.writerow(dict(method=args.method, dataset=args.dataset, d=args.d, seed=args.seed,
                                            eps=eps, landmarks=landmarks, neighbors=neighbors, failed=True))
                            continue
                    elif args.method == "lcn":
                        f = lcn_factors(x, y, eps, landmarks, neighbors, seed=args.seed)
                    else:
                        f = sparse_kernel(x, y, neighbors)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if args.method == "lcn_ref":
                        res = R.lcn_solve(lcn, f, a, b, eps, args.max_iter, tol)
                    elif args.method == "sparse_ref":
                        res = R.sparse_solve(lcn, f, a, b, eps, args.max_iter, tol)
                    else:
                        solve = lcn_sinkhorn if args.method == "lcn" else sparse_sinkhorn
                        res = solve(f, a, b, eps, args.max_iter, tol)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    row = dict(method=args.method, dataset=args.dataset, d=args.d, seed=args.seed, eps=eps,
                               landmarks=landmarks, neighbors=neighbors,
                               pairs=int(f.rows.numel()) if hasattr(f, "rows") else None,
                               iters=res.iters, converged=res.converged, failed=res.failed,
                               setup_s=t1 - t0, solve_s=t2 - t1)
                    if not res.failed:
                        if args.method == "lcn_ref":
                            pm = R.lcn_metrics(lcn, f, res, x, y, eps)
                        elif args.method == "sparse_ref":
                            pm = R.sparse_metrics(f, res, eps)
                        else:
                            pm = (lcn_plan_metrics(f, res, x, y) if args.method == "lcn"
                                  else sparse_plan_metrics(f, res, eps))
                        if "pairs" in pm:
                            row["pairs"] = pm["pairs"]
                        feas = plan_feasibility(pm["row_sums"], pm["col_sums"], a.double(), b.double())
                        row.update(cost_gap_pct=cost_gap(pm["plan_cost"], ref),
                                   marg_viol=feas["marg_viol"], marg_viol_l1=feas["marg_viol_l1"],
                                   negative_mass=pm["negative_mass"])
                    w.writerow(row)
                    fh.flush()
                    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
