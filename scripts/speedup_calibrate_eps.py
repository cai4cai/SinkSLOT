"""Cost gap vs eps for one slice, to place the speedup and scalability eps.

FlashSinkhorn alternating, strict fp32, one seed (--seed, default 0), stop rule 1e-5 * median(C),
over median(C) * geomspace(1e-4, 0.5, 20). median(C) comes from
configs/speedup.py's MEDIAN_C for seed 0 when the slice is there, else is computed here.
Prints one line per eps and writes a CSV. Rows that hit max_iter still report
their gap.

    python scripts/speedup_calibrate_eps.py --dataset 8gaussians --d 2 --out calib.csv
    python scripts/speedup_calibrate_eps.py --dataset gaussian --d 4 --max-iter 500000 --only 1,2 --out rerun.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import speedup  # noqa: E402
from scripts.speedup_prepare import lower_median_sq_cost  # noqa: E402
from sinkslot.bench.bench_forward import StopCfg, _sample_problem, bench_flashsinkhorn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--n", type=int, default=speedup.N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--only", default="", help="comma-separated indices of the eps grid to run (default: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    median = speedup.MEDIAN_C.get((args.dataset, args.d)) if args.seed == 0 else None
    if median is None:
        x, y, _, _ = _sample_problem(args.n, args.n, args.d, device, args.dataset, args.seed)
        median = lower_median_sq_cost(x, y)
        del x, y
    print(f"median(C) = {median!r}", flush=True)
    stop = StopCfg(mode="potential", max_iter=args.max_iter, tol=speedup.REL_TOL * median, check_every=5)
    fields = ["dataset", "d", "seed", "n", "median_c", "eps", "eps_over_median", "cost_gap_pct", "iters_run",
              "converged", "total_ms"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        only = {int(i) for i in args.only.split(",") if i}
        for i, factor in enumerate(np.geomspace(1e-4, 0.5, args.points)):
            if only and i not in only:
                continue
            eps = float(f"{median * factor:.6g}")
            r = bench_flashsinkhorn(args.n, args.n, args.d, eps, args.max_iter, device, warmup=0, rep=1,
                                    backend="alternating", allow_tf32=False, dataset=args.dataset,
                                    stop=stop, seed=args.seed)
            row = dict(dataset=args.dataset, d=args.d, seed=args.seed, n=args.n, median_c=median, eps=eps,
                       eps_over_median=float(factor),
                       cost_gap_pct=r.cost_gap_pct, iters_run=r.iters_run, converged=r.converged,
                       total_ms=r.total_ms)
            writer.writerow(row)
            fh.flush()
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
