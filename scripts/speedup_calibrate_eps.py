"""Cost gap vs eps for one slice, to place the speedup benchmark's eps grid.

FlashSinkhorn alternating, strict fp32, seed 0, the speedup config's stop rule,
over median(C) * geomspace(1e-4, 0.5, 20). Prints one line per eps and writes a
CSV. Rows that hit max_iter still report their gap.

    python scripts/speedup_calibrate_eps.py --dataset 8gaussians --d 2 --out calib.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import speedup  # noqa: E402
from sinkslot.bench.bench_forward import StopCfg, bench_flashsinkhorn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    median = speedup.MEDIAN_C[(args.dataset, args.d)]
    stop = StopCfg(mode="potential", max_iter=20000, tol=speedup.REL_TOL * median, check_every=5)
    device = torch.device("cuda")
    fields = ["dataset", "d", "eps", "eps_over_median", "cost_gap_pct", "iters_run", "converged", "total_ms"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for factor in np.geomspace(1e-4, 0.5, args.points):
            eps = float(f"{median * factor:.6g}")
            r = bench_flashsinkhorn(speedup.N, speedup.N, args.d, eps, 20000, device, warmup=0, rep=1,
                                    backend="alternating", allow_tf32=False, dataset=args.dataset,
                                    stop=stop, seed=0)
            row = dict(dataset=args.dataset, d=args.d, eps=eps, eps_over_median=float(factor),
                       cost_gap_pct=r.cost_gap_pct, iters_run=r.iters_run, converged=r.converged,
                       total_ms=r.total_ms)
            writer.writerow(row)
            fh.flush()
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
