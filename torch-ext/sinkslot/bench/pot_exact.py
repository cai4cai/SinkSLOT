"""Exact OT with POT's network simplex on the benchmark instances (CPU, float64).

    python -m sinkslot.bench.pot_exact --datasets half_moon:2 gaussian:64 --n 10000 --seeds 0 1 2 3 4 \
        --kinds dense_1 dense_24 --out pot_exact.csv

Kinds:
  dense_T  ot.dist then ot.emd2 on the dense cost matrix with T OpenMP threads;
           setup_ms is the cost matrix, mean_ms the solve, total_ms their sum.
  lazy     ot.emd2_lazy, which computes the cost on the fly (O(N + M) memory).

Instances come from bench_forward._sample_problem. They are drawn on --device
(default: CUDA when available), so they match the GPU benchmark only when drawn
on CUDA. Each solve runs in its own process; peak_rss_mb is that process's peak
resident memory (None where the resource module is unavailable, e.g. Windows).
"""

import argparse
import csv
import multiprocessing as mp
import queue as queue_mod
import time

import torch


def _solve(kind, xs, xt, a, b, queue):
    import ot

    setup_s = 0.0
    if kind == "lazy":
        t0 = time.perf_counter()
        cost, log = ot.emd2_lazy(xs, xt, a, b, metric="sqeuclidean", numItermax=100_000_000, log=True,
                                 return_matrix=False)
        solve_s = time.perf_counter() - t0
    else:
        threads = int(kind.split("_")[1])
        t0 = time.perf_counter()
        M = ot.dist(xs, xt, metric="sqeuclidean")
        setup_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cost, log = ot.emd2(a, b, M, numItermax=100_000_000, log=True, numThreads=threads)
        solve_s = time.perf_counter() - t0
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        peak = None
    queue.put(dict(setup_ms=1e3 * setup_s, mean_ms=1e3 * solve_s, total_ms=1e3 * (setup_s + solve_s),
                   cost=float(cost), result_code=int(log.get("result_code", -1)),
                   warning=str(log.get("warning")), peak_rss_mb=peak))


def _result(queue, proc, label):
    """The worker's result; raises if the worker exits without one."""
    while True:
        try:
            return queue.get(timeout=5)
        except queue_mod.Empty:
            if not proc.is_alive():
                raise RuntimeError(f"POT worker for {label} exited with code {proc.exitcode} and no result")


def run(datasets, n, seeds, kinds, device):
    """One row per (dataset, seed, kind); `datasets` is a list of (name, d)."""
    from sinkslot.bench.bench_forward import _sample_problem

    ctx = mp.get_context("spawn")
    rows = []
    for dataset, d in datasets:
        for seed in seeds:
            x, y, a, b = _sample_problem(n, n, d, device, dataset, seed)
            xs, xt = x.double().cpu().numpy(), y.double().cpu().numpy()
            ad, bd = a.double().cpu().numpy(), b.double().cpu().numpy()
            ad, bd = ad / ad.sum(), bd / bd.sum()
            for kind in kinds:
                queue = ctx.Queue()
                proc = ctx.Process(target=_solve, args=(kind, xs, xt, ad, bd, queue))
                proc.start()
                row = _result(queue, proc, f"{dataset} seed {seed} {kind}")
                proc.join()
                row.update(method="pot_exact", kind=kind, dataset=dataset, n=n, m=n, d=d, seed=seed,
                           device=str(device), converged=row["result_code"] == 1)
                rows.append(row)
                print(row, flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["half_moon:2", "8gaussians:2", "two_rings:2", "gaussian:64"],
                   help="name:d pairs")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--kinds", nargs="+", default=["dense_1"], help="dense_T (T threads) or lazy")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="device the instances are drawn on")
    p.add_argument("--out", default="pot_exact.csv")
    args = p.parse_args()
    datasets = [(s.split(":")[0], int(s.split(":")[1])) for s in args.datasets]
    rows = run(datasets, args.n, args.seeds, args.kinds, torch.device(args.device))
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
