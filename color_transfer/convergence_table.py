"""Final convergence numbers (cost, runtime, peak memory, iterations,
marginal violation) for SinkSLOT, FlashSinkhorn and GeomLoss, one native
call per method per pair -- no restart-based re-solving, no per-checkpoint
trajectory (see trajectory_potential.py for that instead).

    python -m color_transfer.convergence_table --output_dir DIR --pair_idx 2 9
    python -m color_transfer.convergence_table --output_dir DIR --max_pairs 0

--max_pairs 0 (or omitted with --sweep) runs every ordered pair (132 for
the 12 bundled paintings), saving incrementally after each pair (so an
interrupted run keeps its progress and a re-run skips pairs already done),
then reports mean +/- standard error of the mean per method across pairs.

Marginal violation is a post-hoc sanity check computed directly from each
method's converged potentials (reference_solvers.marginal_violation_dense /
sinkslot_marginal_violation), independent of each solver's own internal
shortcut check -- confirms potential-change convergence also implies
well-satisfied marginals.
"""

import argparse
import json
import math
import os
import time

import torch

from color_transfer.trajectory_potential import DEFAULT_PAINTINGS_DIR, StopCfg, list_images, pixels_and_weights
from sinkslot.bench.reference_solvers import (
    flashsinkhorn_native_run, geomloss_multiscale_native, geomloss_online_native, marginal_violation_dense,
    sinkslot_marginal_violation,
)


def measure(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    return out, dt, peak


def sinkslot_row(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L):
    from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
    from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost, to_csr

    def solve():
        n, m = sc.shape[0], tc.shape[0]
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0)
        cost_mat = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost_mat / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        phi, psi, it, converged, change = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(), n, m, max_iter,
            stop=StopCfg(mode="potential", max_iter=max_iter, check_every=check_every, tol=tol), eps=eps)
        cost = float((sw * (eps * phi)).sum() + (tw * (eps * psi)).sum())
        return phi, psi, it, converged, cost, rows, cols, S, cost_mat

    solve()  # warmup
    (phi, psi, it, converged, cost, rows, cols, S, cost_mat), dt, peak = measure(solve)
    viol = sinkslot_marginal_violation(phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
    return {"method": "SinkSLOT", "cost": cost, "time": dt, "peak_memory_bytes": peak,
            "iterations": it, "converged": bool(converged), "marginal_violation": viol}


def flashsinkhorn_row(name, sc, tc, sw, tw, eps, max_iter, tol, check_every, symmetric):
    flashsinkhorn_native_run(sc, tc, sw, tw, eps, 10, symmetric=symmetric)  # warmup

    def solve():
        return flashsinkhorn_native_run(
            sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every, symmetric=symmetric)

    (f, g, it, converged, cost), dt, peak = measure(solve)
    viol = marginal_violation_dense(sc, tc, sw, tw, eps, f, g)
    return {"method": name, "cost": cost, "time": dt, "peak_memory_bytes": peak,
            "iterations": it, "converged": bool(converged), "marginal_violation": viol}


def geomloss_row(sc, tc, sw, tw, eps, max_iter, tol, check_every):
    geomloss_online_native(sc, tc, sw, tw, eps, 10)  # warmup

    def solve():
        return geomloss_online_native(sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every)

    (f, g, it, converged, cost, change), dt, peak = measure(solve)
    viol = marginal_violation_dense(sc, tc, sw, tw, eps, f, g)
    return {"method": "GeomLoss (online)", "cost": cost, "time": dt, "peak_memory_bytes": peak,
            "iterations": it, "converged": bool(converged), "marginal_violation": viol}


def geomloss_multiscale_row(sc, tc, sw, tw, eps, max_iter, tol, check_every):
    geomloss_multiscale_native(sc, tc, sw, tw, eps, 10)  # warmup

    def solve():
        return geomloss_multiscale_native(sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every)

    (f, g, it, converged, cost, change), dt, peak = measure(solve)
    viol = marginal_violation_dense(sc, tc, sw, tw, eps, f, g)
    return {"method": "GeomLoss (multiscale)", "cost": cost, "time": dt, "peak_memory_bytes": peak,
            "iterations": it, "converged": bool(converged), "marginal_violation": viol}


_METHOD_KEYS = ["sinkslot", "flashsinkhorn_alt", "flashsinkhorn_sym", "geomloss_online", "geomloss_multiscale"]


def run_pair(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L, methods=_METHOD_KEYS):
    rows = []
    if "sinkslot" in methods:
        rows.append(sinkslot_row(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L))
    if "flashsinkhorn_alt" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (alternating)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=False))
    if "flashsinkhorn_sym" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (symmetric)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=True))
    if "geomloss_online" in methods:
        rows.append(geomloss_row(sc, tc, sw, tw, eps, max_iter, tol, check_every))
    if "geomloss_multiscale" in methods:
        rows.append(geomloss_multiscale_row(sc, tc, sw, tw, eps, max_iter, tol, check_every))
    return rows


def print_table(rows, header_prefix=""):
    header = f"{header_prefix}{'Method':<28} {'Cost':>10} {'Time (s)':>10} {'Peak mem (GB)':>14} {'Iterations':>11} {'Converged':>10} {'Marg. viol.':>12}"
    print(header)
    for r in rows:
        print(f"{header_prefix}{r['method']:<28} {r['cost']:>10.6f} {r['time']:>10.4f} "
              f"{r['peak_memory_bytes']/1e9:>14.4f} {r['iterations']:>11d} "
              f"{str(r['converged']):>10} {r['marginal_violation']:>12.3e}")


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n < 2:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def aggregate(pair_results):
    by_method = {}
    for entry in pair_results:
        for r in entry["rows"]:
            by_method.setdefault(r["method"], []).append(r)
    summary = []
    for method, rs in by_method.items():
        converged = [r for r in rs if r["converged"]]
        row = {"method": method, "n_pairs": len(rs), "n_converged": len(converged)}
        for key in ("cost", "time", "iterations", "marginal_violation"):
            mean, se = mean_se([r[key] for r in converged])
            row[f"{key}_mean"], row[f"{key}_se"] = mean, se
        mean, se = mean_se([r["peak_memory_bytes"] / 1e9 for r in converged])
        row["peak_memory_gb_mean"], row["peak_memory_gb_se"] = mean, se
        summary.append(row)
    return summary


def print_summary(summary):
    header = (f"{'Method':<28} {'Pairs':>7} {'Cost':>18} {'Time (s)':>16} "
              f"{'Peak mem (GB)':>16} {'Iterations':>14} {'Marg. viol.':>14}")
    print(header)
    for r in summary:
        print(f"{r['method']:<28} {r['n_converged']:>3d}/{r['n_pairs']:<3d} "
              f"{r['cost_mean']:>8.5f}+/-{r['cost_se']:<8.5f} "
              f"{r['time_mean']:>6.3f}+/-{r['time_se']:<7.3f} "
              f"{r['peak_memory_gb_mean']:>6.3f}+/-{r['peak_memory_gb_se']:<7.3f} "
              f"{r['iterations_mean']:>6.1f}+/-{r['iterations_se']:<6.1f} "
              f"{r['marginal_violation_mean']:>6.2e}+/-{r['marginal_violation_se']:<6.2e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--check_every", type=int, default=5)
    p.add_argument("--pair_idx", type=int, nargs=2, default=None,
                    help="Run just this one ordered pair. Omit (or use --max_pairs) to sweep all pairs.")
    p.add_argument("--max_pairs", type=int, default=0,
                    help="0 = every ordered pair (n*(n-1) for n images); ignored if --pair_idx is given.")
    p.add_argument("--methods", type=str, nargs="+", default=_METHOD_KEYS, choices=_METHOD_KEYS,
                    help="Subset of methods to run in this invocation -- run several in parallel "
                         "processes (one per method) for wall-clock speed, each writes its own "
                         "output file, then combine with --combine.")
    p.add_argument("--combine", type=str, nargs="+", default=None,
                    help="Instead of running anything, merge these convergence_sweep*.json files "
                         "(e.g. from separate --methods runs) and print/save the aggregate summary.")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def combine_sweeps(paths_in, output_dir, eps, tol):
    by_pair = {}
    for p in paths_in:
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            key = tuple(entry["pair_idx"])
            merged = by_pair.setdefault(key, {"pair_idx": list(key), "pair": entry["pair"], "rows": []})
            merged["rows"].extend(entry["rows"])
    results = list(by_pair.values())
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "convergence_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Combined {len(paths_in)} files -> {len(results)} pairs. Saved: {out_path}\n")
    summary = aggregate(results)
    print_summary(summary)
    summary_path = os.path.join(output_dir, "convergence_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"eps": eps, "tol": tol, "n_pairs": len(results), "summary": summary}, f, indent=2)
    print(f"Saved: {summary_path}")


def main():
    args = parse_args()

    if args.combine is not None:
        combine_sweeps(args.combine, args.output_dir, args.eps, args.tol)
        return

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")
    device = torch.device(args.device)
    dtype = torch.float32
    os.makedirs(args.output_dir, exist_ok=True)

    paths = list_images(args.paintings_dir)

    if args.pair_idx is not None:
        i, j = args.pair_idx
        sc, sw = pixels_and_weights(paths[i], device, dtype)
        tc, tw = pixels_and_weights(paths[j], device, dtype)
        print(f"Pair: {os.path.basename(paths[i])} -> {os.path.basename(paths[j])}  eps={args.eps:g}")
        rows = run_pair(sc, tc, sw, tw, args.eps, args.max_iter, args.tol, args.check_every, args.sinkslot_L,
                          methods=args.methods)
        print_table(rows)
        out_path = os.path.join(args.output_dir, "convergence_table.json")
        with open(out_path, "w") as f:
            json.dump({"eps": args.eps, "tol": args.tol,
                        "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])],
                        "rows": rows}, f, indent=2)
        print(f"\nSaved: {out_path}")
        return

    all_pairs = [(i, j) for i in range(len(paths)) for j in range(len(paths)) if i != j]
    if args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]

    suffix = "" if list(args.methods) == _METHOD_KEYS else "_" + "_".join(args.methods)
    out_path = os.path.join(args.output_dir, f"convergence_sweep{suffix}.json")
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)}/{len(all_pairs)} pairs already done.")
    done = {tuple(r["pair_idx"]) for r in results}

    for n, (i, j) in enumerate(all_pairs, 1):
        if (i, j) in done:
            continue
        sc, sw = pixels_and_weights(paths[i], device, dtype)
        tc, tw = pixels_and_weights(paths[j], device, dtype)
        rows = run_pair(sc, tc, sw, tw, args.eps, args.max_iter, args.tol, args.check_every, args.sinkslot_L,
                          methods=args.methods)
        results.append({"pair_idx": [i, j],
                          "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])], "rows": rows})
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{n}/{len(all_pairs)}] {os.path.basename(paths[i])} -> {os.path.basename(paths[j])} done")
        print_table(rows, header_prefix="  ")

    print(f"\nSaved: {out_path}\n")
    summary = aggregate(results)
    print_summary(summary)
    summary_path = os.path.join(args.output_dir, "convergence_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"eps": args.eps, "tol": args.tol, "n_pairs": len(results), "summary": summary}, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
