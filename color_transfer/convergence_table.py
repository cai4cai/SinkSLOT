"""Final convergence numbers (cost, runtime, peak memory, iterations,
marginal violation) for SinkSLOT and FlashSinkhorn, one native call per
method -- no restart-based re-solving, no per-checkpoint trajectory (see
trajectory_potential.py for that instead).

    python -m color_transfer.convergence_table --output_dir DIR

Marginal violation is a post-hoc sanity check computed directly from each
method's converged potentials (reference_solvers.marginal_violation_dense /
sinkslot_marginal_violation), independent of each solver's own internal
shortcut check -- confirms potential-change convergence also implies
well-satisfied marginals.
"""

import argparse
import json
import os
import time

import torch

from color_transfer.trajectory_potential import DEFAULT_PAINTINGS_DIR, StopCfg, list_images, pixels_and_weights
from sinkslot.bench.reference_solvers import (
    flashsinkhorn_native_run, geomloss_online_native, marginal_violation_dense,
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--check_every", type=int, default=5)
    p.add_argument("--pair_idx", type=int, nargs=2, default=[2, 9])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")
    device = torch.device(args.device)
    dtype = torch.float32

    paths = list_images(args.paintings_dir)
    i, j = args.pair_idx
    sc, sw = pixels_and_weights(paths[i], device, dtype)
    tc, tw = pixels_and_weights(paths[j], device, dtype)
    print(f"Pair: {os.path.basename(paths[i])} -> {os.path.basename(paths[j])}  eps={args.eps:g}")

    rows = [
        sinkslot_row(sc, tc, sw, tw, args.eps, args.max_iter, args.tol, args.check_every, args.sinkslot_L),
        flashsinkhorn_row("FlashSinkhorn (alternating)", sc, tc, sw, tw, args.eps, args.max_iter,
                           args.tol, args.check_every, symmetric=False),
        flashsinkhorn_row("FlashSinkhorn (symmetric)", sc, tc, sw, tw, args.eps, args.max_iter,
                           args.tol, args.check_every, symmetric=True),
        geomloss_row(sc, tc, sw, tw, args.eps, args.max_iter, args.tol, args.check_every),
    ]

    header = f"{'Method':<28} {'Cost':>10} {'Time (s)':>10} {'Peak mem (GB)':>14} {'Iterations':>11} {'Converged':>10} {'Marg. viol.':>12}"
    print(header)
    for r in rows:
        print(f"{r['method']:<28} {r['cost']:>10.6f} {r['time']:>10.4f} "
              f"{r['peak_memory_bytes']/1e9:>14.4f} {r['iterations']:>11d} "
              f"{str(r['converged']):>10} {r['marginal_violation']:>12.3e}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "convergence_table.json")
    with open(out_path, "w") as f:
        json.dump({"eps": args.eps, "tol": args.tol,
                    "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])],
                    "rows": rows}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
