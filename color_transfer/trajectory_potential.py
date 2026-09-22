"""Convergence trajectories under the potential-change stopping rule --
now safe to use after fixing a real bug in SinkSLOT's own "potential" mode
(torch-ext/sinkslot/sinkhorn_solvers.py: torch.utils.swap_tensors silently
rewound phi/psi on any non-converging check, corrupting both the returned
potentials and the reported change -- see that file's own updated comment
for the fix, and torch-ext/sinkslot/bench/reference_solvers.py's module
docstring for the verification that SinkSLOT's eps*max(|dphi|,|dpsi|) and
FlashSinkhorn's/GeomLoss's max(|df|,|dg|) are the mathematically IDENTICAL
quantity once phi=f/eps is accounted for).

Four methods, all checked under the SAME verified-equivalent criterion:
SinkSLOT (native "potential" mode, now fixed), FlashSinkhorn alternating
and symmetric (upstream's own native threshold/check_every, no fork
needed), GeomLoss online (reference_solvers.geomloss_online_native, a
from-scratch reimplementation of sinkhorn_loop's own update math with the
same native-style check added, since GeomLoss's library code has no
per-iteration hook at all).

    python -m color_transfer.trajectory_potential --output_dir DIR --method sinkslot

Restart-based checkpointing, same limitation as trajectory.py: none of
these support resuming from a previous call's potentials, so each
checkpoint re-solves from scratch at an increasing max_iter budget,
stopping once a checkpoint genuinely converges (the checkpoint loop does
not continue past that point, since a larger budget would just re-converge
to the same answer).
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from color_transfer.main_pixel import DEFAULT_PAINTINGS_DIR, list_images, pixels_and_weights
from sinkslot.bench.reference_solvers import flashsinkhorn_native_run, geomloss_online_native


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                    choices=["sinkslot", "flashsinkhorn", "flashsinkhorn_symmetric", "geomloss"])
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--check_every", type=int, default=50,
                    help="Outer restart-checkpoint spacing. The native check_every "
                         "passed to each solver's OWN internal check is fixed at 5 "
                         "(matching FlashSinkhorn's own default), independent of this.")
    p.add_argument("--warmup_iters", type=int, default=10)
    p.add_argument("--pair_idx", type=int, nargs=2, default=[2, 9])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def sinkslot_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol, L, seed=0):
    from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
    from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost, to_csr
    from color_transfer.main_pixel import StopCfg

    n, m = sc.shape[0], tc.shape[0]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=L, seed=seed)
    cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    lam = log_S - cost / eps
    r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
    c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
    torch.cuda.synchronize()
    setup_time = time.perf_counter() - t0

    log_a, log_b = sw.log(), tw.log()
    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        phi, psi, it, converged, change = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_try,
            stop=StopCfg(mode="potential", max_iter=n_try, check_every=5, tol=tol), eps=eps)
        torch.cuda.synchronize()
        dt = setup_time + (time.perf_counter() - t0)
        checkpoints.append({"iters": it, "time": dt, "potential_change": float(change),
                             "converged": bool(converged)})
        if converged:
            break
    return setup_time, checkpoints


def flashsinkhorn_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol, symmetric):
    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g, it, converged, change = flashsinkhorn_native_run(
            sc, tc, sw, tw, eps, n_try, threshold=tol, check_every=5, symmetric=symmetric,
            report_change=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        checkpoints.append({"iters": it, "time": dt, "potential_change": float(change),
                             "converged": bool(converged)})
        if converged:
            break
    return 0.0, checkpoints


def geomloss_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol):
    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g, it, converged, change = geomloss_online_native(
            sc, tc, sw, tw, eps, n_try, threshold=tol, check_every=5)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        checkpoints.append({"iters": it, "time": dt, "potential_change": float(change),
                             "converged": bool(converged)})
        if converged:
            break
    return 0.0, checkpoints


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")
    device = torch.device(args.device)
    dtype = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    paths = list_images(args.paintings_dir)
    i, j = args.pair_idx
    sc, sw = pixels_and_weights(paths[i], device, dtype)
    tc, tw = pixels_and_weights(paths[j], device, dtype)
    print(f"Method: {args.method}  pair: {os.path.basename(paths[i])} -> "
          f"{os.path.basename(paths[j])}  tol={args.tol:g}")

    if args.method == "sinkslot":
        from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
        from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost, to_csr
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=args.sinkslot_L, seed=0)
        cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost / args.eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, sc.shape[0])
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, tc.shape[0])
        sinkslot_alternating_triton(r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam,
                                     sw.log(), tw.log(), sc.shape[0], tc.shape[0],
                                     args.warmup_iters, stop=None)
        torch.cuda.synchronize()
        setup_time, checkpoints = sinkslot_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol, args.sinkslot_L)
    elif args.method in ("flashsinkhorn", "flashsinkhorn_symmetric"):
        symmetric = args.method == "flashsinkhorn_symmetric"
        flashsinkhorn_native_run(sc, tc, sw, tw, args.eps, args.warmup_iters, symmetric=symmetric)
        torch.cuda.synchronize()
        setup_time, checkpoints = flashsinkhorn_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol, symmetric)
    else:  # geomloss
        geomloss_online_native(sc, tc, sw, tw, args.eps, args.warmup_iters)
        torch.cuda.synchronize()
        setup_time, checkpoints = geomloss_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol)

    for c in checkpoints:
        pot = c.get("potential_change")
        pot_str = f"{pot:.4e}" if pot is not None else "n/a"
        print(f"  iters={c['iters']:5d}  time={c['time']:.4f}s  "
              f"potential_change={pot_str}  converged={c['converged']}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"trajectory_potential_{args.method}.json")
    with open(out_path, "w") as f:
        json.dump({"method": args.method, "eps": args.eps, "tol": args.tol,
                    "max_iter": args.max_iter, "check_every": args.check_every,
                    "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])],
                    "n": sc.shape[0], "m": tc.shape[0],
                    "setup_time": setup_time, "checkpoints": checkpoints}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
