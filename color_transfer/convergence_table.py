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

Two cost fields, both post-hoc from the converged potentials, not each
solver's own shortcut: "cost" is the true transport cost <C,P>, comparable
across methods regardless of reference measure; "dual_cost" is the Sinkhorn
dual value <a,f>+<b,g> = <C,P> + eps*KL(P|reference), NOT comparable across
methods whose reference measure differs (SinkSLOT's sparse P^SOT vs the
other four's dense a(x)b), since the KL term is then relative to a
different baseline for each. Marginal violation is the same kind of
post-hoc check, confirming potential-change convergence also implies
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
    flashsinkhorn_samplesloss_run, geomloss_online_native, plan_diagnostics_dense,
    plan_diagnostics_dense_l1, plan_diagnostics_dense_rounded, sinkslot_plan_diagnostics, sinkslot_plan_diagnostics_l1,
    sinkslot_plan_diagnostics_rounded,
)

# Two representative pairs (lowest/highest SinkSLOT transport cost at eps=0.01,
# per the paper's own qualitative-figure selection) get an embedded marginal-
# violation trajectory recorded alongside their normal (unaffected) row --
# piggybacked on whichever sweep job is already solving that pair for that
# method, rather than a separate restart-heavy script re-solving from scratch.
TRAJECTORY_PAIRS = {(6, 7), (4, 2)}
TRAJECTORY_CHECKPOINTS = [10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400]


def record_trajectory(method_key, sc, tc, sw, tw, eps, tol, check_every, sinkslot_L, allow_tf32=True):
    """Checkpoint the true L1 marginal violation (row, col, mass) over the
    course of solving, for one of the two TRAJECTORY_PAIRS.

    Diagnostic computation time (plan_diagnostics_*_l1, an O(N*M/block) or
    O(nnz) reconstruction pass) is deliberately NOT included in the recorded
    'time' field -- only the solver's own time to reach that checkpoint is
    measured, so this never contaminates a speed comparison the way it would
    if left in.

    FlashSinkhorn symmetric warm-starts between checkpoints (f_init/g_init,
    genuine single continuous run, supported by the low-level function even
    though SamplesLoss doesn't expose it). FlashSinkhorn alternating and
    SinkSLOT have no warm-start hook, so those restart from scratch at each
    checkpoint's iteration count -- acceptable here since this only runs for
    2 of 132 pairs, not the full sweep.
    """
    checkpoints = []

    if method_key == "sinkslot":
        from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
        from sinkslot.solver import (
            _ot_1d_coo_batched, _ot_1d_coo_batched_cuda_fp32, sot_plan_coo, sparse_sqeuclidean_cost, to_csr,
        )
        ot1d = _ot_1d_coo_batched_cuda_fp32 if sc.is_cuda else _ot_1d_coo_batched
        n, m = sc.shape[0], tc.shape[0]
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0, ot1d=ot1d)
        cost_mat = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost_mat / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        log_a, log_b = sw.log(), tw.log()
        for n_iter in TRAJECTORY_CHECKPOINTS:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            phi, psi, it, converged, change = sinkslot_alternating_triton(
                r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_iter, stop=None, eps=eps)
            torch.cuda.synchronize()
            solve_dt = time.perf_counter() - t0
            row_l1, col_l1, mass_l1 = sinkslot_plan_diagnostics_l1(phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
            checkpoints.append({"iters": n_iter, "solve_time": solve_dt,
                                 "row_l1": row_l1, "col_l1": col_l1, "mass_l1": mass_l1})
        return checkpoints

    # flashsinkhorn_alt / flashsinkhorn_sym (+ their _fp32 variants)
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating, sinkhorn_flashstyle_symmetric
    symmetric = "sym" in method_key
    f, g = None, None
    prev = 0
    for n_iter in TRAJECTORY_CHECKPOINTS:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if symmetric:
            f, g = sinkhorn_flashstyle_symmetric(
                sc, tc, sw, tw, eps=eps, n_iters=n_iter - prev, use_epsilon_scaling=False,
                last_extrapolation=False, allow_tf32=allow_tf32, f_init=f, g_init=g)
        else:
            f, g = sinkhorn_flashstyle_alternating(sc, tc, sw, tw, eps=eps, n_iters=n_iter, allow_tf32=allow_tf32)
        torch.cuda.synchronize()
        solve_dt = time.perf_counter() - t0
        row_l1, col_l1, mass_l1 = plan_diagnostics_dense_l1(sc, tc, sw, tw, eps, f, g)
        checkpoints.append({"iters": n_iter, "solve_time": solve_dt,
                             "row_l1": row_l1, "col_l1": col_l1, "mass_l1": mass_l1})
        prev = n_iter
    return checkpoints


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
    from sinkslot.solver import (
        _ot_1d_coo_batched, _ot_1d_coo_batched_cuda_fp32, sot_plan_coo, sparse_sqeuclidean_cost, to_csr,
    )

    # Same device-switch convention as sinkslot_solve's own default and every
    # gradient_flow/ call site: the CUDA-optimised (transposed-layout) support
    # builder is ~7x faster at construction with no accuracy cost, at this
    # pipeline's own float32 (the _fp32 variant, not the fp64-internal
    # SinkSLOT-CUDA baseline used elsewhere in the paper's own benchmarks).
    ot1d = _ot_1d_coo_batched_cuda_fp32 if sc.is_cuda else _ot_1d_coo_batched

    def solve():
        n, m = sc.shape[0], tc.shape[0]
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0, ot1d=ot1d)
        cost_mat = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost_mat / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        phi, psi, it, converged, change = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(), n, m, max_iter,
            stop=StopCfg(mode="potential", max_iter=max_iter, check_every=check_every, tol=tol), eps=eps)
        dual_cost = float((sw * (eps * phi)).sum() + (tw * (eps * psi)).sum())
        return phi, psi, it, converged, dual_cost, rows, cols, S, cost_mat

    solve()  # warmup
    (phi, psi, it, converged, dual_cost, rows, cols, S, cost_mat), dt, peak = measure(solve)
    # Rounding (issue #57, Altschuler/Niles-Weed/Rigollet NeurIPS 2017 Algorithm
    # 2) is deliberately OUTSIDE measure() / not part of `dt`: it's post-hoc
    # feasibility enforcement on the final plan, not part of the solve being
    # timed. "cost"/"marginal_violation" below are the ROUNDED plan's; the
    # unrounded ones (what the solver actually returned) are kept alongside.
    unrounded_viol, unrounded_cost = sinkslot_plan_diagnostics(phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
    viol, cost = sinkslot_plan_diagnostics_rounded(sc, tc, phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
    return {"method": "SinkSLOT", "cost": cost, "dual_cost": dual_cost, "time": dt,
            "peak_memory_bytes": peak, "iterations": it, "converged": bool(converged),
            "marginal_violation": viol,
            "unrounded_cost": unrounded_cost, "unrounded_marginal_violation": unrounded_viol}


def sinkslot_sym_row(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L):
    """Same construction as sinkslot_row, but the alpha=0.5 Jacobi
    (sinkslot_symmetric_triton) solver instead of Gauss-Seidel. Its
    "potential" stop check is structurally identical to
    sinkslot_alternating_triton's / FlashSinkhorn-alternating's (checkpoint-
    to-checkpoint delta of the CURRENT state, every check_every iterations)
    -- but here that state is itself alpha-damped
    (phi_new = (1-alpha)*phi + alpha*phi_cand), so the observed checkpoint
    delta is ~alpha=0.5 times the raw (undamped) step for the same
    underlying progress, exactly the same asymmetry FlashSinkhorn-symmetric
    has relative to alternating/GeomLoss. tol/2 restores parity: see
    chat -- observed_step = alpha*raw_step, so observed_step < alpha*tol
    <=> raw_step < tol, matching alternating's own precision.
    """
    from sinkslot.sinkhorn_solvers import sinkslot_symmetric_triton
    from sinkslot.solver import (
        _ot_1d_coo_batched, _ot_1d_coo_batched_cuda_fp32, sot_plan_coo, sparse_sqeuclidean_cost, to_csr,
    )

    ot1d = _ot_1d_coo_batched_cuda_fp32 if sc.is_cuda else _ot_1d_coo_batched
    sym_tol = tol / 2

    def solve():
        n, m = sc.shape[0], tc.shape[0]
        rows, cols, S = sot_plan_coo(sc, tc, sw, tw, L=sinkslot_L, seed=0, ot1d=ot1d)
        cost_mat = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        lam = log_S - cost_mat / eps
        r_ptr, r_idx, r_lam, _ = to_csr(rows, cols, lam, n)
        c_ptr, c_idx, c_lam, _ = to_csr(cols, rows, lam, m)
        phi, psi, it, converged, change = sinkslot_symmetric_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, sw.log(), tw.log(), n, m, max_iter,
            stop=StopCfg(mode="potential", max_iter=max_iter, check_every=check_every, tol=sym_tol), eps=eps)
        dual_cost = float((sw * (eps * phi)).sum() + (tw * (eps * psi)).sum())
        return phi, psi, it, converged, dual_cost, rows, cols, S, cost_mat

    solve()  # warmup
    (phi, psi, it, converged, dual_cost, rows, cols, S, cost_mat), dt, peak = measure(solve)
    unrounded_viol, unrounded_cost = sinkslot_plan_diagnostics(phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
    viol, cost = sinkslot_plan_diagnostics_rounded(sc, tc, phi, psi, rows, cols, S, cost_mat, eps, sw, tw)
    return {"method": "SinkSLOT (symmetric)", "cost": cost, "dual_cost": dual_cost, "time": dt,
            "peak_memory_bytes": peak, "iterations": it, "converged": bool(converged),
            "marginal_violation": viol,
            "unrounded_cost": unrounded_cost, "unrounded_marginal_violation": unrounded_viol}


def flashsinkhorn_row(name, sc, tc, sw, tw, eps, max_iter, tol, check_every, symmetric, allow_tf32=True):
    flashsinkhorn_samplesloss_run(sc, tc, sw, tw, eps, 10, symmetric=symmetric, allow_tf32=allow_tf32)  # warmup

    def solve():
        return flashsinkhorn_samplesloss_run(
            sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every,
            symmetric=symmetric, allow_tf32=allow_tf32)

    (f, g, it, converged, dual_cost), dt, peak = measure(solve)
    unrounded_viol, unrounded_cost = plan_diagnostics_dense(sc, tc, sw, tw, eps, f, g)
    viol, cost = plan_diagnostics_dense_rounded(sc, tc, sw, tw, eps, f, g)
    return {"method": name, "cost": cost, "dual_cost": dual_cost, "time": dt,
            "peak_memory_bytes": peak, "iterations": it, "converged": bool(converged),
            "marginal_violation": viol,
            "unrounded_cost": unrounded_cost, "unrounded_marginal_violation": unrounded_viol}


def geomloss_row(sc, tc, sw, tw, eps, max_iter, tol, check_every):
    geomloss_online_native(sc, tc, sw, tw, eps, 10)  # warmup

    def solve():
        return geomloss_online_native(sc, tc, sw, tw, eps, max_iter, threshold=tol, check_every=check_every)

    (f, g, it, converged, dual_cost, change), dt, peak = measure(solve)
    unrounded_viol, unrounded_cost = plan_diagnostics_dense(sc, tc, sw, tw, eps, f, g)
    viol, cost = plan_diagnostics_dense_rounded(sc, tc, sw, tw, eps, f, g)
    return {"method": "GeomLoss (online)", "cost": cost, "dual_cost": dual_cost, "time": dt,
            "peak_memory_bytes": peak, "iterations": it, "converged": bool(converged),
            "marginal_violation": viol,
            "unrounded_cost": unrounded_cost, "unrounded_marginal_violation": unrounded_viol}


_METHOD_KEYS = ["sinkslot", "sinkslot_sym", "flashsinkhorn_alt", "flashsinkhorn_sym", "geomloss_online",
                "flashsinkhorn_alt_fp32", "flashsinkhorn_sym_fp32"]


def run_pair(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L, methods=_METHOD_KEYS):
    rows = []
    if "sinkslot" in methods:
        rows.append(sinkslot_row(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L))
    if "sinkslot_sym" in methods:
        rows.append(sinkslot_sym_row(sc, tc, sw, tw, eps, max_iter, tol, check_every, sinkslot_L))
    if "flashsinkhorn_alt" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (alternating)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=False))
    if "flashsinkhorn_sym" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (symmetric)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=True))
    if "flashsinkhorn_alt_fp32" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (alternating, fp32)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=False, allow_tf32=False))
    if "flashsinkhorn_sym_fp32" in methods:
        rows.append(flashsinkhorn_row("FlashSinkhorn (symmetric, fp32)", sc, tc, sw, tw, eps, max_iter,
                                       tol, check_every, symmetric=True, allow_tf32=False))
    if "geomloss_online" in methods:
        rows.append(geomloss_row(sc, tc, sw, tw, eps, max_iter, tol, check_every))
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
        for key in ("cost", "dual_cost", "time", "iterations", "marginal_violation"):
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
    p.add_argument("--max_iter", type=int, default=8000)
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
    p.add_argument("--num_shards", type=int, default=1,
                    help="Split all_pairs round-robin across this many shards, for further "
                         "parallelizing a single slow --methods entry (e.g. flashsinkhorn_sym_fp32) "
                         "across several concurrent jobs. Each shard writes its own output file.")
    p.add_argument("--shard_idx", type=int, default=0,
                    help="Which shard (0-indexed, < --num_shards) this invocation computes.")
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
    if args.num_shards > 1:
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError(f"--shard_idx must be in [0, {args.num_shards}), got {args.shard_idx}")
        all_pairs = all_pairs[args.shard_idx::args.num_shards]

    suffix = "" if list(args.methods) == _METHOD_KEYS else "_" + "_".join(args.methods)
    if args.num_shards > 1:
        suffix += f"_shard{args.shard_idx}of{args.num_shards}"
    out_path = os.path.join(args.output_dir, f"convergence_sweep{suffix}.json")
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)}/{len(all_pairs)} pairs already done.")
    done = {tuple(r["pair_idx"]) for r in results}

    traj_methods = {m: allow_tf32 for m, allow_tf32 in
                     [("sinkslot", True), ("flashsinkhorn_alt", True), ("flashsinkhorn_sym", True),
                      ("flashsinkhorn_alt_fp32", False), ("flashsinkhorn_sym_fp32", False)]
                     if m in args.methods}
    traj_path = os.path.join(args.output_dir, f"convergence_trajectory{suffix}.json")
    trajectories = {}
    if os.path.exists(traj_path):
        with open(traj_path) as f:
            trajectories = json.load(f)

    for n, (i, j) in enumerate(all_pairs, 1):
        if (i, j) in done:
            continue
        sc, sw = pixels_and_weights(paths[i], device, dtype)
        tc, tw = pixels_and_weights(paths[j], device, dtype)
        rows = run_pair(sc, tc, sw, tw, args.eps, args.max_iter, args.tol, args.check_every, args.sinkslot_L,
                          methods=args.methods)
        results.append({"pair_idx": [i, j],
                          "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])], "rows": rows})

        if (i, j) in TRAJECTORY_PAIRS and traj_methods:
            pair_key = f"{i}_{j}"
            entry = trajectories.setdefault(pair_key, {})
            for method_key, allow_tf32 in traj_methods.items():
                if method_key in entry:
                    continue
                print(f"  [trajectory] {method_key} for pair ({i},{j})...")
                entry[method_key] = record_trajectory(
                    method_key, sc, tc, sw, tw, args.eps, args.tol, args.check_every,
                    args.sinkslot_L, allow_tf32=allow_tf32)
            with open(traj_path, "w") as f:
                json.dump(trajectories, f, indent=2)

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
