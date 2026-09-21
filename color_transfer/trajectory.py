"""Convergence trajectories for one representative image pair: marginal
violation and primal-dual gap, checkpointed every --check_every iterations,
against cumulative wall-clock runtime -- for SinkSLOT, FlashSinkhorn, and
GeomLoss.

marginal_viol: for SinkSLOT, comes directly from its own internal
stopping-check machinery (sinkslot_alternating_triton's stop.mode="marginal",
one extra cheap call per checkpoint -- see sinkslot_trajectory's own
docstring), the same quantity --stop_mode=marginal actually stops on in
main_pixel.py, not a hand-rolled reimplementation. FlashSinkhorn and GeomLoss
expose no such internal check value at all (verified directly against
sinkhorn_flashstyle_alternating's own signature: it returns only
(f, g[, n_iters_used]) under any stop_mode; GeomLoss's low-level API has no
stopping check of any kind), so their marginal_viol is a post-hoc computation
(_marginal_violation_dense_chunked).

No potential_change field: SinkSLOT's own stop.mode="potential" turns out to
be unsafe to call in the "run to n_try without ever converging" pattern this
script needs (tol=0.0, forced full budget) -- see sinkslot_trajectory's
docstring for the confirmed bug (a checkpoint that doesn't converge can
silently rewind phi/psi to a stale, non-final state, not just fail to
report a meaningful change). Left out entirely rather than falling back to
a post-hoc/checkpoint-to-checkpoint approximation for now.

primal_dual_gap has no native equivalent in any of the three libraries, so
it is always a post-hoc computation (_kl_gap_sparse/_kl_gap_dense_chunked),
including for SinkSLOT.

    python -m color_transfer.trajectory --output_dir DIR --method sinkslot
    python -m color_transfer.trajectory --output_dir DIR --method flashsinkhorn
    python -m color_transfer.trajectory --output_dir DIR --method geomloss --geomloss_backend online

Requires a CUDA GPU, same as main_pixel.py.

primal_dual_gap can read exactly 0.0 for many early checkpoints -- this is
main_pixel.py's _kl_gap_sparse/_kl_gap_dense_chunked own documented
max(gap, 0.0) clamp, not a bug: verified directly (raw, unclamped
primal-dual value checked across n_iters=10..3000 on the default pair)
that the true value is genuinely negative early on -- a known property of
the feasible-plan-reconstruction approximation these functions use, whose
error shrinks smoothly and monotonically alongside marginal violation and
crosses over to genuinely positive once well-converged, not a stuck or
unbounded artifact. Practical consequence: since a log-scale y-axis can't
plot a literal 0.0, any plotting code consuming this script's output
should skip checkpoints where primal_dual_gap == 0.0 for that curve
specifically, rather than treat them as real (tiny) values.

Why this is a separate script from main_pixel.py: main_pixel.py records one
number per (pair, eps) -- whichever quantity --stop_mode selects, at
whatever iteration count that mode's check happened to stop at. This script
instead records BOTH quantities (marginal violation and primal-dual gap) at
EVERY checkpoint along the way, for one fixed, deliberately small pair, to
produce a convergence plot -- a fundamentally different data-collection
shape, not a mode of the sweep script.

Checkpointing method: none of sinkslot_alternating_triton,
sinkhorn_flashstyle_alternating, or GeomLoss's low-level sinkhorn_loop
support warm-starting (same limitation main_pixel.py's primal_dual mode
already works around), so each checkpoint re-solves from scratch at that
checkpoint's iteration count -- the recorded "time" for checkpoint k is
therefore the wall-clock duration of one from-scratch call reaching k
iterations, not a running sum. This makes the total cost of recording a K-
checkpoint trajectory grow like O(K^2), not O(K): deliberately kept cheap
by (a) using a modest --max_iter (default 400) and --check_every (default
10, so 40 checkpoints -- sum(10, 20, ..., 400) = 8200 iteration-equivalents
total, not 400) and (b) defaulting to the two SMALLEST images in the
bundled dataset (--pair_idx, default (2, 9): "the-thames..." (40,711
colors) and "the-toques..." (56,488 colors) -- see list_images's own sort
order) rather than main_pixel.py's own largest-scale smoke-test pair.

SinkSLOT-specific: unlike FlashSinkhorn/GeomLoss, which derive their
kernels implicitly from the data with no precomputation, SinkSLOT needs an
explicit one-time step to build its sparse support (sot_plan_coo + cost +
CSR/CSC conversion, exactly the "setup_ms" this repo's own scalability
benchmarks separate from the solve loop -- see bench_forward.py's
bench_sinkslotcuda). Built ONCE here (not redone per checkpoint), timed
separately, and reported as its own `setup_time` field alongside the
checkpoint list -- every checkpoint's `time` already has this cost folded
in (checkpoint time = setup_time + that checkpoint's from-scratch solve
time), so the reported times are genuine cumulative wall-clock from t=0,
and a plot can still shade the first `setup_time` seconds as its own
region if it wants to call that out explicitly.
"""

import argparse
import json
import os
import time

import torch

from color_transfer.main_pixel import (
    DEFAULT_PAINTINGS_DIR,
    _GEOMLOSS_BACKENDS,
    _kl_gap_dense_chunked,
    _kl_gap_sparse,
    _marginal_violation_dense_chunked,
    list_images,
    pixels_and_weights,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR))
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--method", type=str, required=True, choices=["sinkslot", "flashsinkhorn", "geomloss"])
    p.add_argument("--geomloss_backend", type=str, default="online", choices=["online", "multiscale"])
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--max_iter", type=int, default=2000,
                    help="At eps=0.1 on the default (smallest) pair, the primal-dual gap's raw "
                         "value is genuinely negative (clamped to 0 -- not a bug, see module "
                         "docstring) for roughly the first 500-1000 iterations; max_iter needs to "
                         "reach past that for the primal_dual curve to show real, non-clamped "
                         "values over most of its range.")
    p.add_argument("--check_every", type=int, default=50,
                    help="Checkpointing is restart-based (see module docstring): total cost grows "
                         "like O((max_iter/check_every)^2), so this trades trajectory resolution "
                         "for cost. 50 keeps a 2000-iteration trajectory at 40 checkpoints.")
    p.add_argument("--warmup_iters", type=int, default=10)
    p.add_argument("--pair_idx", type=int, nargs=2, default=[2, 9],
                    help="Indices into the sorted image list (default: the two smallest bundled "
                         "paintings by unique-color count, to keep this script's O(K^2) restart "
                         "cost affordable).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _cost_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n=2048):
    """<C, P> for the plan implied by (f, g), P_ij = a_i*b_j*exp((f_i+g_j-C_ij)/eps).
    Never materializes a dense (n,m) tensor -- same row-blocked pattern as
    _kl_gap_dense_chunked/_marginal_violation_dense_chunked."""
    n = sc.shape[0]
    cost_val = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]
        P_blk = sw[start:end, None] * tw[None, :] * torch.exp(
            (f[start:end, None] + g[None, :] - C_blk) / eps)
        cost_val += float((P_blk * C_blk).sum())
        del C_blk, P_blk
    return cost_val


def sinkslot_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, L, seed=0):
    """Build the sparse support once (timed separately as setup_time), then
    checkpoint sinkslot_alternating_triton at n_try = check_every,
    2*check_every, ..., max_iter. Returns (setup_time, checkpoints), where
    each checkpoint's own `time` already includes setup_time (see module
    docstring).

    marginal_viol comes from SinkSLOT's own internal stopping-check
    machinery directly (stop.mode="marginal"), not a hand-rolled
    reimplementation: one EXTRA call per checkpoint (beyond the timed
    stop=None call used for cost/primal_dual_gap), with tol=0.0 (can never
    trigger, since the checked quantity is non-negative by construction --
    so this still runs exactly n_try iterations, same as stop=None) and
    check_every=1 (fine here -- see below). Per sinkslot_alternating_triton's
    own docstring, "marginal" mode's check is a proxy for the true max
    row/col violation (traded for one fewer LSE call per side per check),
    not necessarily bit-identical to an independently-derived exact
    computation -- using it here means trusting the same quantity SinkSLOT's
    own --stop_mode=marginal actually stops on, which is the point.

    No potential_change field: stop.mode="potential" has a confirmed
    correctness bug in this forced-full-budget (tol=0.0) usage pattern.
    Its prev_phi/prev_psi update uses torch.utils.swap_tensors -- an
    in-place content swap, not a copy -- gated behind `if it % check_every
    == 0`. When that check does NOT converge, the swap doesn't just update
    the comparison reference: it rewinds phi/psi themselves back to
    whatever prev_phi/prev_psi held (the state as of the PREVIOUS check),
    so the following check_every-iteration block starts over from that old
    state instead of continuing forward, silently discarding the iterations
    in between. Confirmed directly on GPU: stop=StopCfg(mode="potential",
    max_iter=100, check_every=50, tol=0.0) returned phi identical (to
    machine precision) to a genuine stop=None 50-iteration run, and 4.6e-1
    away (max-abs) from the true 100-iteration result -- iterations 51-100
    were computed and then thrown away. check_every=1 hits the same bug on
    every single iteration (traced separately: iteration 2 recomputes
    iteration 1's result from the swapped-back-in stale input, then compares
    it against a prev_phi the same swap had just set to that identical
    value, reporting change=0.0 and, worse, converged=True regardless of
    tol -- confirmed at tol=1e-2, 1e-4, and 1e-8 alike). "marginal" mode
    does not have this problem: its phi_old/psi_old swap is unconditional
    (happens every iteration via torch.utils.swap_tensors before each
    half-step, not gated on the check), so its viol is always a fresh
    1-step residual regardless of check_every. This looks like a genuine
    latent bug in SinkSLOT's own library (not just an issue with this
    script's usage pattern): any real --stop_mode=potential run in
    main_pixel.py that exhausts max_iter without converging could return a
    stale, non-final phi/psi rather than the true last iterate. Worth
    flagging upstream; out of scope to patch sinkhorn_solvers.py itself
    here.

    FlashSinkhorn/GeomLoss can't provide marginal_viol this way either:
    verified directly against sinkhorn_flashstyle_alternating's own
    signature that it returns only (f, g[, n_iters_used]) under ANY
    stop_mode, no convergence-check value at all -- so those two methods'
    marginal_viol stays a post-hoc computation (see
    flashsinkhorn_trajectory/geomloss_trajectory). primal_dual_gap has no
    native equivalent in any of the three libraries (confirmed for all
    three across this file's development), so it always needs the post-hoc
    _kl_gap_sparse/_kl_gap_dense_chunked computation, including here for
    SinkSLOT.
    """
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
        phi, psi, _, _, _ = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_try, stop=None)
        torch.cuda.synchronize()
        dt = setup_time + (time.perf_counter() - t0)

        vals = (phi[rows] + psi[cols] + lam).exp()
        cost_val = float((vals * cost).sum())
        gap = _kl_gap_sparse(phi, psi, rows, cols, S, cost, sw, tw, eps)

        _, _, _, _, mv = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_try,
            stop=StopCfg(mode="marginal", max_iter=n_try, check_every=1, tol=0.0))

        checkpoints.append({"iters": n_try, "time": dt, "marginal_viol": float(mv),
                             "primal_dual_gap": gap, "cost": cost_val})
    return setup_time, checkpoints


def flashsinkhorn_trajectory(sc, tc, sw, tw, eps, max_iter, check_every):
    """No setup phase (FlashSinkhorn derives its kernel implicitly from the
    data every call) -- returns (0.0, checkpoints)."""
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g, _ = sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=n_try, threshold=None,
            allow_tf32=False, return_n_iters=True,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mv = _marginal_violation_dense_chunked(f, g, sc, tc, sw, tw, eps)
        gap = _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps)
        cost_val = _cost_dense_chunked(f, g, sc, tc, sw, tw, eps)
        checkpoints.append({"iters": n_try, "time": dt, "marginal_viol": mv,
                             "primal_dual_gap": gap, "cost": cost_val})
    return 0.0, checkpoints


def geomloss_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, backend):
    """No setup phase, same as FlashSinkhorn -- returns (0.0, checkpoints)."""
    solve = _GEOMLOSS_BACKENDS[backend]
    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g = solve(sc, tc, sw, tw, eps, n_try)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mv = _marginal_violation_dense_chunked(f, g, sc, tc, sw, tw, eps)
        gap = _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps)
        cost_val = _cost_dense_chunked(f, g, sc, tc, sw, tw, eps)
        checkpoints.append({"iters": n_try, "time": dt, "marginal_viol": mv,
                             "primal_dual_gap": gap, "cost": cost_val})
    return 0.0, checkpoints


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU (all three solvers dispatch to "
                            "fused Triton/KeOps kernels with no pure-torch fallback here).")
    device = torch.device(args.device)
    dtype = torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    paths = list_images(args.paintings_dir)
    i, j = args.pair_idx
    sc, sw = pixels_and_weights(paths[i], device, dtype)
    tc, tw = pixels_and_weights(paths[j], device, dtype)
    print(f"Method: {args.method}  pair: {os.path.basename(paths[i])} "
          f"({sc.shape[0]} colors) -> {os.path.basename(paths[j])} ({tc.shape[0]} colors)")

    # untimed warmup: absorb this shape's Triton/KeOps JIT/autotune cost
    # before the timed checkpoint loop, so it doesn't leak into checkpoint 1.
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
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.sinkslot_L)
    elif args.method == "flashsinkhorn":
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
        sinkhorn_flashstyle_alternating(sc, tc, sw, tw, eps=args.eps, n_iters=args.warmup_iters,
                                         threshold=None, allow_tf32=False)
        torch.cuda.synchronize()
        setup_time, checkpoints = flashsinkhorn_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every)
    else:  # geomloss
        solve = _GEOMLOSS_BACKENDS[args.geomloss_backend]
        solve(sc, tc, sw, tw, args.eps, args.warmup_iters)
        torch.cuda.synchronize()
        setup_time, checkpoints = geomloss_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.geomloss_backend)

    for c in checkpoints:
        print(f"  iters={c['iters']:5d}  time={c['time']:.4f}s  "
              f"marginal_viol={c['marginal_viol']:.4e}  primal_dual_gap={c['primal_dual_gap']:.4e}  "
              f"cost={c['cost']:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    method_key = f"{args.method}_{args.geomloss_backend}" if args.method == "geomloss" else args.method
    out_path = os.path.join(args.output_dir, f"trajectory_{method_key}.json")
    with open(out_path, "w") as f:
        json.dump({
            "method": args.method,
            "geomloss_backend": args.geomloss_backend if args.method == "geomloss" else None,
            "eps": args.eps, "max_iter": args.max_iter, "check_every": args.check_every,
            "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])],
            "n": sc.shape[0], "m": tc.shape[0],
            "setup_time": setup_time,
            "checkpoints": checkpoints,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
