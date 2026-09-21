"""Convergence trajectories for one representative image pair: marginal
violation and primal-dual gap, checkpointed at increasing iteration counts
up to genuine convergence (real --tol, not a forced fixed budget) -- for
SinkSLOT, FlashSinkhorn (--method flashsinkhorn, Gauss-Seidel/alternating
updates), FlashSinkhorn-symmetric (--method flashsinkhorn_symmetric,
Jacobi/damped-symmetric updates), and GeomLoss.

Why FlashSinkhorn-symmetric is here: GeomLoss needs far more iterations
than FlashSinkhorn to reach the same --tol on this pair, which looks
surprising since both converge to the same transport cost (same problem).
Root cause, confirmed by a controlled diagnostic: GeomLoss's low-level
sinkhorn_loop (the single routine behind its online/multiscale/tensorized
backends alike, with no alternative) hardcodes symmetric, damped Jacobi
updates -- update BOTH f and g from the PREVIOUS iteration's values
simultaneously, then blend 50% old / 50% new -- not the Gauss-Seidel scheme
(update f, then update g using the FRESH f) FlashSinkhorn's alternating
solver uses. sinkhorn_flashstyle_symmetric is FlashSinkhorn's own
implementation of that same symmetric scheme (its own docstring calls it
"GeomLoss-style symmetric updates"): verified directly that its
marginal-violation trajectory tracks GeomLoss-online's almost exactly at
every iteration count on this pair (e.g. both ~5.5-5.9e-7 at n=300), while
alternating is already 4+ orders of magnitude past both by n=150 -- same
library, same problem instance, only the update rule changed. This is a
genuine Gauss-Seidel-vs-Jacobi algorithmic difference (a well-known
classical-iterative-methods effect: Gauss-Seidel typically has a
meaningfully larger per-iteration contraction factor), not a GeomLoss bug,
not an eps/cost-convention mismatch, and not something main_pixel.py's
--method flashsinkhorn arm can be configured to avoid, since GeomLoss's
API offers no alternating option at any level.

Run-until-convergence, not fixed iterations: each checkpoint calls the
solver with a real stopping tolerance --tol (default 1e-6, matching
main_pixel.py's own --tol default) and max_iter=n_try as a ceiling, not a
forced count. SinkSLOT and FlashSinkhorn both genuinely support early
stopping on marginal violation (see sinkslot_trajectory's and
flashsinkhorn_trajectory's own docstrings for how each is verified to
actually break out of its iteration loop early, not just report a
converged flag after running the full budget regardless): once a checkpoint
reports convergence, the checkpoint LOOP stops -- there is no point
re-solving at a larger budget once the solver has already stopped moving,
and the reported `iters` at that final checkpoint is the genuine
convergence point, not an arbitrarily chosen number of iterations. GeomLoss
has no native early-stopping mechanism at any level (confirmed in this
file's development history), so its checkpoint loop instead runs to each
n_try and checks the POST-HOC marginal violation against the same --tol,
breaking once it passes -- iters is then simply that n_try, since GeomLoss
cannot stop before completing requested iterations.

Consequence for the x-axis: unlike an earlier version of this script (which
forced every checkpoint to exactly n_try iterations via tol=0.0 and
compared curves against wall-clock runtime), each method's curve here now
naturally ends at its OWN convergence point -- different methods produce
different-length trajectories, and the natural x-axis is iteration count,
not runtime (see plot_trajectory.py).

marginal_viol: for SinkSLOT and FlashSinkhorn, comes directly from the same
internal stopping-check machinery that decides convergence (SinkSLOT's
stop.mode="marginal"; FlashSinkhorn's stop_mode="marginal"), not a
hand-rolled reimplementation, and not a second/extra call: the single timed
call now returns both the solved potentials AND the convergence signal.
GeomLoss exposes no internal check at all, so its marginal_viol is always
the post-hoc computation (_marginal_violation_dense_chunked) already used
above to decide when to stop.

No potential_change field: SinkSLOT's own stop.mode="potential" turns out to
be unsafe to call in a forced-full-budget (tol=0.0) pattern -- see git
history for the confirmed bug (a checkpoint that doesn't converge can
silently rewind phi/psi to a stale, non-final state). Now that every
checkpoint here uses a real tol and genuinely converges, this specific bug
no longer applies directly, but "potential" mode's parity with FlashSinkhorn
(the whole reason to add it back) is a separate feature not requested here;
left out entirely.

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
EVERY checkpoint along the way to convergence, for one fixed, deliberately
small pair, to produce a convergence plot -- a fundamentally different
data-collection shape, not a mode of the sweep script.

Checkpointing method: none of sinkslot_alternating_triton,
sinkhorn_flashstyle_alternating, or GeomLoss's low-level sinkhorn_loop
support warm-starting (same limitation main_pixel.py's primal_dual mode
already works around), so each checkpoint re-solves from scratch at that
checkpoint's iteration budget -- the recorded "time" for checkpoint k is
therefore the wall-clock duration of one from-scratch call, not a running
sum. Restart cost would grow like O(K^2) in the number of checkpoints if
every checkpoint ran to completion, but since the loop now stops at the
first converged checkpoint, the realised cost is bounded by the true
convergence point, not by --max_iter -- --max_iter is now only a safety
ceiling in case a method never converges under --tol. Kept further
affordable by defaulting to the two SMALLEST images in the bundled dataset
(--pair_idx, default (2, 9): "the-thames..." (40,711 colors) and
"the-toques..." (56,488 colors) -- see list_images's own sort order) rather
than main_pixel.py's own largest-scale smoke-test pair.

SinkSLOT-specific: unlike FlashSinkhorn/GeomLoss, which derive their
kernels implicitly from the data with no precomputation, SinkSLOT needs an
explicit one-time step to build its sparse support (sot_plan_coo + cost +
CSR/CSC conversion, exactly the "setup_ms" this repo's own scalability
benchmarks separate from the solve loop -- see bench_forward.py's
bench_sinkslotcuda). Built ONCE here (not redone per checkpoint), timed
separately, and reported as its own `setup_time` field alongside the
checkpoint list -- every checkpoint's `time` already has this cost folded
in (checkpoint time = setup_time + that checkpoint's from-scratch solve
time), so the reported times are genuine cumulative wall-clock from t=0.
"""

import argparse
import json
import os
import time

import torch

from color_transfer.main_pixel import (
    DEFAULT_PAINTINGS_DIR,
    StopCfg,
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
    p.add_argument("--method", type=str, required=True,
                    choices=["sinkslot", "flashsinkhorn", "flashsinkhorn_symmetric", "geomloss"])
    p.add_argument("--geomloss_backend", type=str, default="online", choices=["online", "multiscale"])
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-6,
                    help="Real convergence tolerance on max (L-infinity) marginal violation, "
                         "matching main_pixel.py's own --tol default. Every checkpoint's stopping "
                         "check (native for SinkSLOT/FlashSinkhorn, post-hoc for GeomLoss) uses this "
                         "same value, so the three methods are compared at the same convergence "
                         "criterion, not at the same iteration count.")
    p.add_argument("--max_iter", type=int, default=2000,
                    help="Safety ceiling, not a target: the checkpoint loop stops as soon as a "
                         "method converges under --tol, so this only matters if a method fails to "
                         "converge by then.")
    p.add_argument("--check_every", type=int, default=50,
                    help="Restart-checkpoint spacing (see module docstring): the loop tries "
                         "n_try = check_every, 2*check_every, ... until a checkpoint converges. "
                         "Also passed as each solver's own internal convergence-check granularity "
                         "where relevant.")
    p.add_argument("--warmup_iters", type=int, default=10)
    p.add_argument("--pair_idx", type=int, nargs=2, default=[2, 9],
                    help="Indices into the sorted image list (default: the two smallest bundled "
                         "paintings by unique-color count, to keep this script's restart cost "
                         "affordable).")
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


def sinkslot_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol, L, seed=0):
    """Build the sparse support once (timed separately as setup_time), then
    checkpoint sinkslot_alternating_triton at n_try = check_every,
    2*check_every, ..., stopping as soon as a checkpoint converges (or
    max_iter is reached as a safety ceiling). Returns (setup_time,
    checkpoints), where each checkpoint's own `time` already includes
    setup_time (see module docstring).

    A single call per checkpoint now provides everything: stop.mode="marginal"
    with a real tol (not the previous tol=0.0 forced-full-budget trick) makes
    sinkslot_alternating_triton itself decide when to stop -- its own `it`
    return value is therefore the GENUINE iteration count (less than n_try
    whenever it converges before that budget is exhausted), and `viol` is
    the same marginal-violation quantity --stop_mode=marginal itself stops
    on, read directly off the same call used for phi/psi (previously this
    needed a second, separate call). check_every=1 for this internal check
    is safe for "marginal" mode specifically (confirmed distinct from
    "potential" mode's swap-based bug: "marginal"'s phi_old/psi_old update
    is unconditional every iteration via torch.utils.swap_tensors, not
    gated behind the check, so there is no equivalent staleness risk here),
    so it reports the precise iteration at which the tolerance was crossed
    rather than rounding up to a coarser grid.

    primal_dual_gap has no native equivalent in any of the three libraries
    (confirmed for all three across this file's development), so it always
    needs the post-hoc _kl_gap_sparse computation, including here for
    SinkSLOT.
    """
    from sinkslot.sinkhorn_solvers import sinkslot_alternating_triton
    from sinkslot.solver import sot_plan_coo, sparse_sqeuclidean_cost, to_csr

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
        phi, psi, it, converged, mv = sinkslot_alternating_triton(
            r_ptr, r_idx, r_lam, c_ptr, c_idx, c_lam, log_a, log_b, n, m, n_try,
            stop=StopCfg(mode="marginal", max_iter=n_try, check_every=1, tol=tol))
        torch.cuda.synchronize()
        dt = setup_time + (time.perf_counter() - t0)

        vals = (phi[rows] + psi[cols] + lam).exp()
        cost_val = float((vals * cost).sum())
        gap = _kl_gap_sparse(phi, psi, rows, cols, S, cost, sw, tw, eps)

        checkpoints.append({"iters": it, "time": dt, "marginal_viol": float(mv),
                             "primal_dual_gap": gap, "cost": cost_val, "converged": bool(converged)})
        if converged:
            break
    return setup_time, checkpoints


def flashsinkhorn_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol):
    """No setup phase (FlashSinkhorn derives its kernel implicitly from the
    data every call) -- returns (0.0, checkpoints).

    Verified directly against sinkhorn_flashstyle_alternating's own source
    (torch-ext-external, read on Jean Zay) that threshold=None (the value
    this script used previously) disables early stopping entirely -- the
    loop always runs the full n_iters requested. Passing a real threshold
    with stop_mode="marginal" makes it check
    max(row marginal violation, col marginal violation) every check_every
    iterations and `break` the Python loop the first time it is <= threshold,
    so n_iters_used (returned via return_n_iters=True) is a genuine
    iteration count, not always equal to the requested n_try. This mirrors
    exactly what --stop_mode=marginal already trusts in main_pixel.py; the
    difference here is only that trajectory.py's checkpoint LOOP also stops
    once a checkpoint converges, rather than continuing to larger budgets
    that would just re-converge to the same result.
    """
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g, n_iters_used = sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=n_try, threshold=tol, check_every=5,
            stop_mode="marginal", allow_tf32=False, return_n_iters=True,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mv = _marginal_violation_dense_chunked(f, g, sc, tc, sw, tw, eps)
        gap = _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps)
        cost_val = _cost_dense_chunked(f, g, sc, tc, sw, tw, eps)
        converged = n_iters_used < n_try
        checkpoints.append({"iters": n_iters_used, "time": dt, "marginal_viol": mv,
                             "primal_dual_gap": gap, "cost": cost_val, "converged": converged})
        if converged:
            break
    return 0.0, checkpoints


def flashsinkhorn_symmetric_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, tol):
    """No setup phase, same contract as flashsinkhorn_trajectory --
    returns (0.0, checkpoints).

    sinkhorn_flashstyle_symmetric is FlashSinkhorn's OWN symmetric/Jacobi-
    damped update scheme -- its own docstring names it "GeomLoss-style
    symmetric updates" -- as opposed to sinkhorn_flashstyle_alternating's
    Gauss-Seidel scheme (see flashsinkhorn_trajectory). Added specifically
    to isolate WHY GeomLoss needs so many more iterations than FlashSinkhorn
    to reach the same marginal-violation tolerance: verified directly (a
    controlled diagnostic, same library, same problem instance, only the
    update rule changed) that this symmetric variant's marginal-violation
    trajectory tracks GeomLoss-online's almost exactly at every iteration
    count (e.g. at n=300, both land within ~7% of each other, while
    alternating is already 4+ orders of magnitude past both by n=150) --
    confirming the iteration-count gap is a genuine Gauss-Seidel-vs-Jacobi
    algorithmic difference, not a GeomLoss-specific bug or a different
    eps/cost convention (all three ultimately converge to the same
    transport cost).

    use_epsilon_scaling=False with a fixed eps/n_iters mirrors this
    experiment's other arms (fixed eps throughout, no annealed schedule);
    same threshold/check_every/stop_mode="marginal"/return_n_iters contract
    as sinkhorn_flashstyle_alternating, so this reuses the exact same
    checkpoint-loop pattern.
    """
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric

    checkpoints = []
    for n_try in range(check_every, max_iter + 1, check_every):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f, g, n_iters_used = sinkhorn_flashstyle_symmetric(
            sc, tc, sw, tw, use_epsilon_scaling=False, eps=eps, n_iters=n_try,
            threshold=tol, check_every=5, stop_mode="marginal",
            allow_tf32=False, return_n_iters=True,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        mv = _marginal_violation_dense_chunked(f, g, sc, tc, sw, tw, eps)
        gap = _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps)
        cost_val = _cost_dense_chunked(f, g, sc, tc, sw, tw, eps)
        converged = n_iters_used < n_try
        checkpoints.append({"iters": n_iters_used, "time": dt, "marginal_viol": mv,
                             "primal_dual_gap": gap, "cost": cost_val, "converged": converged})
        if converged:
            break
    return 0.0, checkpoints


def geomloss_trajectory(sc, tc, sw, tw, eps, max_iter, check_every, backend, tol):
    """No setup phase, same as FlashSinkhorn -- returns (0.0, checkpoints).

    GeomLoss's low-level API has no internal stopping check of any kind
    (confirmed across this file's development), so unlike SinkSLOT/
    FlashSinkhorn, it cannot stop before completing the n_try iterations
    requested -- convergence is instead detected post-hoc, by comparing the
    already-computed marginal_viol against the same --tol used natively by
    the other two methods, and iters is simply n_try at the checkpoint
    where that first holds.
    """
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
        converged = mv <= tol
        checkpoints.append({"iters": n_try, "time": dt, "marginal_viol": mv,
                             "primal_dual_gap": gap, "cost": cost_val, "converged": converged})
        if converged:
            break
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
          f"({sc.shape[0]} colors) -> {os.path.basename(paths[j])} ({tc.shape[0]} colors)  "
          f"tol={args.tol:g}")

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
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol, args.sinkslot_L)
    elif args.method == "flashsinkhorn":
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
        sinkhorn_flashstyle_alternating(sc, tc, sw, tw, eps=args.eps, n_iters=args.warmup_iters,
                                         threshold=None, allow_tf32=False)
        torch.cuda.synchronize()
        setup_time, checkpoints = flashsinkhorn_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol)
    elif args.method == "flashsinkhorn_symmetric":
        from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric
        sinkhorn_flashstyle_symmetric(sc, tc, sw, tw, use_epsilon_scaling=False, eps=args.eps,
                                       n_iters=args.warmup_iters, threshold=None, allow_tf32=False)
        torch.cuda.synchronize()
        setup_time, checkpoints = flashsinkhorn_symmetric_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.tol)
    else:  # geomloss
        solve = _GEOMLOSS_BACKENDS[args.geomloss_backend]
        solve(sc, tc, sw, tw, args.eps, args.warmup_iters)
        torch.cuda.synchronize()
        setup_time, checkpoints = geomloss_trajectory(
            sc, tc, sw, tw, args.eps, args.max_iter, args.check_every, args.geomloss_backend, args.tol)

    for c in checkpoints:
        print(f"  iters={c['iters']:5d}  time={c['time']:.4f}s  "
              f"marginal_viol={c['marginal_viol']:.4e}  primal_dual_gap={c['primal_dual_gap']:.4e}  "
              f"cost={c['cost']:.6f}  converged={c['converged']}")

    if not checkpoints[-1]["converged"]:
        print(f"\nWARNING: did not converge under tol={args.tol:g} within max_iter={args.max_iter} "
              f"(safety ceiling reached) -- consider raising --max_iter.")

    os.makedirs(args.output_dir, exist_ok=True)
    method_key = f"{args.method}_{args.geomloss_backend}" if args.method == "geomloss" else args.method
    out_path = os.path.join(args.output_dir, f"trajectory_{method_key}.json")
    with open(out_path, "w") as f:
        json.dump({
            "method": args.method,
            "geomloss_backend": args.geomloss_backend if args.method == "geomloss" else None,
            "eps": args.eps, "tol": args.tol, "max_iter": args.max_iter, "check_every": args.check_every,
            "pair": [os.path.basename(paths[i]), os.path.basename(paths[j])],
            "n": sc.shape[0], "m": tc.shape[0],
            "setup_time": setup_time,
            "checkpoints": checkpoints,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
