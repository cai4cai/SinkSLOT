"""Pixel-space color transfer: SinkSLOT vs. FlashSinkhorn, head to head.

    python -m color_transfer.main_pixel --output_dir DIR --method sinkslot
    python -m color_transfer.main_pixel --output_dir DIR --method flashsinkhorn

Requires a CUDA GPU (both solvers dispatch to fused Triton kernels with no
pure-torch fallback here). Install the FlashSinkhorn side of the comparison
via the `bench` extra (`pip install sinkslot[bench]`); SinkSLOT's own path
needs nothing beyond the core package.

Point cloud: the UNIQUE RGB colors of each image, not raw pixels and not a
palette/cluster reduction -- duplicate pixels collapse into one point, and
their counts become that point's mass (a/b), via pixels_and_weights() below.
At typical photo/painting resolutions this is 4-5 orders of magnitude more
points than a palette-based reduction, exercising the solvers at a much
larger, more irregularly-weighted scale than the gradient_flow experiment's
N=1000 synthetic clouds.

--paintings_dir should contain same-sized RGB images (.jpg/.jpeg/.png); every
ordered pair is run. color_transfer/paintings/ ships 12 Monet paintings
(public domain; Monet died 1926) for this purpose -- point --paintings_dir
elsewhere to use your own photos or paintings instead.

Three stopping modes (--stop_mode):
  * "potential": each library's own internal potential-change stop, trusted
    directly (SinkSLOT's own "potential" name; translated to FlashSinkhorn's
    "potential_linf", since the two libraries name the same rule
    differently). Watch out: SinkSLOT scales its check by eps before
    comparing to tol, FlashSinkhorn's potential_linf does not, so the same
    --tol is roughly 1/eps stricter for FlashSinkhorn than for SinkSLOT --
    can cause a spurious non-convergence at small eps if not accounted for.
  * "marginal": each library's own internal marginal-violation stop, trusted
    directly. Simplest to reason about, but only as trustworthy as each
    library's own internal check actually is.
  * "primal_dual": a genuine Fenchel primal-dual duality-gap stop, computed
    identically for both libraries in this script (_kl_gap_sparse /
    _kl_gap_dense_chunked below) rather than trusting either library's own
    internal signal. Both solvers regularize toward a non-uniform reference
    measure (SinkSLOT's sparse P^SOT; FlashSinkhorn's product coupling
    a⊗b), so the gap used here is the generalized-KL Fenchel dual for a
    normalized reference measure Q (sum(Q)=1):
        dual(f,g)   = <f,a> + <g,b> - eps * sum_ij Q_ij * exp((f_i+g_j-C_ij)/eps)
        primal(P)   = <C,P> + eps * sum_ij P_ij * (log(P_ij/Q_ij) - 1)
    with P the feasible plan obtained by row-then-column rescaling the
    (possibly infeasible) plan implied by (f,g) -- verified on a small CPU
    toy problem that the gap shrinks monotonically to ~0 as Sinkhorn
    iterates converge. Neither sinkslot_solve nor
    sinkhorn_flashstyle_alternating support warm-starting, so a genuine
    per-check early stop can't resume from a previous call's potentials:
    each check re-solves from scratch, at a geometrically-doubling iteration
    budget (check_every, 2*check_every, 4*check_every, ..., capped at
    max_iter) to bound the total restart overhead at roughly 2-3x a single
    full-budget solve rather than 5x+ under linear stepping. This is real,
    measurable overhead beyond the other two modes.

Fairness details, since this compares two independently-developed libraries
rather than benchmarking one method against itself:
  * TF32 is disabled globally (torch.backends.cuda.matmul.allow_tf32 = False
    etc.) before either solver touches the GPU. sinkslot_solve has no
    per-call TF32 argument -- it inherits whatever the global flag happens
    to be, which defaults to enabled on Ampere+ -- while FlashSinkhorn is
    passed allow_tf32=False explicitly; without the global override this
    would be a real, easy-to-miss precision/speed asymmetry.
  * An untimed warmup call (small n_iters, its own --warmup_check_every so
    any periodic-check-specific kernel still fires at least once) runs
    immediately before every timed call, so a shape's first-occurrence
    Triton JIT/autotune cost never leaks into that pair's measured time.
    Point cloud sizes vary several-fold across a real image set, and Triton
    kernels specialize per shape.
  * SinkSLOT and FlashSinkhorn should be run as separate OS processes (two
    invocations of this script with different --method, e.g. from a job
    script), not back-to-back in one process: `reset_peak_memory_stats`
    resets the counter but not the caching allocator, so one method's
    cached blocks would pollute the other's peak-memory reading if they
    shared a CUDA context.
  * `support_size` is recorded for both methods: SinkSLOT's plan lives only
    on its L-slice sparse support (support_size < n*m); FlashSinkhorn's is
    the full dense grid (support_size = n*m). The reported <C,P> is
    therefore a support-restricted-vs-dense comparison, not two estimates
    of the same object, and the output records this explicitly.
  * `hit_max_iters` is recorded separately from `converged`: a cost value
    from a run that hit the iteration cap isn't comparable to one that
    actually converged.

Resume-safe: if --output_dir already has a record file for this method,
already-completed (pair, eps) rows are skipped, so a run that gets
interrupted can be resubmitted to pick up where it left off.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import torch
from PIL import Image

from sinkslot.sinkhorn_solvers import sinkslot_solve
from sinkslot.solver import sparse_sqeuclidean_cost

DEFAULT_PAINTINGS_DIR = Path(__file__).parent / "paintings"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paintings_dir", type=str, default=str(DEFAULT_PAINTINGS_DIR),
                    help=f"Defaults to the 12 bundled Monet paintings ({DEFAULT_PAINTINGS_DIR}); "
                         "point elsewhere for your own same-sized RGB images.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--method", type=str, required=True, choices=["sinkslot", "flashsinkhorn"])
    p.add_argument("--eps_list", type=float, nargs="+", default=[0.01])
    p.add_argument("--sinkslot_L", type=int, default=100)
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument("--stop_mode", type=str, default="potential",
                    choices=["potential", "marginal", "primal_dual"],
                    help="'potential'/'marginal': shared user-facing stop-mode name, translated per "
                         "library at the call site (SinkSLOT's own native names; FlashSinkhorn maps "
                         "'potential'->'potential_linf'), trusting each library's own internal "
                         "converged/violation signal directly. 'primal_dual': genuine Fenchel "
                         "primal-dual duality-gap stopping rule, computed independently in this "
                         "script via geometric-doubling restarts (see module docstring) since "
                         "neither library exposes this natively or supports warm-starting.")
    p.add_argument("--tol", type=float, default=1e-6,
                    help="Convergence tolerance passed directly to each library's own internal "
                         "stop check (potential-change tolerance under --stop_mode=potential).")
    p.add_argument("--check_every", type=int, default=500)
    p.add_argument("--warmup_iters", type=int, default=10,
                    help="Untimed iterations run immediately before every timed call, to absorb "
                         "per-shape Triton JIT/autotune compilation out of the timing.")
    p.add_argument("--warmup_check_every", type=int, default=5,
                    help="check_every for the warmup call specifically (not the real run's "
                         "--check_every=500) so the periodic marginal-check code path actually "
                         "fires at least once during warmup, given warmup_iters defaults to 10.")
    p.add_argument("--max_pairs", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@dataclass
class StopCfg:
    mode: str = "potential"
    max_iter: int = 5000
    check_every: int = 500
    tol: float = 1e-6


# FlashSinkhorn names the potential-change stop rule "potential_linf"; SinkSLOT
# names it "potential". "marginal" is spelled the same in both.
_FLASH_STOP_MODE = {"potential": "potential_linf", "marginal": "marginal"}


def list_images(root):
    valid_ext = {".jpg", ".jpeg", ".png"}
    return [os.path.join(root, n) for n in sorted(os.listdir(root))
            if os.path.splitext(n)[1].lower() in valid_ext]


def pixels_and_weights(path, device, dtype):
    """Unique RGB pixel values with summed weights -- duplicate pixels combined."""
    img = Image.open(path).convert("RGB")
    raw = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(-1, 3).to(device)
    total = raw.shape[0]
    uniq, counts = torch.unique(raw, dim=0, return_counts=True)
    pixels = uniq.to(dtype) / 255.0
    weights = counts.to(dtype) / total
    return pixels, weights


def sinkslot_run(sc, tc, sw, tw, eps, L, seed, n_iters, stop):
    """One sinkslot_solve call. Returns (cost, iters, converged, internal_viol, support_size).

    Trusts sinkslot_solve's own internal converged/viol return values
    directly -- under stop_mode="potential" these are exactly
    (change < stop.tol, change) where change = eps * max(|dphi|, |dpsi|),
    computed inside the library's own convergence loop. No independent
    post-hoc check."""
    phi, psi, rows, cols, S, it, converged, viol = sinkslot_solve(
        sc, tc, sw, tw, eps=eps, L=L, seed=seed,
        n_iters=n_iters, stop_mode=stop.mode,
        stop_max_iter=n_iters, stop_tol=stop.tol, stop_check_every=stop.check_every,
    )
    cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
    log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
    vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
    cost_val = float((vals * cost).sum())
    converged_out = bool(converged) if converged is not None else None
    viol_out = float(viol) if viol is not None else None
    return cost_val, int(it), converged_out, viol_out, int(rows.shape[0])


def flashsinkhorn_run(sc, tc, sw, tw, eps, n_iters, stop, block_n=2048):
    """One sinkhorn_flashstyle_alternating call plus chunked cost computation
    (never materializes a dense (n,m) tensor). Returns
    (cost, iters, converged, internal_viol, support_size).

    sinkhorn_flashstyle_alternating exposes no converged flag or final
    violation/potential-change value of its own (only n_iters_used), so
    converged := n_iters_used < n_iters is used as the internal signal and
    internal_viol is always None here -- no independent post-hoc marginal
    check is computed."""
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

    n, m = sc.shape[0], tc.shape[0]
    flash_mode = _FLASH_STOP_MODE.get(stop.mode, stop.mode)
    f, g, n_iters_used = sinkhorn_flashstyle_alternating(
        sc, tc, sw, tw, eps=eps, n_iters=n_iters,
        stop_mode=flash_mode, threshold=stop.tol, check_every=stop.check_every,
        allow_tf32=False, return_n_iters=True,
    )
    cost_val = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]  # (blk, m)
        P_blk = sw[start:end, None] * tw[None, :] * torch.exp(
            (f[start:end, None] + g[None, :] - C_blk) / eps)
        cost_val += float((P_blk * C_blk).sum())
        del C_blk, P_blk
    converged = n_iters_used < n_iters
    return cost_val, int(n_iters_used), converged, None, int(n) * int(m)


def _kl_gap_sparse(phi, psi, rows, cols, S, cost, a, b, eps):
    """Fenchel primal-dual gap for SinkSLOT's <C,P> + eps*KL(P||P^SOT)
    objective, evaluated at (phi, psi), restricted to the sparse support
    (rows, cols, S). Returns a float clamped to >= 0 (the feasible-plan
    reconstruction is an approximate correction, so the raw value can come
    out very slightly negative near convergence).

    `phi`/`psi` are SinkSLOT's own dimensionless log-potentials (already in
    "f/eps" units -- see `vals` below, which matches
    `sinkslot_alternating_torch`'s own `(phi[rows]+psi[cols]+lam).exp()`
    convention), NOT same-units-as-cost potentials like FlashSinkhorn's
    `f`/`g` in `_kl_gap_dense_chunked`. The dual's linear term must be scaled
    by `eps` to match units with the rest of the formula -- omitting it
    (an earlier version of this function did) inflates `dual` by roughly
    `1/eps` relative to `primal`, so `primal - dual` never approaches 0 and
    the final `max(gap, 0.0)` silently floors the reported gap to exactly
    0.0 regardless of true convergence, verified against a CPU toy problem
    where the un-scaled version got stuck at a gap of exactly 0.0 even one
    iteration in (row-marginal violation 0.11), while this scaled version
    converges cleanly to ~1e-7 as Sinkhorn iterates converge."""
    tiny = torch.finfo(cost.dtype).tiny
    log_S = S.clamp_min(tiny).log()
    vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()

    dual = float(eps * (phi @ a + psi @ b) - eps * vals.sum())

    n, m = a.shape[0], b.shape[0]
    row_sum = torch.zeros(n, device=a.device, dtype=vals.dtype).index_add_(0, rows, vals).clamp_min(tiny)
    step1 = vals * (a[rows] / row_sum[rows])
    col_sum = torch.zeros(m, device=a.device, dtype=vals.dtype).index_add_(0, cols, step1).clamp_min(tiny)
    P_feas = step1 * (b[cols] / col_sum[cols])

    positive = P_feas > 0
    entropy = (P_feas[positive] * (P_feas[positive].clamp_min(tiny).log() - log_S[positive] - 1.0)).sum()
    primal = float((cost * P_feas).sum() + eps * entropy)

    return max(primal - dual, 0.0)


def _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n=2048):
    """Fenchel primal-dual gap for FlashSinkhorn's standard EOT objective
    <C,P> + eps*KL(P||a(x)b), evaluated at (f, g). Never materializes a
    dense (n,m) tensor: the implied plan is recomputed from (f,g) in three
    row-blocked passes (row sums; column sums of the row-rescaled
    intermediate; final feasible plan + primal objective), mirroring
    flashsinkhorn_run's own chunked cost computation."""
    n, m = sc.shape[0], tc.shape[0]
    tiny = torch.finfo(sc.dtype).tiny
    log_a = sw.clamp_min(tiny).log()
    log_b = tw.clamp_min(tiny).log()

    def plan_block(start, end):
        C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]
        P_blk = sw[start:end, None] * tw[None, :] * torch.exp(
            (f[start:end, None] + g[None, :] - C_blk) / eps)
        return C_blk, P_blk

    row_sum = torch.zeros(n, device=sc.device, dtype=sc.dtype)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        _, P_blk = plan_block(start, end)
        row_sum[start:end] = P_blk.sum(1)
        del P_blk
    dual = float(f @ sw + g @ tw - eps * row_sum.sum())
    row_sum = row_sum.clamp_min(tiny)

    col_sum = torch.zeros(m, device=sc.device, dtype=sc.dtype)
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        _, P_blk = plan_block(start, end)
        step1 = P_blk * (sw[start:end, None] / row_sum[start:end, None])
        col_sum += step1.sum(0)
        del P_blk, step1
    col_sum = col_sum.clamp_min(tiny)

    primal_lin = 0.0
    entropy = 0.0
    for start in range(0, n, block_n):
        end = min(start + block_n, n)
        C_blk, P_blk = plan_block(start, end)
        step1 = P_blk * (sw[start:end, None] / row_sum[start:end, None])
        P_feas = step1 * (tw[None, :] / col_sum[None, :])
        primal_lin += float((C_blk * P_feas).sum())
        positive = P_feas > 0
        log_Q_blk = log_a[start:end, None] + log_b[None, :]
        entropy += float((P_feas[positive] * (P_feas[positive].clamp_min(tiny).log()
                                               - log_Q_blk[positive] - 1.0)).sum())
        del C_blk, P_blk, step1, P_feas
    primal = primal_lin + eps * entropy

    return max(primal - dual, 0.0)


def sinkslot_run_primal_dual(sc, tc, sw, tw, eps, L, seed, max_iter, tol, check_every):
    """Restart-based early stopping on the primal-dual gap. sinkslot_solve
    has no warm-start parameter, so each check re-solves from scratch at a
    larger iteration budget (geometric doubling) rather than resuming --
    correct, but pays for the repeated early iterations (see module
    docstring). Returns (cost, iters, converged, gap, support_size)."""
    n_try = check_every
    last = None
    while True:
        phi, psi, rows, cols, S, it, _lib_converged, _lib_viol = sinkslot_solve(
            sc, tc, sw, tw, eps=eps, L=L, seed=seed,
            n_iters=n_try, stop_mode="fixed",
        )
        cost = sparse_sqeuclidean_cost(sc, tc, rows, cols)
        gap = _kl_gap_sparse(phi, psi, rows, cols, S, cost, sw, tw, eps)
        log_S = S.clamp_min(torch.finfo(S.dtype).tiny).log()
        vals = (phi[rows] + psi[cols] + log_S - cost / eps).exp()
        cost_val = float((vals * cost).sum())
        last = (cost_val, n_try, gap, int(rows.shape[0]))
        if gap <= tol or n_try >= max_iter:
            break
        n_try = min(n_try * 2, max_iter)
    cost_val, iters, gap, support_size = last
    return cost_val, iters, gap <= tol, gap, support_size


def flashsinkhorn_run_primal_dual(sc, tc, sw, tw, eps, max_iter, tol, check_every, block_n=2048):
    """Restart-based early stopping on the primal-dual gap, analogous to
    sinkslot_run_primal_dual. sinkhorn_flashstyle_alternating with
    threshold=None runs exactly n_iters with no internal check at all.
    Returns (cost, iters, converged, gap, support_size)."""
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating

    n = sc.shape[0]
    n_try = check_every
    last = None
    while True:
        f, g = sinkhorn_flashstyle_alternating(
            sc, tc, sw, tw, eps=eps, n_iters=n_try, threshold=None,
            allow_tf32=False,
        )
        gap = _kl_gap_dense_chunked(f, g, sc, tc, sw, tw, eps, block_n=block_n)
        cost_val = 0.0
        for start in range(0, n, block_n):
            end = min(start + block_n, n)
            C_blk = torch.cdist(sc[start:end][None], tc[None], p=2).pow(2)[0]
            P_blk = sw[start:end, None] * tw[None, :] * torch.exp(
                (f[start:end, None] + g[None, :] - C_blk) / eps)
            cost_val += float((P_blk * C_blk).sum())
            del C_blk, P_blk
        last = (cost_val, n_try, gap, n * tc.shape[0])
        if gap <= tol or n_try >= max_iter:
            break
        n_try = min(n_try * 2, max_iter)
    cost_val, iters, gap, support_size = last
    return cost_val, iters, gap <= tol, gap, support_size


def measure(fn, device):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    try:
        cost, iters, converged, viol, support_size = fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, None, None, None, "OOM"
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return cost, dt, peak_mb, iters, converged, viol, support_size


def load_existing(out_path):
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    return None


def already_done(records, eps, pair_idx):
    rec = records.get(str(eps), [])
    return any(r["pair"] == pair_idx and r["status"] != "OOM" for r in rec)


def main():
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU (both solvers dispatch to fused "
                            "Triton kernels with no pure-torch fallback here).")
    device = torch.device(args.device)
    dtype = torch.float32
    # SinkSLOT's matmuls (e.g. sot_plan_coo's random projections) have no
    # per-call tf32 argument -- they inherit this global flag, which Ampere+
    # defaults to enabled. FlashSinkhorn is passed allow_tf32=False per-call
    # below; without this, the two methods would run under different
    # precision/speed regimes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"pixel_records_{args.method}.json")
    existing = load_existing(out_path)
    if existing is not None:
        print(f"Resuming from existing {out_path}")

    paths = list_images(args.paintings_dir)
    print(f"Method: {args.method}  Images: {len(paths)}")

    cached = [pixels_and_weights(p, device, dtype) for p in paths]
    for p, (px, w) in zip(paths, cached):
        print(f"  {os.path.basename(p):60s} unique_pixels={px.shape[0]}")

    idx_pairs = list(permutations(range(len(paths)), 2))
    if args.max_pairs > 0:
        idx_pairs = idx_pairs[: args.max_pairs]
    P_pairs = len(idx_pairs)
    print(f"Ordered pairs: {P_pairs}")

    stop = StopCfg(mode=args.stop_mode, max_iter=args.max_iter, check_every=args.check_every, tol=args.tol)
    warmup_stop = StopCfg(mode=args.stop_mode, max_iter=args.warmup_iters,
                           check_every=args.warmup_check_every, tol=args.tol)
    records = existing["records"] if existing else {str(eps): [] for eps in args.eps_list}
    for eps in args.eps_list:
        records.setdefault(str(eps), [])

    for pair_idx, (i, j) in enumerate(idx_pairs, start=1):
        sc, sw = cached[i]
        tc, tw = cached[j]
        n, m = sc.shape[0], tc.shape[0]

        for eps in args.eps_list:
            if already_done(records, eps, pair_idx):
                continue

            print(f"[{pair_idx}/{P_pairs}] n={n} m={m} eps={eps} "
                  f"({os.path.basename(paths[i])} -> {os.path.basename(paths[j])})")

            # untimed warmup: absorb this shape's Triton JIT/autotune cost
            # before the timed call, using a tiny iteration budget. Under
            # primal_dual, warmup is a single plain fixed-iters call (no gap
            # logic) -- it only needs to touch the same kernels for JIT, not
            # produce a meaningful result.
            if args.stop_mode == "primal_dual":
                if args.method == "sinkslot":
                    warmup = lambda: sinkslot_solve(
                        sc, tc, sw, tw, eps=eps, L=args.sinkslot_L, seed=0,
                        n_iters=args.warmup_iters, stop_mode="fixed")
                    real = lambda: sinkslot_run_primal_dual(
                        sc, tc, sw, tw, eps, args.sinkslot_L, 0,
                        args.max_iter, args.tol, args.check_every)
                else:
                    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_alternating
                    warmup = lambda: sinkhorn_flashstyle_alternating(
                        sc, tc, sw, tw, eps=eps, n_iters=args.warmup_iters,
                        threshold=None, allow_tf32=False)
                    real = lambda: flashsinkhorn_run_primal_dual(
                        sc, tc, sw, tw, eps, args.max_iter, args.tol, args.check_every)
            elif args.method == "sinkslot":
                warmup = lambda: sinkslot_run(sc, tc, sw, tw, eps, args.sinkslot_L, 0, args.warmup_iters, warmup_stop)
                real = lambda: sinkslot_run(sc, tc, sw, tw, eps, args.sinkslot_L, 0, args.max_iter, stop)
            else:
                warmup = lambda: flashsinkhorn_run(sc, tc, sw, tw, eps, args.warmup_iters, warmup_stop)
                real = lambda: flashsinkhorn_run(sc, tc, sw, tw, eps, args.max_iter, stop)
            try:
                warmup()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            cost, dt, mem, iters, converged, viol, status_or_support = measure(real, device)
            if cost is None:
                status, support_size = "OOM", None
            else:
                status, support_size = "ok", status_or_support

            hit_max_iters = (iters is not None) and (iters >= args.max_iter)

            records[str(eps)].append({
                "pair": pair_idx, "n": n, "m": m,
                "cost": cost, "time": dt, "peak_mb": mem,
                "iters": iters, "internal_viol": viol,
                "support_size": support_size,
                "converged": converged, "hit_max_iters": hit_max_iters,
                "status": status,
            })

            print(f"    -> status={status} iters={iters} internal_viol={viol} "
                  f"converged={converged} hit_max_iters={hit_max_iters} time={dt}")

        # save incrementally -- long runs, don't lose progress on a crash/timeout
        with open(out_path, "w") as f:
            json.dump({"method": args.method, "num_images": len(paths), "num_pairs": P_pairs,
                       "sinkslot_L": args.sinkslot_L if args.method == "sinkslot" else None,
                       "stop_mode": args.stop_mode, "tol": args.tol, "max_iter": args.max_iter,
                       "warmup_iters": args.warmup_iters,
                       "records": records}, f, indent=2)

    print("\nSummary (excluding OOM):")
    for eps in args.eps_list:
        rec = records[str(eps)]
        attempted = [r for r in rec if r["status"] == "ok"]
        n_not_converged = sum(1 for r in attempted if not r["converged"])
        n_hit_cap = sum(1 for r in attempted if r["hit_max_iters"])
        oom_count = sum(1 for r in rec if r["status"] == "OOM")
        print(f"  eps={eps}  attempted={len(attempted)}/{len(rec)}  "
              f"not_converged={n_not_converged}  hit_max_iters={n_hit_cap}  OOM={oom_count}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
